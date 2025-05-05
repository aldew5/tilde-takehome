from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import torch

"""
UNTESTED implementation of Recursive KVMerger2.  

UPDATE: I switched to an easier cluster assignment (see "Improved cluster assignment") so the recursion is no longer strictly 
necessary (we can just choose the highest sim adj cluster). So this file is a bit of a hybrid right now.
"""


class RecursiveCache(Cache):
    """
    An implementation of KVMerger2 recursive in transformers' KV cache framework.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the short context window.
        max_length (`int`):
            The maximum cache length.
        eps (`float`): 
            partition similarity threshold
        k (`int`):
            How many heavy-hitters to exclude from merging
    """

    is_sliding = False
    
    # TODO: typing
    def __init__(self, max_length:int, window_length: int, eps: float) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.real_counts: List[torch.Tensor] = []

        self.max_length = max_length
        self.window_length = window_length
        self.eps = eps
        # TODO: set k correctly. 0 right now to test merging (merge everything) 
        self.k = 10
        

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        return self.window_length

    
    def _recursive_partition(self, rem_idxs, comp_key, num_clusters, eps) -> List[List[List[int]]]:
        """
        Returns num_clusters partitions of the remaining (not top k) indices for merging.
        """

        if len(rem_idxs) < num_clusters:
            return [[ [i] ] for i in rem_idxs] #+ [ [ [rem_idxs[-1]] ] ] * (num_clusters - len(rem_idxs))
        
        if not len(rem_idxs) or not num_clusters or eps < 0:
            return [[[]]]
        
        norms = comp_key[rem_idxs].norm(dim=-1)          
        topk = torch.topk(norms, k=num_clusters).indices
        # cluster centers
        clusters = {rem_idxs[i] : [rem_idxs[i]] for i in topk.tolist()}

        centers = sorted([ rem_idxs[i] for i in topk.tolist() ])
        # for non-centers, we will add to a cluster

        for idx in rem_idxs:
            if idx in centers: 
                continue

            lower = [c for c in centers if c < idx]
            upper = [c for c in centers if c > idx]

            #low_sim, high_sim = -1, -1
            sims = []
            # "bidirectional greedy": compare to both adj clusters
            if lower:
                c_lo = lower[-1]
                sims.append((c_lo, F.cosine_similarity( comp_key[idx:idx+1], comp_key[c_lo:c_lo+1], dim=-1 ).item()))
            if upper:
                c_hi = upper[0]
                sims.append((c_hi, F.cosine_similarity( comp_key[idx:idx+1], comp_key[c_hi:c_hi+1], dim=-1 ).item()))
            
            # can add to some cluster
            center, best_sim = max(sims, key=lambda x: x[1])
            if best_sim > eps:
                clusters[center].append(idx)
                #print("HERE")

        assigned = set()
        for v in clusters.values():
            assigned.update(v)

        leftover = [i for i in rem_idxs if i not in assigned]
        clusters_rec = self._recursive_partition(
            leftover, comp_key, num_clusters, eps - 0.1)

        return [clusters[c] for c in centers] + clusters_rec

    #TODO
    def _partition_value_aware(self, 
                layer_idx: int) -> List[List[List[int]]]:
        pass

    def _gaussian_kernel_merge(self, keys: torch.Tensor, values: torch.Tensor, 
                damping=1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian kernel merge as in KVMerger. TODO: Guassian kernel with values

        keys: [n, D] keys in one cluster
        values: [n, D] corresponding values
        returns (kM, vM) each [head_dim]
        """
        # pick pivot based on highest cos sim
        keys_norm = F.normalize(keys, dim=-1)  
        cos_sim = keys_norm @ keys_norm.transpose(0,1)   
        agg = cos_sim.sum(dim=1)          
        pivot = int(agg.argmax().item()) 

        # dist to pivot squared
        diffs = keys - keys[pivot: pivot+1]   
        d2 = (diffs**2).sum(dim=1)    
        g0 = torch.exp(-0.5 * d2) 

        # sigma = sum(g0)/(root(2)*n)
        sigma = g0.sum() / (torch.sqrt(torch.tensor(2.0, device=keys.device)) * keys.size(0) + damping)

        g = torch.exp(-d2.div(2 * sigma*sigma + damping)) 
        w = g / (g.sum() + damping)

        kM = (w.unsqueeze(1) * keys).sum(dim=0) 
        vM = (w.unsqueeze(1) * values).sum(dim=0)  
        return kM, vM

    
    def merge_with_recursive(
        self,
        comp_key: torch.Tensor,            
        comp_value: torch.Tensor,          
        rem_idxs: List[List[List[int]]],  
        num_clusters: int,
    ) -> (torch.Tensor, torch.Tensor):
        """
        generate merged reps using recursive partitioning.
        """

        bsz, heads, seq_length, head_dim = comp_key.shape
        merged_k, merged_v = [], []
        for b in range(bsz):
            bh_k, bh_v = [], []
            for h in range(heads):
                idxs = rem_idxs[b][h]
                # call the recursive clustering
                clusters = self._recursive_partition(
                    rem_idxs=idxs,
                    comp_key=comp_key[b, h],   
                    num_clusters=num_clusters,
                    eps=self.eps
                )

                # merge by guassian kernel weights
                reps_k, reps_v = [], []
                for cluster in clusters:
                    cluster_idxs = torch.tensor(cluster, device=comp_key.device, dtype=torch.long)
                    keys = comp_key[b, h, cluster_idxs]     
                    vals = comp_value[b, h, cluster_idxs] 
                    kM, vM = self._gaussian_kernel_merge(keys, vals)
                    reps_k.append(kM)
                    reps_v.append(vM)

                #print(reps_k, num_clusters)
                # should return exactly compression, unless list was smaller
                assert len(reps_k) <= num_clusters

                if len(reps_k):
                    while len(reps_k) < num_clusters:
                        reps_k.append(reps_k[-1].clone())
                        reps_v.append(reps_v[-1].clone())
                
                    bh_k.append(torch.stack(reps_k, dim=0))
                    bh_v.append(torch.stack(reps_v, dim=0))

            if len(bh_k):
                merged_k.append(torch.stack(bh_k, dim=0))
                merged_v.append(torch.stack(bh_v, dim=0))

        if len(merged_k):
            return torch.stack(merged_k, dim=0), torch.stack(merged_v, dim=0)
        # may be nothing to merge in early runs when cache is small
        else:
            return torch.tensor([]), torch.tensor([])


    def update(
        self,
        key_states: torch.Tensor,     
        value_states: torch.Tensor,   
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Implements Cache update, calling recursive merge on non-top k most important keys.
        """

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states.clone())
            self.value_cache.append(value_states.clone())
        else:
            self.key_cache[layer_idx] = torch.cat(
                (self.key_cache[layer_idx], key_states), dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value_states), dim=2
            )
        
        K_full = self.key_cache[layer_idx]      
        V_full = self.value_cache[layer_idx]
        W = self.window_length
        bsz, heads, seq_len, _ = K_full.shape

        comp_K = K_full[..., : seq_len - W, :]  
        #comp_V = V_full[..., : seq_len - W, :]
        budget = seq_len - W

        if budget <= 0:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # topk by norm for keys to be compressed
        norms  = comp_K.norm(dim=-1)                  
        _, topk_idx= norms.sort(dim=-1, descending=True) 
        topk_idx = topk_idx[..., : (self.max_length - W)]

        all_idx = torch.arange(budget, device=comp_K.device)               
        all_idx = all_idx[None, None,:].expand(bsz, heads, budget)                
        mask_topk = torch.zeros(bsz,heads, budget, dtype=torch.bool, device=comp_K.device)
        mask_topk.scatter_(2, topk_idx, True)                         

        rem_idxs  = [
            [ all_idx[b,h, ~mask_topk[b,h]].tolist() for h in range(heads) ]
            for b in range(bsz)
        ]
        #print("HERE", rem_idxs)

        num_clusters = self.max_length - (self.window_length + self.k)

        merged_old_K, merged_old_V = self.merge_with_recursive(
            comp_key=self.key_cache[layer_idx][..., :self.max_length-self.window_length, :],
            comp_value=self.value_cache[layer_idx][..., :self.max_length-self.window_length, :],
            rem_idxs=rem_idxs,
            num_clusters=num_clusters,
        )


        K_win = self.key_cache[layer_idx][..., self.max_length-self.window_length:, :]
        V_win = self.value_cache[layer_idx][..., self.max_length-self.window_length:, :]

         # TODO: another ugly fix
        device = K_win.device
        merged_old_K = merged_old_K.to(device)
        merged_old_V = merged_old_V.to(device)


        self.key_cache[layer_idx] = torch.cat((merged_old_K, K_win), dim=2)
        self.value_cache[layer_idx] = torch.cat((merged_old_V, V_win), dim=2)
        
        # TODO: ugly fix for the moment
        dtype = key_states.dtype
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(dtype)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(dtype)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

