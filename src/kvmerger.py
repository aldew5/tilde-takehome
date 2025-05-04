from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple, Union
import torch



class KVMerger(Cache):
    """
    An implementation of KNorm filtering in transformers' KV cache framework.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the short context window.
        max_length (`int`):
            The maximum cache length.
    """

    is_sliding = False

    def __init__(self, max_length:int, window_length: int, eps: float) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.max_length = max_length
        self.window_length = window_length
        self.eps = eps

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
    
    def _partition(self,
        layer_idx: int) -> List[List[List[int]]]:
        """
        Returns a greedy partition of the cache object *ignoring window tokens*
        as a list of indices, as in KVMerger (Wang et al., 2024).
        Retains the KV states whose corresponding aggregated attention scores fall within
        the top-k range, and the current window.
        """
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        if key_cache.shape[-2] <= self.max_length:
            L = key_cache.shape[-2]
            B, H = key_cache.shape[0], key_cache.shape[1]
            clusters = [
                [  # for each head, make each index its own cluster
                    [[i] for i in range(L)]
                    for _ in range(H)
                ]
                for _ in range(B)]
            return clusters

        key_length = key_cache.shape[-2]
        
        comp_key, window_key = key_cache[..., :key_length-self.window_length, :], key_cache[..., key_length-self.window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-self.window_length, :], value_cache[..., key_length-self.window_length:,:]

        proj_key = comp_key.norm(dim=-1)
        key_idx = (-proj_key).topk(self.max_length - self.window_length, -1).indices.sort().values
        key_idx = key_idx[..., None].repeat(1, 1, 1, key_cache.shape[-1])

        # indices not included in top k for partitioning
        all_idx = torch.arange(proj_key.shape[-1], device=proj_key.device)
        all_idx = all_idx[None, None, :].expand(proj_key.shape[0], proj_key.shape[1], -1)
        rem_idx = torch.where(~torch.isin(all_idx, key_idx[..., 0]))[2]

        # compute cos similarity matrix
        #[bsz, heads, rem_len, head_dim]
        rem_keys = comp_key[..., rem_idx, :] 
        rem_keys_norm = torch.nn.functional.normalize(rem_keys, dim=-1)
        # [bsz, heads, rem_len, rem_len]
        cos_sim = torch.matmul(rem_keys_norm, rem_keys_norm.transpose(-2, -1))  
        
        # NOTE: cos_sim[..., i, j]  = sim(k_i, k_j)
        bsz, num_heads, rem_len, head_dim = rem_keys.shape
        partition_idx = [[[] for _ in range(num_heads)] for _ in range(bsz)]
        split_point = key_length - self.window_length

        rem_lists = [
            [
                [i for i in range(split_point) if i not in key_idx[b,h,:,0].tolist()]
                for h in range(num_heads)
            ]
            for b in range(bsz)
        ]

        for b in range(bsz):
            for h in range(num_heads):
                # merge starting from newest token
                rem_list = rem_lists[b][h]
                if len(rem_list) < 2:
                    continue

                i = len(rem_list) - 1
                while i >= 0:
                    # init new cluster
                    # NOTE: make sure to use absolute pos
                    cluster_idxs = [ rem_list[i] ]
                    j = i - 1
                    while j >= 0:
                        sim = (cos_sim[b,h,i] * cos_sim[b,h,j]).sum()
                        if sim > self.eps:
                            cluster_idxs.append(rem_list[j])
                            i = j
                            j = i-1
                        else:
                            break
                    partition_idx[b][h].append(cluster_idxs)
                    i -= 1

                # add the window for batch b and head h to partition_idx (absolute pos)
                # NOTE: we will add window in update
                #window_indices = list(range(split_point, key_length))
                #partition_idx[b][h].append(window_indices)

        
        # append top k indxs
        for b in range(bsz):
            for h in range(num_heads):
                top_k_indices = key_idx[b, h, :, 0].tolist()
                partition_idx[b][h].append(top_k_indices)

        return partition_idx


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merges the clusters and updates KV cache. partition function incorrect, padding to fix.
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

        # assume correct partition, no padding necessary
        clusters = self._partition(layer_idx)  

        # [B, H, L, D]
        K_full = self.key_cache[layer_idx]   
        V_full = self.value_cache[layer_idx] 
        B, H, L, D = K_full.shape

        W = self.window_length
        split = L - W
        K_win = K_full[..., split:, :]  
        V_win = V_full[..., split:, :]

        old_budget = self.max_length - W
        merged_K, merged_V = [], [] 
        for b in range(B):
            bh_K, bh_V = [], []
            for h in range(H):
                reps_k, reps_v = [], []
                for idxs in clusters[b][h]:
                    idx = torch.tensor(idxs, device=K_full.device, dtype=torch.long)
                    Kc = K_full[b, h, idx, :]  
                    Vc = V_full[b, h, idx, :]   
                    n  = Kc.size(0)

                    Kn = F.normalize(Kc, dim=-1)       
                    simm = Kn @ Kn.transpose(0,1)         
                    agg = simm.sum(dim=1)                 
                    p = int(agg.argmax().item())      

                    diffs = Kc - Kc[p:p+1]                
                    d2 = (diffs * diffs).sum(dim=1)    

                    g0 = torch.exp(-0.5 * d2)        
                    sigma = g0.sum() / (math.sqrt(2) * n + 1e-6)
                    g = torch.exp(-d2.div(2 * sigma*sigma + 1e-6)) 
                    w = g / (g.sum() + 1e-6)                       

                    kM = (w.unsqueeze(1) * Kc).sum(dim=0) 
                    vM = (w.unsqueeze(1) * Vc).sum(dim=0) 
                    reps_k.append(kM)
                    reps_v.append(vM)

                # stack all cluster reps for this head
                # NOTE: length should equal old_budget so no padding
                bh_K.append(torch.stack(reps_k, dim=0))  
                bh_V.append(torch.stack(reps_v, dim=0))

            merged_K.append(torch.stack(bh_K, dim=0))  
            merged_V.append(torch.stack(bh_V, dim=0))

        K_old = torch.stack(merged_K, dim=0) 
        V_old = torch.stack(merged_V, dim=0)
        self.key_cache[layer_idx] = torch.cat((K_old, K_win), dim=2)
        self.value_cache[layer_idx] = torch.cat((V_old, V_win), dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]