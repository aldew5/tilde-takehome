from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple, Union
import torch



class AverageCache(Cache):
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
        Returns a greedy partition of the cache object as a list of indices, as in KVMerger (Wang et al., 2024).
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
        
        # To get similarity between key i and key j:
        # cos_sim[..., i, j] gives similarity between key i and key j
        # For example, cos_sim[0, 0, 5, 10] gives similarity between key 5 and key 10
        # in the first batch and first head
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
                window_indices = list(range(split_point, key_length))
                partition_idx[b][h].append(window_indices)

        
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
        Merges the clusters and updates KV cache
        """
    
        if len(self.key_cache) <= layer_idx:
            # need to init layer in cache
            self.key_cache.append(key_states.clone())
            self.value_cache.append(value_states.clone())
        else:
            # old cache with new key-value
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=2)

        clusters = self._partition(layer_idx)  # List[B][H] of List[List[int]]

        # 3) Retrieve the full, current cache tensors
        orig_keys   = self.key_cache[layer_idx]   # [B, H, L, D]
        orig_values = self.value_cache[layer_idx] # [B, H, L, D]
        B, H, L, D  = orig_keys.shape

        new_keys = []
        new_values = []
        for b in range(B):
            # determine padding length = max clusters across heads for this batch item
            max_clusters = max(len(clusters[b][h]) for h in range(H))

            bh_keys = []
            bh_vals = []
            for h in range(H):
                reps_k = []
                reps_v = []
                for cluster_idxs in clusters[b][h]:
                    idx = torch.tensor(cluster_idxs, device=orig_keys.device, dtype=torch.long)
                    # gather and average
                    k_rep = orig_keys[b, h, idx, :].mean(dim=0)   # [D]
                    v_rep = orig_values[b, h, idx, :].mean(dim=0) # [D]
                    reps_k.append(k_rep)
                    reps_v.append(v_rep)

                # pad each head’s reps up to max_clusters (repeat last)
                if len(reps_k) < max_clusters:
                    last_k = reps_k[-1]
                    last_v = reps_v[-1]
                    reps_k += [last_k] * (max_clusters - len(reps_k))
                    reps_v += [last_v] * (max_clusters - len(reps_v))

                # stack to [max_clusters, D]
                bh_keys.append(torch.stack(reps_k, dim=0))
                bh_vals.append(torch.stack(reps_v, dim=0))

            # now stack heads → [H, max_clusters, D]
            new_keys.append(torch.stack(bh_keys, dim=0))
            new_values.append(torch.stack(bh_vals, dim=0))

        # 4) Final stack over batch → [B, H, max_clusters, D]
        merged_keys   = torch.stack(new_keys,   dim=0)
        merged_values = torch.stack(new_values, dim=0)

        # 5) Overwrite cache and return
        self.key_cache[layer_idx]   = merged_keys
        self.value_cache[layer_idx] = merged_values
        return merged_keys, merged_values
