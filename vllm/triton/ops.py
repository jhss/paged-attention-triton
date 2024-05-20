from typing import Dict, Optional

import torch
from .attention import *

def _naive_attention_triton(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:

    query_seq, num_heads, head_dim = query.shape
    key_seq, _, _ = key_cache.shape

    # [FIXME] adjust HEAD_BLOCK_SIZE ?
    HEAD_BLOCK_SIZE = 128
    
    # [TODO] kv_cache_dtype?
    naive_attention_kernel[(num_heads, query_seq)](
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes,
        kv_scale, HEAD_BLOCK_SIZE, head_dim,
        key_seq,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1)
    )

    #print("Is Same: ", torch.allclose(query, out, rtol=1e-5, atol=1e-5))