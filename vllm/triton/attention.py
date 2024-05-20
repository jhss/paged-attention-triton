import triton
import triton.language as tl

#import os
#os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def naive_attention_kernel(output_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
                           num_kv_heads, scale, block_tables, context_lens, 
                           BLOCK_SIZE: tl.constexpr, max_context_len, alibi_slopes,
                           kv_scale, HEAD_BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                           KEY_SEQ_LEN: tl.constexpr,
                           query_seq_stride, query_head_stride,
                           key_seq_stride, key_head_stride):
    
    # [num_seqs, num_heads, head_dim]
    # For each block, 
    head_id = tl.program_id(0)
    query_seq_id = tl.program_id(1)
    #context_id = tl.program_id(2)

    offset_head  = tl.arange(0, HEAD_BLOCK_SIZE)
    offset_query = offset_head + query_seq_id * query_seq_stride + head_id * query_head_stride
    offset_key   = offset_head + head_id * key_head_stride

    query_ptrs = query_ptr + offset_query
    output_ptrs = output_ptr + offset_query

    #print("[DEBUG] key_seq_len: ", KEY_SEQ_LEN)
    qk = tl.zeros((KEY_SEQ_LEN,), dtype=tl.float32)

    for head_part_idx in range(0, HEAD_DIM, HEAD_BLOCK_SIZE):
        query = tl.load(query_ptrs, mask=offset_head < HEAD_DIM - head_part_idx*HEAD_BLOCK_SIZE, other=0.0)
        tl.store(output_ptrs, query, mask=offset_head < HEAD_DIM - head_part_idx*HEAD_BLOCK_SIZE)
        
        output_ptrs += HEAD_BLOCK_SIZE * query_head_stride

        key_ptrs = key_cache_ptr + offset_key
        for key_seq_idx in range(0, KEY_SEQ_LEN):
            key = tl.load(key_ptrs, mask=offset_head < HEAD_DIM - head_part_idx*HEAD_BLOCK_SIZE, other=0.0)
            #qk[key_seq_idx] = tl.sum(query*key)
            tl.sum(query*key)
            key_ptrs += key_seq_stride

        query_ptrs += HEAD_BLOCK_SIZE * query_head_stride
        offset_key += HEAD_BLOCK_SIZE * key_head_stride
    # 

    # for head_part_idx in range(0, head_dim, HEAD_BLOCK_SIZE):
    #     # load query [1, 1, HEAD_BLOCK_SIZE]
    #     query = tl.load(query_ptrs, mask = head_dim - head_part_idx*HEAD_BLOCK_SIZE, other=0.0) # TODO: mask
        
    #     for key_seq_idx in range(0, key_seq_len):
    #         key = tl.load(key_ptrs, mask = , other=0.0) # TODO: mask
    #         qk[key_seq_idx] += tl.sum(tl.mul(query, key))
    #         key_ptrs += stride_seq

    #     query_ptrs += HEAD_BLOCK_SIZE * stride_head
    #     key_ptrs   += HEAD_BLOCK_SIZE * stride_key

    # [1, key_seq_len]
    # softmax
    # attention_logits = tl.fdiv(qk, head_dim)
    # attention_logits_max = tl.max(attention_logits, axis=0)
    # exp_qk = tl.exp(attention_logits - attention_logits_max)
    # exp_sum = tl.sum(exp_qk, axis=0)
    # attention_weights = tl.fdiv(exp_qk, exp_sum)

    # attention_scores = tl.zeros((head_dim,))
    # for head_part_idx in range(0, head_dim, HEAD_BLOCK_SIZE):
    #     value = tl.load(value_ptrs, mask=, other=0.0)
    #     # TODO: is it available using tl.dot?
    #     attention_scores[head_part_idx*HEAD_BLOCK_SIZE:(head_part_idx+1)*HEAD_BLOCK_SIZE] = tl.dot(attention_scores, value)
    #     value_ptrs += HEAD_BLOCK_SIZE * stride_value

    # tl.store(output_ptr, attention_scores)

