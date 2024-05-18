#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
    "PagedAttention V2.");

 // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "convert_fp8",
    &convert_fp8,
    "Convert the key and value cache to fp8 data type");

//   // Cuda utils
//   pybind11::module cuda_utils = m.def_submodule("cuda_utils", "vLLM cuda utils");
//   cuda_utils.def(
//     "get_device_attribute",
//     &get_device_attribute,
//     "Gets the specified device attribute.");

//   cuda_utils.def(
//     "get_max_shared_memory_per_block_device_attribute",
//     &get_max_shared_memory_per_block_device_attribute,
//     "Gets the maximum shared memory per block device attribute.");

// #ifndef USE_ROCM
//   // Custom all-reduce kernels
//   pybind11::module custom_ar = m.def_submodule("custom_ar", "custom allreduce");
//   custom_ar.def("init_custom_ar", &init_custom_ar, "init_custom_ar");
//   custom_ar.def("should_custom_ar", &should_custom_ar, "should_custom_ar");
//   custom_ar.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg");
//   custom_ar.def("all_reduce_unreg", &all_reduce_unreg, "all_reduce_unreg");
//   custom_ar.def("dispose", &dispose, "dispose");
//   custom_ar.def("meta_size", &meta_size, "meta_size");
//   custom_ar.def("register_buffer", &register_buffer, "register_buffer");
//   custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
//                 "get_graph_buffer_ipc_meta");
//   custom_ar.def("register_graph_buffers", &register_graph_buffers,
//                 "register_graph_buffers");
// #endif

}
