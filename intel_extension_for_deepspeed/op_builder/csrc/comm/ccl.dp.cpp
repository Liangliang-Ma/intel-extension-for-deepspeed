// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <oneapi/ccl.hpp>

#include <CL/sycl.hpp>
#include <ATen/record_function.h>
#include <ipex.h>
#include <CL/sycl.hpp>
#include <time.h>

#include <aten/core/Device.h>
#include <aten/core/Stream.h>
// #include <runtime/Context.h>
#include <runtime/Device.h>




using namespace sycl;
using namespace xpu;

at::Device ds_device("xpu");
ccl::stream init_stream(at::Device device)
{
    c10::impl::VirtualGuardImpl impl(device.type());
    c10::Stream _stream = impl.getStreamFromGlobalPool(device, /*isHighPriority=*/false);
    sycl::queue ds_queue = xpu::get_queue_from_stream(_stream);
    return ccl::create_stream(ds_queue);
}
ccl::stream ds_stream = init_stream(ds_device);

// Communicatiooon settings
int world_rank = -1;
int world_size = -1;

std::set<int> _comm_ids;
std::set<int> _colors;
ccl::vector_class<ccl::communicator> _ccl_comms;

ccl::communicator& _get_comm_from_group() { return _ccl_comms[0]; }
ccl::communicator& _get_comm_from_group(py::object group) { return _ccl_comms[0]; }

#define CCLCHECK(cmd) \
    do {                                                             \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (ccl::exception& e) {                                      \
      e.what();                                                      \
      throw e;                                                       \
    }                                                                \
  }while(0)

#define KVS_CREATE_SUCCESS 0
#define KVS_CREATE_FAILURE -1

#define CCL_KERNEL_SUBMIT(cmd, q)                              \
({                                                             \
    sycl::event start_evt;                                     \
    start_evt = q.ext_oneapi_submit_barrier();                 \
    CCLCHECK(cmd);                                             \
    sycl::event end_evt;                                       \
    end_evt = q.ext_oneapi_submit_barrier();                   \
})
              
bool is_initialized = 0;

ccl::shared_ptr_class<ccl::kvs> kvs;

bool all_ranks_local_p = false;

void initialize(int size, int rank, torch::Tensor& kvs_data)
{
    torch::Tensor kvs_data_cpu = kvs_data.to(torch::kCPU);
    if (is_initialized) return;

    // Check whether all ranks is on the same physical machine.
    // If true, we will use an SHM based low latency allreduce

    int ws = std::stoi(std::getenv("WORLD_SIZE"));
    int ls = std::stoi(std::getenv("LOCAL_SIZE"));

    if (ws >= 1 && ws == ls) {
        all_ranks_local_p = true;
    }

    world_size = size;
    world_rank = rank;
    is_initialized = 1;

    ccl::kvs::address_type main_addr;

    if (rank != 0) {
        memcpy(main_addr.data(), kvs_data_cpu.data_ptr(), main_addr.size());
        kvs = ccl::create_kvs(main_addr);
    }
    
    ds_device = "xpu:" + std::to_string(rank);
    ds_stream = init_stream(ds_device);
    
    c10::impl::VirtualGuardImpl impl(ds_device.type());
    c10::Stream _stream = impl.getStreamFromGlobalPool(ds_device, /*isHighPriority=*/false);

    auto q = xpu::get_queue_from_stream(_stream);
    auto ctx = ccl::create_context(q.get_context());
    ccl::device _device = ccl::create_device(q.get_device());

    ccl::vector_class<ccl::pair_class<int, cl::sycl::device>> devs_rank;
    auto sycl_dev = xpu::dpcpp::dpcppGetRawDevice(ds_device.index());
    devs_rank.emplace_back(rank, sycl_dev);

    // Create ccl::communicators
    // auto ds_comms = ccl::create_communicators(size, devs_rank, ctx, kvs);
    _ccl_comms.emplace_back(ccl::create_communicator(size, rank, _device, ctx, kvs));
}

/*
    rank == 0: create main kvs and return its address
    rank == else: return an empty address
*/
std::vector<uint8_t> get_kvs_addr(int rank)
{
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        ccl::kvs::address_type main_addr = kvs->get_address();
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    } else {
        ccl::kvs::address_type main_addr;
        auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
        return ccl_kvs_addr;
    }
}

int get_rank(int group = 0) { return world_rank; }

int get_world_size(int group = 0) { return world_size; }

// Find the next ordered, unique value to a set. E.g. <0,1,2,7> --> 3
int next_unique_val(std::set<int> s)
{
    std::set<int>::iterator itr;
    // Base case. Add 0 to start of set.
    if (s.empty() || *s.begin() != 0) {
        return 0;
        // second base case where s = {0} (the case of s = {n != 0} is caught above)
    } else if (s.size() == 1) {
        return 1;
    } else {
        int prev_val = *s.begin();
        for (itr = std::next(s.begin()); itr != s.end(); itr++) {
            if (*itr != prev_val + 1) { return prev_val + 1; }
            prev_val = *itr;
        }
        return *(s.end()) + 1;
    }
}

py::object new_group(std::vector<int> ranks)
{
    int comm_id = next_unique_val(_comm_ids);
    int color = next_unique_val(_colors);
    std::cout << "RANK: " << get_rank() << " COMM_ID: " << comm_id << " COLOR: " << color
              << std::endl;
}

ccl::datatype get_ccl_datatype(c10::ScalarType type)
{
    ccl::datatype ccl_type;
    switch (type) {
        case c10::ScalarType::Int: ccl_type = ccl::datatype::int32; break;
        case c10::ScalarType::Long: ccl_type = ccl::datatype::int64; break;
        case c10::ScalarType::Float: ccl_type = ccl::datatype::float32; break;
        case c10::ScalarType::Double: ccl_type = ccl::datatype::float64; break;
        case c10::ScalarType::BFloat16: ccl_type = ccl::datatype::bfloat16; break;
        case c10::ScalarType::Half: ccl_type = ccl::datatype::float16; break;
        default: ccl_type = ccl::datatype::int8;
    }
    return ccl_type;
}

ccl::reduction get_ccl_reduce_op(py::object op, at::Tensor& input)
{
    py::object ReduceOp = py::module_::import("deepspeed.comm").attr("ReduceOp");
    if (!py::isinstance(op, ReduceOp)) {
        throw std::runtime_error("Error: Op must be of type ReduceOp");
    }

    int op_val = py::int_(op.attr("value"));
    ccl::reduction ccl_op;

    if (input.scalar_type() == at::kBool) {
        if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
            // For bool tensors, map sum to max, which both represent a bitwise or.
            // This is to prevent overflow issues with sum, since we use uint8 to
            // represent a bool (see cclDataType mapping).
            ccl_op = ccl::reduction::max;
        } else if (op_val == (int)py::int_(ReduceOp.attr("AVG").attr("value"))) {
            throw std::runtime_error("Error: For bool tensors, op must be of type ReduceOp");
        }
    }

    if (op_val == (int)py::int_(ReduceOp.attr("SUM").attr("value"))) {
        ccl_op = ccl::reduction::sum;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MIN").attr("value"))) {
        ccl_op = ccl::reduction::min;
    } else if (op_val == (int)py::int_(ReduceOp.attr("MAX").attr("value"))) {
        ccl_op = ccl::reduction::max;
    } else if (op_val == (int)py::int_(ReduceOp.attr("PRODUCT").attr("value"))) {
        ccl_op = ccl::reduction::prod;
    } else {
        throw std::runtime_error("Error: Unrecognized ReduceOp type");
    }
    return ccl_op;
}

void broadcast(torch::Tensor& data, int src, py::object group, bool async_op)
{
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::broadcast(data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            src,
                            _get_comm_from_group(group),
                            ds_stream), ds_stream.get_native());
}

void all_gather(std::vector<torch::Tensor>& vec_data_out, torch::Tensor& data, py::object group, bool async_op)
{
    std::vector<size_t> recvCounts(vec_data_out.size(), data.numel());
    std::vector<void*> recvBufs;
    std::transform(vec_data_out.begin(),
                   vec_data_out.end(),
                   std::back_inserter(recvBufs),
                   [](const at::Tensor& t) { return t.data_ptr(); });
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::allgatherv(data.data_ptr(),
                             (size_t)data.numel(),
                             recvBufs,
                             recvCounts,
                             get_ccl_datatype(data.scalar_type()),
                             _get_comm_from_group(group),
                             ds_stream), ds_stream.get_native());
}

void barrier(py::object group, bool async_op)
{
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::barrier(_get_comm_from_group(group), ds_stream), ds_stream.get_native());
}

struct timeval start_t, end_t;

void reduce(torch::Tensor& data, int dst, py::object op, py::object group, bool async_op)
{
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::reduce(data.data_ptr(),
                         data.data_ptr(),
                         data.numel(),
                         get_ccl_datatype(data.scalar_type()),
                         get_ccl_reduce_op(op, data),
                         dst,
                         _get_comm_from_group(group),
                         ds_stream), ds_stream.get_native());
}

void reduce_scatter(torch::Tensor& data, std::vector<torch::Tensor>& vec_data_in, py::object op, py::object group, bool async_op)
{
    torch::Tensor input = vec_data_in[0];
    for(int i=1;i<vec_data_in.size();++i)
    {
        input = torch::cat({input, vec_data_in[0]}, 0);
    }
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::reduce_scatter(input.data_ptr(),
                                 data.data_ptr(),
                                 data.numel(),
                                 get_ccl_datatype(data.scalar_type()),
                                 get_ccl_reduce_op(op, data),
                                 _get_comm_from_group(group),
                                 ds_stream), ds_stream.get_native());
}

// TODO: implement torch's async_op behavior, document it.
void all_reduce(torch::Tensor& data, py::object op, py::object group, bool async_op)
{
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::allreduce(data.data_ptr(),
                            data.data_ptr(),
                            data.numel(),
                            get_ccl_datatype(data.scalar_type()),
                            get_ccl_reduce_op(op, data),
                            _get_comm_from_group(group),
                            ds_stream), ds_stream.get_native());
}

void send(torch::Tensor& data, int dst, py::object group, bool async_op)
{
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::send(data.data_ptr(),
                       data.numel(),
                       get_ccl_datatype(data.scalar_type()),
                       dst,
                       _get_comm_from_group(group),
                       ds_stream), ds_stream.get_native());
}

void recv(torch::Tensor& data, int src, py::object group, bool async_op)
{
    ccl::event ret_evt;
    CCL_KERNEL_SUBMIT(ret_evt = ccl::recv(data.data_ptr(),
                       data.numel(),
                       get_ccl_datatype(data.scalar_type()),
                       src,
                       _get_comm_from_group(group),
                       ds_stream), ds_stream.get_native());
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_kvs_addr", &get_kvs_addr, "create and get main kvs addr");
    m.def("initialize", &initialize, "ccl initialize");
    m.def("get_rank", &get_rank, "get rank");
    m.def("get_world_size", &get_world_size, "get world size");
    m.def("broadcast", &broadcast, "ccl broadcast");
    m.def("all_gather", &all_gather, "ccl all_gather");
    m.def("barrier", &barrier, "barrier");
    m.def("reduce", &reduce, "ccl reduce");
    m.def("reduce_scatter", &reduce_scatter, "ccl reduce_scatter");
    m.def("all_reduce", &all_reduce, "ccl all_reduce");
    m.def("send", &send, "ccl send");
    m.def("recv", &recv, "ccl recv");
}