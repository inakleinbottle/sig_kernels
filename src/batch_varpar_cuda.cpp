//
// Created by sam on 03/11/2021.
//
#include "implementation_types.h"
#include "kernel_backends.h"
#include <thrust/device_vector.h>
#include <pybind11/pybind11.h>
#include <cassert>


namespace py = pybind11;






sig_kernels::np_darray_t sig_kernels::sig_kernel_batch_varpar_cuda(const py::array_t<scalar_t, py::array::c_style>& arg)
{
    auto ndim = arg.ndim();
    assert(ndim == 3);
    auto strides = arg.strides();
    thrust::device_vector<scalar_t> device_data(arg.data(), arg.data()+arg.size());
    thrust::device_vector<scalar_t> result;

    auto get_element = [&](size_t block, size_t row, size_t col) {
        return device_data[block*strides[0] + row*strides[1] + col];
    };

    for (size_t block=0; block<arg.shape(0); ++block) {

    }


}