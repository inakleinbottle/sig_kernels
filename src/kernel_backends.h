//
// Created by sam on 03/11/2021.
//

#ifndef SIG_KERNELS_KERNEL_BACKENDS_H
#define SIG_KERNELS_KERNEL_BACKENDS_H

#include "implementation_types.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstddef>

using std::size_t;

namespace sig_kernels {

namespace py = pybind11;

using np_darray_t = py::array_t<scalar_t>;

np_darray_t sig_kernel_batch_varpar_naive(const np_darray_t&);
np_darray_t sig_kernel_batch_varpar(const np_darray_t&);
std::vector<scalar_t> sig_kernel_batch_varpar(const scalar_t* ips, size_t length_x, size_t length_y);

np_darray_t sig_kernel_batch_varpar_cuda(const py::array_t<scalar_t, py::array::c_style>&);

std::vector<std::vector<scalar_t>> compute_antidiagonals(const scalar_t* inner_prods, size_t length_x, size_t length_y);



}



#endif //SIG_KERNELS_KERNEL_BACKENDS_H
