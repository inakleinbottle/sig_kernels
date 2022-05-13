//
// Created by sam on 03/11/2021.
//

#include "kernel_backends.h"
#include <cassert>


namespace py = pybind11;
using namespace pybind11::literals;


py::array_t<double, py::array::c_style> pyarray_diff(const py::array_t<double>& arg)
{
    std::vector<ssize_t> shape(arg.shape(), arg.shape() + arg.ndim());
    assert(shape[1] > 1);
    shape[1] -= 1;

    py::array_t<double, py::array::c_style> incr(shape);

    for (auto batch=0; batch<shape[0]; ++batch) {
        for (auto i=0; i<shape[1]; ++i) {
            auto* out = incr.mutable_data(batch, i, 0);
            const auto* p1 = arg.data(batch, i, 0);
            const auto* p2 = arg.data(batch, i+1, 0);
            for (auto j=0; j<shape[2]; ++j) {
                out[j] = p2[j] - p1[j];
            }
        }
    }

    return incr;
}

struct linear_kernel
{
    double operator()(const double* x, const double* y, ssize_t d) const
    {
        double result = 0.0;
        for (ssize_t i = 0; i<d; ++i) {
            result += x[i]*y[i];
        }
        return result;
    }

};


py::array_t<double> sigkernel(const py::array_t<double>& X,
        const py::array_t<double>& Y)
{
    // X and Y are batches of value paths of equal size.
    // In esig_paths, these will be iterators of increments.
    auto X_incr = pyarray_diff(X);
    auto Y_incr = pyarray_diff(Y);
    linear_kernel ker;

    py::array_t<double, py::array::c_style> static_kernel({X_incr.shape(0), X_incr.shape(1), Y_incr.shape(1)});

    // For now, do the kernel calculation here
    // Since we will eventually want to take in a vector of increments, this is probably best done
    // in the computation itself, since we will be iterating through there anyway.
    for (auto i=0; i<X_incr.shape(0); ++i) {
        for (auto j=0; j<X_incr.shape(1); ++j) {
            for (auto k=0; k<Y_incr.shape(1); ++k) {
                static_kernel.mutable_at(i,j,k) = ker(X_incr.data(i,j,0), Y_incr.data(i,k,0), X_incr.shape(2));
            }
        }
    }



    auto tmp = sig_kernels::compute_antidiagonals(static_kernel.data(), X_incr.shape(1), Y_incr.shape(1));

    // Unpack into a numpy array
    py::array_t<double> result({X_incr.shape(1), Y_incr.shape(1)});

    for (size_t row = 0; row < X_incr.shape(1); ++row) {
        for (size_t col = 0; col <Y_incr.shape(1); ++col) {
            auto k = row + col;
            // The antidiagonal length is the minimum of min{x_length, y_length}, k, and num_antidiagonals - k
            auto r = (k > Y_incr.shape(1)) ? Y_incr.shape(1) - 1 - col : row;
            result.mutable_at(row, col) = tmp[k][r];
        }
    }
    return result;
}





PYBIND11_MODULE(sigker_backends, m) {

    m.def(
            "sig_kernel_batch_varpar",
            sig_kernels::sig_kernel_batch_varpar,
            "paths_data"_a
            );
    m.def(
            "sig_kernel_batch_varpar_naive",
            sig_kernels::sig_kernel_batch_varpar_naive,
            "paths_data"_a
            );

    m.def(
            "sig_kernel",
            sigkernel,
            "X"_a, "Y"_a
            );




}