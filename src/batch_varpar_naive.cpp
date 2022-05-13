//
// Created by sam on 03/11/2021.
//

#include "implementation_types.h"
#include "kernel_backends.h"

namespace py = pybind11;


/*
 * This is a direct translation of Cris Salvi's cython implementation of the sig_kernel_batch_varpar
 * function, with a few minor tweaks, with the _naive_solver flag true. This is going to be our reference
 * implementation in the end.
 * See https://github.com/crispitagorico/sigkernel/blob/master/sigkernel/cython_backend.pyx
 */

sig_kernels::np_darray_t sig_kernels::sig_kernel_batch_varpar_naive(const np_darray_t& data)
{
    assert(data.ndim()==3);
    auto A = data.shape(0);
    auto M = data.shape(1);
    auto N = data.shape(2);

    py::array_t<scalar_t, py::array::c_style> result({A, M+1, N+1});

    py::gil_scoped_release release_the_gil;

    for (auto l = 0; l<A; ++l) {

        /*
         * This loop is fast, because we're looping over a contiguous block of memory.
         * Let's just grab the pointer to the beginning of this range and rattle through
         * using pointer arithmetic.
         */
        {
            auto* ptr = result.mutable_data(l, 0, 0);
            for (auto j = 0; j<=N; ++j) {
                *(ptr++) = 1.0;
            }
        }


        /*
         * This loop is slower than the above because the result[l, i, 0] are not
         * contiguous in memory. Let's just use the mutable_at accessor.
         *
         * Note we can skip index 0, because it's already done by the above loop.
         */
        for (auto i = 1; i<=M; ++i) {
            result.mutable_at(l, i, 0) = 1.0;
        }

        /*
         * Now for the fun stuff.
         */
        for (auto i = 0; i<M; ++i) {
            for (auto j = 0; j<N; ++j) {
                // This is just to simplify the formula here
                auto& datalij = data.at(l, i, j);
                /*
                 * Define C_{i,j} = <x_{s_{i+1}} - x_{s_i}, y_{t_{j+1}} - y_{t_j}}>
                 *
                 * So in the paper arXiv::2006.1479v9, the explicit finite difference scheme for finding the
                 * signature kernel is given by
                 *  k(s_{i+1}, t_{j+1}) = k(s_{i+1}, t_j) + k(s_i, t_{j+1}) - k(s_i, t_j)
                 *      + 0.5C_{i,j}(k(s_{i+1}, t_j) + k(s_i, t_{j+1}))
                 *                      = (k(s_{i+1}, t_j) + k(s_i, t_{j+1}))(1 + 0.5C_{i,j}) - k(s_i, t_j)
                 *
                 */
                result.mutable_at(l, i+1, j+1) =
                        result.at(l, i+1, j) + result.at(l, i, j+1) + result.at(l, i, j)*(datalij - 1.0);
            }
        }
    }

    return result;

}