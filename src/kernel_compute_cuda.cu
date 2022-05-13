
#include "kernel_compute_cuda.h"
#include <thrust/host_vector.h>
#include <iostream>
#include <iomanip>
#include <chrono>



template __host__ __device__ sig_kernels::antidiagonal_matrix<scalar_t, thrust::device_vector>
sig_kernels::compute_antidiagonals_cuda<sig_kernels::cuda_dot_product<scalar_t>>(
        const thrust::device_vector<scalar_t>& X,
        const thrust::device_vector<scalar_t>& Y,
        size_t length_x, size_t length_y, size_t dimension,
        sig_kernels::cuda_dot_product<scalar_t> kernel);


namespace {

struct timer
{
    using clock = std::chrono::high_resolution_clock;
    typename clock::time_point start;

    timer() : start(clock::now()) {}

    ~timer()
    {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start).count() << '\n';
    }

};


}


__host__ void sig_kernels::compute_cuda_result(const std::vector<scalar_t>& X, const std::vector<scalar_t>& Y, size_t lengthx, size_t lengthy, size_t dimension)
{

    sig_kernels::cuda_dot_product<scalar_t> cdp(dimension);
    thrust::device_vector<scalar_t> CX(X.begin(), X.end());
    thrust::device_vector<scalar_t> CY(Y.begin(), Y.end());

    sig_kernels::antidiagonal_matrix<scalar_t, thrust::device_vector> cuda_result;

    {
        timer t;
        cuda_result = sig_kernels::compute_antidiagonals_cuda(CX, CY, lengthx, lengthy, dimension, cdp);
    }


/*
    {
        auto diags_to_val = [&](int r, int c) {
            return cuda_result[r + c][(r + c >= lengthy) ? lengthy - 1 - c : r];
        };

        for (int i = 0; i < lengthx; ++i) {
            for (int j = 0; j < lengthy; ++j) {
                std::cout << std::setw(15) << diags_to_val(i, j) << ' ';
            }
            std::cout << '\n';
        }
    }
*/
    std::cout << cuda_result.back() << '\n';

}

sig_kernels::antidiagonal_matrix<scalar_t, thrust::device_vector> sig_kernels::compute_cuda_result_in(const std::vector<scalar_t>& X, const std::vector<scalar_t>& Y, size_t lengthx, size_t lengthy, size_t dimension)
{
    sig_kernels::cuda_dot_product<scalar_t> cdp(dimension);
    thrust::device_vector<scalar_t> CX(X.begin(), X.end());
    thrust::device_vector<scalar_t> CY(Y.begin(), Y.end());

    sig_kernels::antidiagonal_matrix<scalar_t, thrust::device_vector> cuda_result;

    cuda_result = sig_kernels::compute_antidiagonals_cuda(CX, CY, lengthx, lengthy, dimension, cdp);
    return cuda_result;
}
