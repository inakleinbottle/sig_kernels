//
// Created by sam on 17/11/2021.
//

#ifndef SIG_KERNELS_KERNEL_COMPUTE_CUDA_H
#define SIG_KERNELS_KERNEL_COMPUTE_CUDA_H

#include "implementation_types.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/zip_function.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_ptr.h>


#include "antidiagonal_matrix.h"

namespace sig_kernels {


template <typename T>
struct cuda_dot_product
{
    size_t dimension;

    __host__ __device__ explicit cuda_dot_product(size_t d) : dimension(d) {}

    __host__ __device__
    T operator()(thrust::device_ptr<const T> x, thrust::device_ptr<const T> y) const noexcept
    {
        T result(0);
        for (size_t i=0; i < dimension; ++i) {
            result += x[i]*y[i];
        }
        return result;
    }

};

template <typename Kernel>
struct numerical_method_advance
{
    Kernel m_kernel;
    thrust::device_ptr<const scalar_t> x_ptr, y_ptr;
    size_t  dimension;

    __host__ __device__
    numerical_method_advance(Kernel ker, size_t ix, size_t iy, size_t dim,
                             const thrust::device_ptr<const scalar_t>& x,
                             const thrust::device_ptr<const scalar_t>& y)
            : m_kernel(ker), x_ptr(x + static_cast<std::ptrdiff_t>(ix*dim)),
          y_ptr(y + static_cast<std::ptrdiff_t>(iy*dim)), dimension(dim)
    {
    }

    __host__ __device__
    double operator()(size_t r, double k1r1, double k1r, double k2r1) const
    {
        auto dkr = m_kernel(x_ptr + r*dimension, y_ptr - (r+1)*dimension);
        auto dkr2 = dkr * dkr;
        return (k1r1 + k1r)*(1.0 + 0.5*dkr + (1.0/12.0)*dkr2) - k2r1*(1.0 - (1.0/12.0)*dkr2);
    }

};





template<typename Kernel>
__host__ __device__
antidiagonal_matrix<scalar_t, thrust::device_vector>
compute_antidiagonals_cuda(
        const thrust::device_vector<scalar_t>& X,
        const thrust::device_vector<scalar_t>& Y,
        size_t length_x, size_t length_y, size_t dimension,
        Kernel kernel)
{
    antidiagonal_matrix<scalar_t, thrust::device_vector> result(length_x, length_y);
    result[0].push_back(1.0);
    result[1].push_back(1.0);
    result[1].push_back(1.0);
    result.resize(result.capacity());

    /*
     * A length_x x length_y matrix has length_x + length_y - 1 antidiagonals.
     * The size of the kth antidiagonal is min(k, num_antidiagonals-k, length_x, length_y)
     */
    auto min_xy = thrust::min(length_x, length_y);
    auto max_xy = thrust::max(length_x, length_y);
    auto num_antidiagonals = length_x + length_y - 1;

    for (size_t k = 2; k < min_xy; ++k) {

        result[k][0] = 1.0;
        result[k][k] = 1.0;

        auto zipped_iter_begin = thrust::make_zip_iterator(
                        thrust::counting_iterator<size_t>(0),
                        result[k-1].begin(),
                        ++(result[k-1].begin()),
                        result[k-2].begin()
                        );
        auto zipped_iter_end = thrust::make_zip_iterator(
                        thrust::counting_iterator<size_t>(k-1),
                        --(result[k-1].end()),
                        result[k-1].end(),
                        result[k-2].end()
                );
        auto zip_advance = thrust::make_zip_function(numerical_method_advance<Kernel>(kernel, 0, k-1, dimension, X.data(), Y.data()));
        thrust::transform(
                    zipped_iter_begin, zipped_iter_end, ++result[k].begin(), zip_advance
                );

    }

    if (length_x > length_y) {

    } else if (length_y > length_x) {

    } else {

        auto zipped_iter_begin = thrust::make_zip_iterator(
                        thrust::counting_iterator<size_t>(0),
                        result[max_xy-1].begin(),
                        ++(result[max_xy-1].begin()),
                        result[max_xy-2].begin()
                        );
        auto zipped_iter_end = thrust::make_zip_iterator(
                        thrust::counting_iterator<size_t>(num_antidiagonals-max_xy),
                        --(result[max_xy-1].end()),
                        result[max_xy-1].end(),
                        result[max_xy-2].end()
                );
        auto zip_advance = thrust::make_zip_function(numerical_method_advance<Kernel>(
                kernel, 0, max_xy-1, dimension, X.data(), Y.data()));

        thrust::transform(
                zipped_iter_begin, zipped_iter_end, result[max_xy].begin(), zip_advance);
    }

    for (size_t k = max_xy + 1; k < num_antidiagonals; ++k) {
        auto zipped_iter_begin = thrust::make_zip_iterator(
                        thrust::counting_iterator<size_t>(0),
                        result[k-1].begin(),
                        ++(result[k-1].begin()),
                        ++(result[k-2].begin())
                        );
        auto zipped_iter_end = thrust::make_zip_iterator(
                        thrust::counting_iterator<size_t>(num_antidiagonals-k),
                        result[k-1].end(),
                        result[k-1].end(),
                        result[k-2].end()
                );
        auto zip_advance = thrust::make_zip_function(numerical_method_advance<Kernel>(
                kernel, k-length_y, length_y-1, dimension, X.data(), Y.data()));
        thrust::transform(
                zipped_iter_begin, zipped_iter_end, result[k].begin(), zip_advance);
    }


    return result;
}


extern template __host__ __device__
antidiagonal_matrix<scalar_t, thrust::device_vector>
compute_antidiagonals_cuda<cuda_dot_product<scalar_t> >(
        const thrust::device_vector<scalar_t>& X,
        const thrust::device_vector<scalar_t>& Y,
        size_t length_x, size_t length_y, size_t dimension,
        cuda_dot_product<scalar_t> kernel);



__host__ void compute_cuda_result(const std::vector<scalar_t>& X, const std::vector<scalar_t>& Y,
                                  size_t lengthx, size_t lengthy, size_t dimension);


__host__ antidiagonal_matrix<scalar_t, thrust::device_vector> compute_cuda_result_in(
        const std::vector<scalar_t>& X, const std::vector<scalar_t>& Y,
        size_t lengthx, size_t lengthy, size_t dimension
        );

}






#endif//SIG_KERNELS_KERNEL_COMPUTE_CUDA_H
