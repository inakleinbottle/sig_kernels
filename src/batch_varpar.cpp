//
// Created by sam on 03/11/2021.
//
#include "implementation_types.h"
#include "kernel_backends.h"
#include <iostream>

namespace py = pybind11;

/*
 * This is a direct translation of Cris Salvi's cython implementation of the sig_kernel_batch_varpar
 * function, with a few minor tweaks. This is going to be our reference implementation in the end.
 * See https://github.com/crispitagorico/sigkernel/blob/master/sigkernel/cython_backend.pyx
 */




std::vector<std::vector<scalar_t> >
sig_kernels::compute_antidiagonals(const scalar_t* inner_prods, size_t length_x, size_t length_y)
{
    std::vector<std::vector<scalar_t> > result {{1}, {1, 1}};
    /*
     * A length_x x length_y matrix has length_x + length_y - 1 antidiagonals.
     * The size of the kth antidiagonal is min(k, num_antidiagonals-k, length_x, length_y)
     */
    auto min_xy = std::min(length_x, length_y);
    auto max_xy = std::max(length_x, length_y);
    auto num_antidiagonals = length_x + length_y - 1;
    result.resize(num_antidiagonals);

    /*
     * The first min_xy antidiagonals span boundary to boundary, so the first and last entry is 1.0.
     * In this case, the row and column are given by the following formulae:
     *   row = ad_pos
     *   col = anti_diagonal - ad_pos
     *
     */
    //            result.mutable_at(l, i+1, j+1) =
    //                    (result.at(l, i+1, j) + result.at(l, i, j+1))*(1.0 + 0.5*datalij + (1.0/12.0)*datalij2)
    //                            - datalij*(1.0 - (1.0/12.0)*datalij2);
    {
        auto get_ip = [&](size_t ad, size_t pos) {
          return inner_prods[pos*length_y + (ad - pos)];
        };

        // The first two diagonals are trivial
        for (size_t k = 2; k<min_xy; ++k) {
            auto ad_length = std::min(k, min_xy);
            result[k].reserve(ad_length);

            // The first entry is 1, from the boundary conditions.
            result[k].emplace_back(1.0);

            for (size_t r = 1; r<ad_length; ++r) {
                auto dkr = get_ip(k-2, r-1);
                auto dkr2 = dkr*dkr;
                result[k].emplace_back((result[k-1][r-1]+result[k-1][r])*(1.0+0.5*dkr + (1.0/ 12.0)*dkr2)
                                - result[k-2][r-1]*(1.0 - (1.0/12.0)*dkr2));
            }
            result[k].emplace_back(1.0);
        }

    }

    /*
     * The middle cases are tricky because they depend on whether the rows or columns are the dominant
     * dimension. If both are equal, so the matrix is square, then middle steps are unnecessary. There
     * are exactly max_xy - min_xy cases to treat here. In all cases, these antidiagonals have length
     * min_xy
     */

    if (length_x > length_y) {

        /*
         * When length_y == min_xy the antidiagonals start on the top row of the matrix, so in general
         * the row is pos. The column starts at the antidiagonal number, and decrease.
         */

        auto get_ip = [&](size_t ad, size_t pos) {
              size_t row, col;
              if (ad>=length_y) {
                  col = length_y-1-pos;
                  row = ad-col;
              }
              else {
                  row = pos;
                  col = ad-row;
              }
              auto r = inner_prods[row*length_y+col];

              return r;
            };

        for (size_t k=length_y+1; k<length_x; ++k) {
            result[k].reserve(min_xy);
            // When length_x is dominant, result[k][0] = 1.0
            result[k].emplace_back(1.0);


            for (size_t r=1; r<min_xy; ++r) {
                auto dkr = get_ip(k-2, r-1);
                auto dkr2 = dkr*dkr;
                result[k].emplace_back((result[k-1][r-1]+result[k-1][r-1])*(1.0+0.5*dkr + (1.0/ 12.0)*dkr2)
                                - result[k-2][r-1]*(1.0 - (1.0/12.0)*dkr2));
            }
        }

    } else if (length_y > length_x) {

        /*
         * When length_x == min_xy the antidiagonals start in the final column of the matrix on
         * and decrease until they reach the left-hand side. Hence row is ad - length_x - pos, and
         * column is given by length_x - pos.
         */

        auto get_ip = [&](size_t ad, size_t pos) {
            return inner_prods[pos * length_y + (ad - pos)];
        };

        for (size_t k = length_x+1; k<length_y; ++k) {
            result[k].reserve(min_xy);
            for (size_t r=0; r<min_xy-1; ++r) {
                auto dkr = get_ip(k-2, r-1);
                auto dkr2 = dkr*dkr;
                result[k].emplace_back((result[k-1][r-1]+result[k-1][r])*(1.0+0.5*dkr + (1.0/ 12.0)*dkr2)
                                - result[k-2][r-1]*(1.0 - (1.0/12.0)*dkr2));
            }
            // When length_y is dominant, result[k][length_y-1] = 1.0
            result[k].emplace_back(1.0);
        }

    } else {

        auto get_ip = [&](size_t ad, size_t pos) {
            return inner_prods[pos * length_y + (ad - pos)];
        };

        /*
         * The max_xy diagonal in a square matrix is special, the (i, j) on antidiagonal
         * max_xy and (i-1,j-1) on antidiagonal max_xy - 2 share the same offset
         */
        result[max_xy].reserve(num_antidiagonals-max_xy);
        for (size_t r=0; r < num_antidiagonals-max_xy; ++r) {
            auto dkr = get_ip(max_xy-2, r);
            auto dkr2 = dkr*dkr;
            result[max_xy].emplace_back((result[max_xy-1][r]+result[max_xy-1][r+1])*(1.0+0.5*dkr + (1.0/ 12.0)*dkr2)
                            - result[max_xy-2][r]*(1.0 - (1.0/12.0)*dkr2));
        }

    }


    /*
     * For the last min_xy - 1 anti-diagonals we're in the bottom right half of the matrix.
     * In this case, the row and column are given by the following formulae:
     *   row = length_x - anti_diagonal - max_xy  + r
     *   col = length_y - antidiagonal - max_xy - r
     */
    {

        auto get_ip = [&](size_t ad, size_t pos) {
            size_t col = length_y - 1 - pos;
            return inner_prods[(ad - col) * length_y + col];
        };

        for (size_t k=max_xy+1; k<num_antidiagonals; ++k) {
            result[k].reserve(num_antidiagonals-k);

            for (size_t r = 0; r < num_antidiagonals - k; ++r) {
                auto dkr = get_ip(k-2, r + 1);
                auto dkr2 = dkr*dkr;
                result[k].emplace_back((result[k-1][r]+result[k-1][r+1])*(1.0+0.5*dkr+(1.0/12.0)*dkr2)
                        -result[k-2][r+1]*(1.0-(1.0/12.0)*dkr2));
            }
        }

    }

    return result;
}

std::vector<scalar_t> sig_kernels::sig_kernel_batch_varpar(const scalar_t* ips, size_t length_x, size_t length_y)
{

    auto get = [&](size_t i, size_t j) -> const scalar_t& {
        return ips[i * length_y + j];
    };

    std::vector<scalar_t> result((length_x + 1) * (length_y + 1));

    auto getr = [&](size_t i, size_t j) -> scalar_t& {
        return result[i * length_y + j];
    };

    for (size_t j = 0; j <= length_x; ++j) {
        getr(0, j) = 1.0;
    }

    for (size_t i = 1; i <= length_y; ++i) {
        getr(i, 0) = 1.0;
    }

    for (auto i = 0; i < length_x; ++i) {
        for (auto j = 0; j < length_y; ++j) {
            // This is just to simplify the formula here
            auto& datalij = get(i, j);
            auto datalij2 = datalij * datalij;// Square of data[l, i, j]

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
            getr(i + 1, j + 1) =
                    (getr(i + 1, j) + getr(i, j + 1)) * (1.0 + 0.5 * datalij + (1.0 / 12.0) * datalij2)
                    - getr(i, j) * (1.0 - (1.0 / 12.0) * datalij2);
        }
    }
    return result;
}

/**
 * @brief Compute the kernel at pairs (s_i, t_j) over the domain.
 *
 *
 * @param data Set of static kernel applied to increments k(X_{s_i,s_{i+1}}, Y_{s_j, s_{j+1}})
 * for each path in the batch.
 * @return
 */
sig_kernels::np_darray_t sig_kernels::sig_kernel_batch_varpar(const np_darray_t& data)
{
    assert(data.ndim() == 3);
    auto A = data.shape(0);
    auto M = data.shape(1);
    auto N = data.shape(2);

    py::array_t<scalar_t, py::array::c_style> result({A, M+1, N+1});

    py::gil_scoped_release release_the_gil;

    auto rdata = data.unchecked();


    auto mresult = result.mutable_unchecked();
    for (auto l=0; l < A; ++l) {

        /*
         * This loop is fast, because we're looping over a contiguous block of memory.
         * Let's just grab the pointer to the beginning of this range and rattle through
         * using pointer arithmetic.
         */
        {
            auto* ptr = mresult.mutable_data(l, 0, 0);
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
        {
            for (auto i = 1; i <= M; ++i) {
                mresult(l, i, 0) = 1.0;
            }
        }

        /*
         * Now for the fun stuff.
         */
        for (auto i=0; i<M; ++i) {
            for (auto j=0; j<N; ++j) {
                // This is just to simplify the formula here
                auto& datalij = rdata(l, i, j);
                auto datalij2 = datalij*datalij; // Square of data[l, i, j]

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
                mresult(l, i+1, j+1) =
                        (mresult(l, i+1, j) + mresult(l, i, j+1))*(1.0 + 0.5*datalij + (1.0/12.0)*datalij2)
                                - mresult(l, i, j)*(1.0 - (1.0/12.0)*datalij2);
            }
        }
    }


    return result;

}