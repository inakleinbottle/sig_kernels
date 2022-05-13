//
// Created by sam on 13/11/2021.
//

#ifndef SIG_KERNELS_KERNEL_COMPUTE_H
#define SIG_KERNELS_KERNEL_COMPUTE_H

#include "implementation_types.h"

#include <iostream>
#include <vector>

#include "antidiagonal_matrix.h"


namespace sig_kernels {

template <typename Kernel>
std::vector<scalar_t> compute_antidiagonals(const scalar_t* X, const scalar_t* Y, size_t length_x, size_t length_y, size_t dimension, Kernel kernel)
{

    constexpr scalar_t one(1), half(0.5), twelth(scalar_t(1)/scalar_t(12));


    antidiagonal_matrix<scalar_t> result(length_x, length_y);
    result[0].emplace_back(one);
    result[1].emplace_back(one);
    result[1].emplace_back(one);

    /*
     * A length_x x length_y matrix has length_x + length_y - 1 antidiagonals.
     * The size of the kth antidiagonal is min(k, num_antidiagonals-k, length_x, length_y)
     */
    auto min_xy = std::min(length_x, length_y);
    auto max_xy = std::max(length_x, length_y);
    auto num_antidiagonals = length_x + length_y - 1;

    auto compute = [](auto rk11, auto rk10, auto rk21, auto dkr, auto dkr2) {
        return (rk11 + rk10)*(one+half*dkr+twelth*dkr2) - rk21*(one-twelth*dkr2);
    };

    auto get_x = [&](size_t idx) {
        return X + idx*dimension;
    };

    auto get_y = [&](size_t idx) {
        return Y + idx*dimension;
    };

    /*
     * The first min_xy antidiagonals span boundary to boundary, so the first and last entry is 1.0.
     * In this case, the row and column are given by the following formulae:
     *   row = ad_pos
     *   col = anti_diagonal - ad_pos
     *
     */

    {
        auto get_ip = [&](size_t ad, size_t pos) {
            return kernel(get_x(pos), get_y(ad-pos));
        };

        // The first two diagonals are trivial
        for (size_t k = 2; k < min_xy; ++k) {


            const auto* x = get_x(0);
            const auto* y = get_y(k-1);

            // The first entry is 1, from the boundary conditions.
            result[k].emplace_back(one);

            const auto* p_rk1 = result[k-1].data();
            const auto* p_rk2 = result[k-2].data();

            for (size_t r = 1; r < k; ++r) {
                auto dkr = kernel(x, y -= dimension);
                x += dimension;
                //auto dkr = get_ip(k-2, r-1);
                auto dkr2 = dkr * dkr;
                result[k].emplace_back(compute(p_rk1[r-1], p_rk1[r], p_rk2[r-1], dkr, dkr2));
            }
            result[k].emplace_back(one);
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
            if (ad >= length_y) {
                col = length_y - 1 - pos;
                row = ad - col;
            }
            else {
                row = pos;
                col = ad - row;
            }
            return kernel(get_x(row), get_y(col));
        };

        for (size_t k = length_y + 1; k < length_x; ++k) {
            //result[k].reserve(min_xy+1);
            // When length_x is dominant, result[k][0] = 1.0
            result[k].emplace_back(one);

            for (size_t r = 1; r < min_xy; ++r) {
                auto dkr = get_ip(k - 2, r - 1);
                auto dkr2 = dkr * dkr;
                result[k].emplace_back(compute(result[k-1][r-1], result[k-1][r], result[k-2][r-1], dkr, dkr2));
            }
        }
    }
    else if (length_y > length_x) {

        /*
         * When length_x == min_xy the antidiagonals start in the final column of the matrix on
         * and decrease until they reach the left-hand side. Hence row is ad - length_x - pos, and
         * column is given by length_x - pos.
         */

        auto get_ip = [&](size_t ad, size_t pos) {
            return kernel(get_x(pos), get_y(ad-pos));
        };

        for (size_t k = length_x + 1; k < length_y; ++k) {
           // result[k].reserve(min_xy+1);
            for (size_t r = 0; r < min_xy - 1; ++r) {
                auto dkr = get_ip(k - 2, r - 1);
                auto dkr2 = dkr * dkr;
                result[k].emplace_back(compute(result[k-1][r-1], result[k-1][r], result[k-2][r-1], dkr, dkr2));
            }
            // When length_y is dominant, result[k][length_y-1] = 1.0
            result[k].emplace_back(one);
        }
    }
    else {

        auto get_ip = [&](size_t ad, size_t pos) {
            return kernel(get_x(pos), get_y(ad-pos));
        };

        /*
         * The max_xy diagonal in a square matrix is special, the (i, j) on antidiagonal
         * max_xy and (i-1,j-1) on antidiagonal max_xy - 2 share the same offset
         */
        const auto* x = get_x(max_xy-2);
        //result[max_xy].reserve(num_antidiagonals - max_xy+1);
        for (size_t r = 0; r < num_antidiagonals - max_xy; ++r) {
            auto dkr = get_ip(max_xy - 2, r);
            auto dkr2 = dkr * dkr;
            result[max_xy].emplace_back(compute(result[max_xy-1][r], result[max_xy-1][r+1], result[max_xy-2][r], dkr, dkr2));
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
            return kernel(get_x(ad - col), get_y(col));
        };

        for (size_t k = max_xy + 1; k < num_antidiagonals; ++k) {
            //result[k].reserve(num_antidiagonals - k + 1);

            for (size_t r = 0; r < num_antidiagonals - k; ++r) {
                auto dkr = get_ip(k - 2, r + 1);
                auto dkr2 = dkr * dkr;
                result[k].emplace_back(compute(result[k-1][r], result[k-1][r], result[k-2][r+1], dkr, dkr2));
            }
        }
    }

    return static_cast<std::vector<scalar_t>>(result);
}

} // sig_kernels

#endif//SIG_KERNELS_KERNEL_COMPUTE_H
