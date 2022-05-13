//
// Created by sam on 15/11/2021.
//

#ifndef SIG_KERNELS_DOT_PRODUCT_H
#define SIG_KERNELS_DOT_PRODUCT_H

#include <cstddef>

template <typename T>
struct dot_product {
    size_t dimension;

    explicit dot_product(size_t d) : dimension(d)
    {}

    T operator()(const T* x, const T* y) const noexcept
    {
        auto result = T(0);
        for (size_t i = 0; i < dimension; ++i) {
            result += x[i] * y[i];
        }
        return result;
    }

};



#endif//SIG_KERNELS_DOT_PRODUCT_H
