//
// Created by sam on 12/11/2021.
//
#include "implementation_types.h"
#include "kernel_backends.h"

#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "kernel_compute.h"
#include "kernel_compute_cuda.h"
#include <thrust/host_vector.h>

#include <pybind11/embed.h>


namespace py = pybind11;


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

struct dot_product
{
    size_t dimension;

    constexpr dot_product(size_t d) : dimension(d)
    {}

    double operator()(const scalar_t* x, const scalar_t* y) const noexcept
    {
        double result = 0.0;
        for (size_t i=0; i<dimension; ++i) {
            result += x[i] * y[i];
        }
        return result;
    }

};


int main()
{
    size_t lengthx = 20000;
    size_t lengthy = 20000;
    size_t dimension = 5;

    std::mt19937 rng;
    std::normal_distribution<scalar_t> dist(0.0, 0.01);


    std::vector<scalar_t> X, Y;
    X.reserve(lengthx*dimension);
    Y.reserve(lengthy*dimension);
    for (size_t i=0; i<lengthx; ++i) {
        for (size_t j=0; j<dimension; ++j) {
            X.emplace_back(dist(rng));
        }
    }
    for (size_t i=0; i<lengthy; ++i) {
        for (size_t j=0; j<dimension; ++j) {
            Y.emplace_back(dist(rng));
        }
    }

    std::vector<scalar_t> ips;
    ips.reserve(lengthx*lengthy);
    for (size_t i=0; i<lengthx; ++i) {
        for (size_t j=0; j<lengthy; ++j) {
            scalar_t v = 0.0;
            for (size_t k=0; k<dimension; ++k) {
                v += X[i*dimension + k]*Y[j*dimension + k];
            }
            ips.push_back(v);
        }
    }

    dot_product dp(dimension);
/*
    for (int i=0; i<lengthx; ++i) {
        for (int j=0; j<lengthy; ++j) {
            std::cout << std::setw(15) << dp(X.data() + i*dimension, Y.data() + j*dimension) << ' ';
        }
        std::cout << '\n';
    }
*/
    py::scoped_interpreter interpreter {};

    std::vector<scalar_t> r1;
    {
        timer t;
        dot_product kernel(dimension);
        r1 = sig_kernels::compute_antidiagonals(X.data(), Y.data(), lengthx, lengthy, dimension, kernel);
    }
    std::cout << r1.back() << '\n';
/*
    {
        sig_kernels::antidiagonal_matrix<double> adr(lengthx, lengthy);
        adr.insert(r1.begin(), r1.end());

        auto diags_to_val = [&](int r, int c) {
            return adr[r+c][(r+c >= lengthy) ? lengthy-1-c : r];
        };

        for (int i = 0; i < lengthx; ++i) {
            for (int j = 0; j < lengthy; ++j) {
                std::cout << std::setw(15) << diags_to_val(i, j) << ' ';
            }
            std::cout << '\n';
        }
    }
*/
    std::vector<std::vector<scalar_t> > result;
    {
        timer t;
        result = sig_kernels::compute_antidiagonals(ips.data(), lengthx, lengthy);
    }
    std::cout << result[lengthx+lengthy-2][0] << '\n';
/*
    {
        auto diags_to_val = [&](int r, int c) {
            return result[r + c][(r + c >= lengthy) ? lengthy - 1 - c : r];
        };

        for (int i = 0; i < lengthx; ++i) {
            for (int j = 0; j < lengthy; ++j) {
                std::cout << std::setw(15) << diags_to_val(i, j) << ' ';
            }
            std::cout << '\n';
        }
    }
*/

    py::array_t<scalar_t> tmp({size_t(1), lengthx , lengthy}, ips.data());

    py::array_t<scalar_t> r;

    {
        timer t;
        r = sig_kernels::sig_kernel_batch_varpar(tmp);
    }

    std::cout << r.at(0, lengthx-1, lengthy-1) << '\n';

    py::array_t<scalar_t> pyr;
    {
        py::module_ sysmodule = py::module_::import("sys");
        sysmodule.attr("path").attr("insert")(0, "/home/sam/CLionProjects/sig_kernels/venv/lib/python3.9/site-packages");
        py::module_ sigker = py::module_::import("sigkernel");

        py::array_t<scalar_t> py_ips({size_t(1), lengthx, lengthy}, ips.data());
        timer t;

        auto varpar_func = getattr(sigker, "sig_kernel_batch_varpar");

        pyr = varpar_func(py_ips, false);
    }
    std::cout << pyr.at(0, lengthx - 1, lengthy - 1) << '\n';

    sig_kernels::compute_cuda_result(X, Y, lengthx, lengthy, dimension);

}