//
// Created by sam on 15/11/2021.
//


#include "implementation_types.h"
#include <benchmark/benchmark.h>
#include "random_paths.h"
#include "kernel_backends.h"
#include "dot_product.h"
#include "bench_args.h"


namespace py = pybind11;

static void BM_varpar_kernel(benchmark::State& state)
{
    size_t lengthx=state.range(0), lengthy=state.range(1), dimension=state.range(2);
    auto p = make_random_paths(lengthx, lengthy, dimension);
    const auto* X = p.first.data();
    const auto* Y = p.second.data();


    for (auto _ : state) {

        dot_product<scalar_t> dp(dimension);

        std::vector<scalar_t> ips;
        ips.reserve(lengthx*lengthy);

        for (size_t i=0; i < lengthx; ++i) {
            for (size_t j=0; j<lengthy; ++j) {
                 ips.emplace_back(dp(X + i*dimension, Y + j*dimension));
            }
        }

        auto result = sig_kernels::sig_kernel_batch_varpar(ips.data(), lengthx, lengthy);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(static_cast<long>(lengthx * lengthy * dimension));
}

BENCHMARK(BM_varpar_kernel)->Apply(make_arguments);

static void BM_varpar_kernel_precomputed_ips(benchmark::State& state)
{
    size_t lengthx=state.range(0), lengthy=state.range(1), dimension=state.range(2);
    auto p = make_random_paths(lengthx, lengthy, dimension);
    const auto* X = p.first.data();
    const auto* Y = p.second.data();

    dot_product<scalar_t> dp(dimension);

    std::vector<scalar_t> ips;
    ips.reserve(lengthx * lengthy);

    for (size_t i = 0; i < lengthx; ++i) {
        for (size_t j = 0; j < lengthy; ++j) {
            ips.emplace_back(dp(X + i * dimension, Y + j * dimension));
        }
    }

    for (auto _ : state) {
        auto result = sig_kernels::sig_kernel_batch_varpar(ips.data(), lengthx, lengthy);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(static_cast<long>(lengthx * lengthy * dimension));
}

BENCHMARK(BM_varpar_kernel_precomputed_ips)->Apply(make_arguments);