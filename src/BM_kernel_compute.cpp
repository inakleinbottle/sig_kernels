//
// Created by sam on 15/11/2021.
//
#include "implementation_types.h"
#include "kernel_compute.h"
#include <benchmark/benchmark.h>

#include "random_paths.h"
#include "dot_product.h"
#include "bench_args.h"


static void BM_kernel_compute(benchmark::State& state)
{
    size_t lengthx=state.range(0), lengthy=state.range(1), dimension=state.range(2);

    auto p = make_random_paths(lengthx, lengthy, dimension);
    auto X = p.first;
    auto Y = p.second;

    dot_product<scalar_t> dp(dimension);

    for (auto _ : state) {
        auto result = sig_kernels::compute_antidiagonals(X.data(), Y.data(), lengthx, lengthy, dimension, dp);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(static_cast<long>(lengthx * lengthy * dimension));
}

BENCHMARK(BM_kernel_compute)->Apply(make_arguments);
//BENCHMARK(BM_kernel_compute)->Args({20000, 20000, 5})->MinTime(60.0);