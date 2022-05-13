//
// Created by sam on 25/11/2021.
//

#include "kernel_compute_cuda.h"
#include <benchmark/benchmark.h>

#include "random_paths.h"
#include "bench_args.h"


static void BM_kernel_compute_cuda(benchmark::State& state)
{
    size_t lengthx=state.range(0), lengthy=state.range(1), dimension=state.range(2);

    auto p = make_random_paths(lengthx, lengthy, dimension);
    auto X = p.first;
    auto Y = p.second;

    for (auto _ : state) {
        auto result = sig_kernels::compute_cuda_result_in(X, Y, lengthx, lengthy, dimension);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_kernel_compute_cuda)->Apply(make_arguments);