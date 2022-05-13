//
// Created by sam on 15/11/2021.
//


#include "bench_args.h"
#include <vector>

using std::size_t;


void make_arguments(benchmark::internal::Benchmark* b)
{
    std::vector<long> sizes{1000, 1500, 2000, 10000};
    std::vector<long> dims{5};

    for (auto length : sizes) {
        for (auto dim : dims) {
            b->Args({length, length, dim});
        }
    }
}
