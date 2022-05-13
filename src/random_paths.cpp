//
// Created by sam on 15/11/2021.
//
#include "implementation_types.h"
#include "random_paths.h"

#include <random>

std::pair<std::vector<scalar_t>, std::vector<scalar_t>> make_random_paths(size_t length_x, size_t length_y, size_t dimension)
{
    std::pair<std::vector<scalar_t>, std::vector<scalar_t> > result;


    std::random_device df;
    std::mt19937 rng(df());
    std::normal_distribution<scalar_t> dist(0.0, 0.01);

    result.first.reserve(length_x * dimension);
    for (size_t i=0; i<length_x; ++i) {
        for (size_t j=0; j<dimension; ++j) {
            result.first.emplace_back(dist(rng));
        }
    }

    result.second.reserve(length_y * dimension);
    for (size_t i=0; i<length_y; ++i) {
        for (size_t j=0; j<dimension; ++j) {
            result.second.emplace_back(dist(rng));
        }
    }

    return result;
}
