//
// Created by sam on 15/11/2021.
//

#ifndef SIG_KERNELS_RANDOM_PATHS_H
#define SIG_KERNELS_RANDOM_PATHS_H

#include "implementation_types.h"
#include <vector>
#include <utility>
#include <cstddef>

using std::size_t;

std::pair<std::vector<scalar_t>, std::vector<scalar_t> > make_random_paths(size_t length_x, size_t length_y, size_t dimension);

#endif//SIG_KERNELS_RANDOM_PATHS_H
