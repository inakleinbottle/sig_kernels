//
// Created by sam on 15/11/2021.
//

#ifndef SIG_KERNELS_ANTIDIAGONAL_MATRIX_H
#define SIG_KERNELS_ANTIDIAGONAL_MATRIX_H

#include <algorithm>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cassert>

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

namespace sig_kernels {

template <typename T, template <typename, typename...> class Vector = std::vector>
class antidiagonal_view
{
    using size_type = std::size_t;
    const Vector<T>& buffer;
    const size_type start;
    const size_type finish;

public:
    __host__ __device__
    antidiagonal_view(const Vector<T>& buf, size_type s, size_type f)
        : buffer(buf), start(s), finish(f)
    {}

    __host__ __device__ const T& operator[](size_type idx) const noexcept
    {
        return buffer[start+idx];
    }

    __host__ __device__ const T* data() const noexcept
    {
        return buffer.data() + start;
    }

};

template <typename T, template<typename, typename...> class Vector = std::vector>
class mutable_antidiagonal_view
{
    using size_type = std::size_t;

    using vector_t = Vector<T>;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

private:
    pointer start, current, finish;

public:
    __host__ __device__
    mutable_antidiagonal_view(pointer s, pointer f) :start(s), finish(f), current(s)
    {}

    __host__ __device__
    const_reference operator[](size_type idx) const noexcept
    {
        return start[idx];
    }

    __host__ __device__
    reference operator[](size_type idx) noexcept
   {
       return start[idx];
   }

   __host__ __device__
           iterator begin() noexcept
   {
       return start;
   }

   __host__ __device__
           iterator end() noexcept
   {
       return finish;
   }

   __host__ __device__
           const_iterator begin() const noexcept
   {
       return start;
   }

   __host__ __device__
           const_iterator end() const noexcept
   {
       return finish;
   }

   template <typename InputIt>
   __host__ __device__ void insert(InputIt begin, InputIt end)
   {
       current = std::copy(begin, end, current);
   }

   __host__ __device__
           reference emplace_back(T arg)
   {
       *current = arg;
       return (*current++);
   }

   __host__ __device__
           reference push_back(T arg)
   {
       *current = arg;
       return (*current++);
   }

   __host__ __device__
           const_pointer data() const noexcept
   {
       return start;
   }

};


template <typename T, template<typename, typename...> class Vector = std::vector>
class antidiagonal_matrix
{
    Vector<T> buffer;
    using size_type = typename Vector<T>::size_type;

    size_type length_x, length_y, num_antidiagonals;
    Vector<size_type> antidiagonal_indices;

public:

    using view_type = antidiagonal_view<T, Vector>;
    using mutable_view_type = mutable_antidiagonal_view<T, Vector>;
    using const_reference = typename Vector<T>::const_reference;

    __host__ __device__
    antidiagonal_matrix() = default;

    __host__ __device__
    antidiagonal_matrix(size_type lx, size_type ly)
        : length_x(lx), length_y(ly), num_antidiagonals(lx+ly-1), buffer(), antidiagonal_indices()
    {
        buffer.reserve(length_x * length_y);
        antidiagonal_indices.resize(num_antidiagonals);
        auto minmax = std::minmax(length_x, length_y);

        size_type current = 0;
        for (size_type k=0; k<minmax.first; ++k) {
            antidiagonal_indices.push_back(current);
            current += k+1;
        }
        for (size_type k=minmax.first; k<minmax.second; ++k) {
            antidiagonal_indices.push_back(current);
            current += minmax.first;
        }
        for (size_type k=minmax.second; k<num_antidiagonals; ++k) {
            antidiagonal_indices.push_back(current);
            current += num_antidiagonals-k;
        }
        antidiagonal_indices.push_back(current);
        /*
        if (current != length_x*length_y) {
            throw std::runtime_error("Blah " + std::to_string(current) + " " + std::to_string(length_x*length_y));
        }
        assert(antidiagonal_indices.size() == num_antidiagonals);
        if (antidiagonal_indices.back() != length_x * length_y - 1) {
            throw std::runtime_error("Other blah");
        }
        */
    }

    __host__ __device__
    size_type capacity() const noexcept
    {
        return buffer.capacity();
    }

    __host__ __device__ void resize(size_type n)
    {
        buffer.resize(n);
    }

    template <typename Iter>
    __host__ __device__ void insert(Iter begin, Iter end)
    {
        buffer.insert(buffer.begin(), begin, end);
    }

    __host__ __device__ explicit operator Vector<T> ()
    {
        return std::move(buffer);
    }

    __host__ __device__
            view_type operator[](size_type idx) const noexcept
    {
        return view_type(buffer, antidiagonal_indices[idx], antidiagonal_indices[idx+1]);
    }

    __host__ __device__
            mutable_view_type operator[](size_type idx) noexcept
    {
        auto dptr = buffer.data();
        return mutable_view_type(dptr + antidiagonal_indices[idx], dptr + antidiagonal_indices[idx+1]);
    }

    __host__ __device__
    const_reference back() const
    {
        return buffer.back();
    }

};


}



#endif//SIG_KERNELS_ANTIDIAGONAL_MATRIX_H
