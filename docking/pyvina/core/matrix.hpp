#pragma once
#ifndef IDOCK_MATRIX_HPP
#define IDOCK_MATRIX_HPP

using namespace std;

//! Returns the flattened 1D index of a triangular 2D index (x, y) where x is the lowest dimension.
inline size_t mr(const size_t x, const size_t y)
{
	assert(x <= y);
	return (y * (y + 1) >> 1) + x;
}

//! Returns the flattened 1D index of a triangular 2D index (x, y) where either x or y is the lowest dimension.
inline size_t mp(const size_t x, const size_t y)
{
	return x <= y ? mr(x, y) : mr(y, x);
}

#endif
