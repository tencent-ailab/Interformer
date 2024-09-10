#include <cmath>
#include <cassert>
#include "array.hpp"

//! Returns true if the absolute difference between a scalar and zero is within the constant tolerance.
inline bool zero(const double a)
{
	return fabs(a) < 1e-4;
}

//! Returns true if the absolute difference between two scalars is within the constant tolerance.
inline bool equal(const double a, const double b)
{
	return zero(a - b);
}

//! Returns true is a vector is approximately (0, 0, 0).
bool zero(const array<double, 3>& a)
{
	return zero(a[0]) && zero(a[1]) && zero(a[2]);
}

//! Returns the square norm of a vector.
double norm_sqr(const array<double, 3>& a)
{
	return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

//! Returns the square norm of a vector.
double norm_sqr(const array<double, 4>& a)
{
	return a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
}

//! Returns the norm of a vector.
double norm(const array<double, 3>& a)
{
	return sqrt(norm_sqr(a));
}

//! Returns the norm of a vector.
double norm(const array<double, 4>& a)
{
	return sqrt(norm_sqr(a));
}

//! Returns true if the norm of a vector is approximately 1.
bool normalized(const array<double, 3>& a)
{
	return equal(norm(a), 1);
}

//! Returns true if the norm of a vector is approximately 1.
bool normalized(const array<double, 4>& a)
{
	return equal(norm(a), 1);
}

//! Returns the normalized vector of a vector.
array<double, 3> normalize(const array<double, 3>& a)
{
	return (1 / norm(a)) * a;
}

//! Returns the normalized vector of a vector.
array<double, 4> normalize(const array<double, 4>& a)
{
	const auto norm_inv = 1 / norm(a);
	return
	{{
		norm_inv * a[0],
		norm_inv * a[1],
		norm_inv * a[2],
		norm_inv * a[3],
	}};
}

//! Returns the dot product of two vectors.
double operator*(const array<double, 3>& a, const array<double, 3>& b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

//! Returns the cross product of two vectors.
array<double, 3> cross(const array<double, 3>& a, const array<double, 3>& b)
{
	return
	{{
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0],
	}};
}

//! Returns the resulting vector of elementwise multiplication of a scalar and a vector.
array<double, 3> operator*(const double f, const array<double, 3>& a)
{
	return
	{{
		f * a[0],
		f * a[1],
		f * a[2],
	}};
}

//! Returns the resulting vector of elementwise multiplication of two vectors.
array<double, 3> operator*(const array<double, 3>& a, const array<size_t, 3>& b)
{
	return
	{{
		a[0] * b[0],
		a[1] * b[1],
		a[2] * b[2],
	}};
}

//! Returns the resulting vector of elementwise addition of two vectors.
array<double, 3> operator+(const array<double, 3>& a, const array<double, 3>& b)
{
	return
	{{
		a[0] + b[0],
		a[1] + b[1],
		a[2] + b[2],
	}};
}

//! Returns the resulting vector of elementwise subtraction of two vectors.
array<double, 3> operator-(const array<double, 3>& a, const array<double, 3>& b)
{
	return
	{{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
	}};
}

//! Adds the second vector to the first vector.
void operator+=(array<double, 3>& a, const array<double, 3>& b)
{
	a[0] += b[0];
	a[1] += b[1];
	a[2] += b[2];
}

//! Subtracts the second vector from the first vector.
void operator-=(array<double, 3>& a, const array<double, 3>& b)
{
	a[0] -= b[0];
	a[1] -= b[1];
	a[2] -= b[2];
}

//! Returns the square Euclidean distance between two vectors.
double distance_sqr(const array<double, 3>& a, const array<double, 3>& b)
{
	return norm_sqr(a - b);
}

//! Returns the accumulated square Euclidean distance between two vectors of vectors.
double distance_sqr(const vector<array<double, 3>>& a, const vector<array<double, 3>>& b)
{
	const size_t n = a.size();
	assert(n > 0);
	assert(n == b.size());
	double sum = 0;
	for (size_t i = 0; i < n; ++i)
	{
		sum += distance_sqr(a[i], b[i]);
	}
	return sum;
}

//! Converts a vector of size 4 to a quaternion.
array<double, 4> vec4_to_qtn4(const array<double, 3>& axis, const double angle)
{
	assert(normalized(axis));
	const double h = angle * 0.5;
	const double s = sin(h);
	const double c = cos(h);
	return
	{{
		c,
		s * axis[0],
		s * axis[1],
		s * axis[2],
	}};
}

//! Converts a vector of size 3 to a quaternion.
array<double, 4> vec3_to_qtn4(const array<double, 3>& rotation)
{
	if (zero(rotation))
	{
		return {{1, 0, 0, 0}};
	}
	else
	{
		const auto angle = norm(rotation);
		const auto axis = (1 / angle) * rotation;
		return vec4_to_qtn4(axis, angle);
	}
}

//! Converts a quaternion to a 3x3 matrix.
//! http://www.boost.org/doc/libs/1_55_0/libs/math/quaternion/TQE.pdf
//! http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
array<double, 9> qtn4_to_mat3(const array<double, 4>& a)
{
	assert(normalized(a));
	const auto ww = a[0]*a[0];
	const auto wx = a[0]*a[1];
	const auto wy = a[0]*a[2];
	const auto wz = a[0]*a[3];
	const auto xx = a[1]*a[1];
	const auto xy = a[1]*a[2];
	const auto xz = a[1]*a[3];
	const auto yy = a[2]*a[2];
	const auto yz = a[2]*a[3];
	const auto zz = a[3]*a[3];
	return
	{{
		ww+xx-yy-zz, 2*(-wz+xy), 2*(wy+xz),
		2*(wz+xy), ww-xx+yy-zz, 2*(-wx+yz),
		2*(-wy+xz), 2*(wx+yz), ww-xx-yy+zz,
	}};
}

//! Transforms a vector by a 3x3 matrix.
array<double, 3> operator*(const array<double, 9>& m, const array<double, 3>& v)
{
	return
	{{
		m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
		m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
		m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
	}};
}

//! Returns the product of two quaternions.
array<double, 4> operator*(const array<double, 4>& a, const array<double, 4>& b)
{
	return
	{{
		a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
		a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
		a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
		a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
	}};
}
