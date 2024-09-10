#pragma once
#ifndef IDOCK_ARRAY_HPP
#define IDOCK_ARRAY_HPP

#include <array>
#include <vector>
using namespace std;

double norm_sqr(const array<double, 3>& a);
double norm_sqr(const array<double, 4>& a);
double norm(const array<double, 3>& a);
double norm(const array<double, 4>& a);
bool normalized(const array<double, 3>& a);
bool normalized(const array<double, 4>& a);
array<double, 3> normalize(const array<double, 3>& a);
array<double, 4> normalize(const array<double, 4>& a);

double operator*(const array<double, 3>& a, const array<double, 3>& b);
array<double, 3> cross(const array<double, 3>& a, const array<double, 3>& b);
array<double, 3> operator*(const double f, const array<double, 3>& a);
array<double, 3> operator*(const array<double, 3>& a, const array<size_t, 3>& b);
array<double, 3> operator+(const array<double, 3>& a, const array<double, 3>& b);
array<double, 3> operator-(const array<double, 3>& a, const array<double, 3>& b);
void operator+=(array<double, 3>& a, const array<double, 3>& b);
void operator-=(array<double, 3>& a, const array<double, 3>& b);
double distance_sqr(const array<double, 3>& a, const array<double, 3>& b);
double distance_sqr(const vector<array<double, 3>>& a, const vector<array<double, 3>>& b);

array<double, 4> vec4_to_qtn4(const array<double, 3>& axis, const double angle);
array<double, 4> vec3_to_qtn4(const array<double, 3>& rotation);
array<double, 9> qtn4_to_mat3(const array<double, 4>& a);
array<double, 3> operator*(const array<double, 9>& m, const array<double, 3>& v);
array<double, 4> operator*(const array<double, 4>& a, const array<double, 4>& b);

#endif
