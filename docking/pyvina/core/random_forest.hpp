#pragma once
#ifndef IDOCK_RANDOM_FOREST_HPP
#define IDOCK_RANDOM_FOREST_HPP

#include <vector>
#include <array>
#include <random>
#include <mutex>
#include <functional>
using namespace std;

//! Represents a node in a tree.
class node
{
public:
	vector<size_t> samples; //!< Node samples.
	double y; //!< Average of y values of node samples.
	double p; //!< Node purity.
	size_t var; //!< Variable used for node split.
	double val; //!< Value used for node split.
	array<size_t, 2> children; //!< Two child nodes.

	//! Constructs an empty node.
	explicit node();
};

//! Represents a tree in a forest.
class tree : public vector<node>
{
public:
	static const size_t nv = 42; //!< Number of variables.
	static const size_t ns = 3704; //!< Number of training samples.

	//! Trains an empty tree from bootstrap samples.
	void train(const size_t mtry, const function<double()> u01);

	//! Predicts the y value of the given sample x.
	double operator()(const array<double, nv>& x) const;

	//! Clears node samples to save memory.
	void clear();
private:
	static const array<array<double, nv>, ns> x; //!< Variables of training samples.
	static const array<double, ns> y; //!< Measured binding affinities of training samples.
};

//! Represents a random forest.
class forest : public vector<tree>
{
public:
	//! Constructs a random forest of a number of empty trees.
	forest(const size_t nt, const size_t seed);

	//! Predicts the y value of the given sample x.
	double operator()(const array<double, tree::nv>& x) const;

	//! Clears node samples to save memory.
	void clear();

	//! Returns a random value from uniform distribution in [0, 1] in a thread safe manner.
	const function<double()> u01_s;
private:
	double nt_inv; //!< Inverse of the number of trees.
	mt19937_64 rng;
	uniform_real_distribution<double> uniform_01;
	mutable mutex m;
};

#endif
