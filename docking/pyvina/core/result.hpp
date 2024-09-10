#pragma once
#ifndef IDOCK_RESULT_HPP
#define IDOCK_RESULT_HPP

#include <vector>

//! Represents a result found by BFGS local optimization for later clustering.
class result
{
public:
	double e; //!< Free energy.
	double f; //!< Inter-molecular free energy.
	double e_nd; //!< Normalized free energy.
	double rf; //!< RF-Score binding affinity.
	vector<array<double, 3>> heavy_atoms; //!< Heavy atom coordinates.
	vector<array<double, 3>> hydrogens; //!< Hydrogen atom coordinates.

	//! Constructs a result from free energy e, force f, heavy atom coordinates and hydrogen atom coordinates.
	explicit result(const double e, const double f, vector<array<double, 3>>&& heavy_atoms_, vector<array<double, 3>>&& hydrogens_) : e(e), f(f), heavy_atoms(move(heavy_atoms_)), hydrogens(move(hydrogens_)) {}

	result(const result&) = default;
	result(result&&) = default;
	result& operator=(const result&) = default;
	result& operator=(result&&) = default;

	//! For sorting vector<result>.
	bool operator<(const result& r) const
	{
		return e < r.e;
	}

	//! Clusters a result into an existing result set with a minimum RMSD requirement.
	static void push(vector<result>& results, result&& r, const double required_square_error);
};

#endif
