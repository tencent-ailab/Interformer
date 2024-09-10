#pragma once
#ifndef IDOCK_SCORING_FUNCTION_HPP
#define IDOCK_SCORING_FUNCTION_HPP

#include <vector>
#include <array>

#include "torch/extension.h"

using namespace std;

//! Represents a scoring function.
class scoring_function
{
public:
	static const size_t n = 15; //!< Number of XScore atom types.
	static const size_t np = n*(n+1)>>1; //!< Number of XScore atom type pairs.
	static const size_t ns = 1024; //!< Number of samples in a unit distance.
	static const size_t cutoff = 8; //!< Atom type pair distance cutoff.
	static const size_t nr = ns*cutoff*cutoff+1; //!< Number of samples within the entire cutoff.
	static const double cutoff_sqr; //!< Cutoff square.

	//! Constructs an empty scoring function.
	explicit scoring_function();

	//! Returns the score between two atoms of XScore atom types t0 and t1 with distance r.
	static double score(const size_t t0, const size_t t1, const double r);
	static double score_pybind11(const size_t t0, const size_t t1, const double r);
	static double score_collision_only(const size_t t0, const size_t t1, const double r);

	//! Accumulates the unweighted score between two atoms of XScore atom types t0 and t1 with square distance r2.
	static void score(double* const v, const size_t t0, const size_t t1, const double r2);

	//! Precalculates the scoring function values of sample points for the type combination of t0 and t1.
	void precalculate(const size_t t0, const size_t t1);

	//! Clears precalculated values.
	void clear();

    void create_grid_map();

	vector<vector<double>> e; //!< Scoring function values.
	vector<vector<double>> d; //!< Scoring function derivatives divided by distance.

	const size_t num_samples_per_angstrom;
	const size_t num_samples_within_cutoff;
	const size_t cutoff_angstrom;
	const double cutoff_sqr_angstrom2;

    vector<vector<double>> grid_collision_only_; //the precalculated only for collision

	//////////////////////////////////////////////////////////////////////////////////////////
	/////
	/////	Torch-related methods
	/////
	//////////////////////////////////////////////////////////////////////////////////////////
	torch::Tensor GridTtr2(int xs_score_1,
						   int xs_score_2,
						   const torch::Tensor &r2) const;

	//////////////////////////////////////////////////////////////////////////////////////////
	/////
	/////	Vanilla-related methods
	/////
	//////////////////////////////////////////////////////////////////////////////////////////
	double VanillaCore_GridTtr2(int xs_score_1,
								int xs_score_2,
								double r2) const;
	double score_by_grid_collision_only(
		int xs_score_1,
		int xs_score_2,
		double r2) const;

	//////////////////////////////////////////////////////////////////////////////////////////
	/////
	/////	FormulaDerivative-related methods
	/////
	//////////////////////////////////////////////////////////////////////////////////////////
	double FormulaDerivativeCore_GridTtr2DerivativeToR2(int xs_score_1,
														int xs_score_2,
														double r2) const;
	double get_derivative_by_grid_collision_only(
		int xs_score_1,
		int xs_score_2,
		double r2) const;

private:
	static const array<double, n> vdw; //!< Van der Waals distances for XScore atom types.
	vector<double> rs; //!< Distance samples.
};

#endif
