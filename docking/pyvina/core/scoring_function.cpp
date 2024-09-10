#include <iostream>
#include <cmath>
#include <cassert>
#include "matrix.hpp"
#include "scoring_function.hpp"

#include <omp.h>

const double scoring_function::cutoff_sqr = cutoff * cutoff;
const array<double, scoring_function::n> scoring_function::vdw
{{
	1.9, //   C_H
	1.9, //   C_P
	1.8, //   N_P
	1.8, //   N_D
	1.8, //   N_A
	1.8, //   N_DA
	1.7, //   O_A
	1.7, //   O_DA
	2.0, //   S_P
	2.1, //   P_P
	1.5, //   F_H
	1.8, //  Cl_H
	2.0, //  Br_H
	2.2, //   I_H
	1.2, // Met_D
}};

//! Returns true if the XScore atom type is hydrophobic.
inline bool is_hydrophobic(const size_t t)
{
	return t ==  0 || t == 10 || t == 11 || t == 12 || t == 13;
}

//! Returns true if the XScore atom type is a hydrogen bond donor.
inline bool is_hbdonor(const size_t t)
{
	return t ==  3 || t ==  5 || t ==  7 || t == 14;
}

//! Returns true if the XScore atom type is a hydrogen bond acceptor.
inline bool is_hbacceptor(const size_t t)
{
	return t ==  4 || t ==  5 || t ==  6 || t ==  7;
}

//! Returns true if the two XScore atom types are a pair of hydrogen bond donor and acceptor.
inline bool is_hbond(const size_t t0, const size_t t1)
{
	return (is_hbdonor(t0) && is_hbacceptor(t1)) || (is_hbdonor(t1) && is_hbacceptor(t0));
}

scoring_function::scoring_function() : 
e(np, vector<double>(nr)), 
d(np, vector<double>(nr)), 
rs(nr), 
cutoff_angstrom(scoring_function::cutoff),
cutoff_sqr_angstrom2(scoring_function::cutoff_sqr),
num_samples_per_angstrom(scoring_function::ns),
num_samples_within_cutoff(scoring_function::nr),
grid_collision_only_(np, vector<double>(nr))
{
	const double ns_inv = 1.0 / ns;
	for (size_t i = 0; i < nr; ++i)
	{
		rs[i] = sqrt(i * ns_inv);
	}
	assert(rs.front() == 0);
	assert(rs.back() == cutoff);
}

bool assertNotWork = false;

double scoring_function::score(const size_t t0, const size_t t1, const double r)
{
	assert(r <= cutoff);

	if( r > cutoff ){
		if(!assertNotWork){
			assertNotWork = true;
			printf("\n\n :: The 'assert' seems not working :: (This message can be ignored) \n\n");
		}
		return 0.0;
	}

        // ZLZ: create a weight array
        // VINA ORIGINAL: const fl a[] = {-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 1.923}; // design.out227
        //const float weights[] = {-0.035579, -0.005156, 0.840245, -0.035069, -0.587439};
    //  const float weights[] = {-0.02057900, -0.00515600, 0.84024500, -0.03506900, -0.58743900}; // ZLZ: TO_BE_MODIFIED 
        //const float weights[] = {-0.035579, -0.005156, 0.840245, -0.035069, -0.607439};

        /*std::cout << "Weighting Factors: " << std::endl;
        for (int i = 0; i<sizeof(weights)/sizeof(weights[0]); ++i)
        {
           std::cout << weights[i] << std::endl;
        }*/ // ZLZ test

	const float weights[] = { -0.035579, -0.005156, 0.840245, -0.035069, -0.587439 };

	// Calculate the surface distance d.
	const double d = r - (vdw[t0] + vdw[t1]);

	// The scoring function is a weighted sum of 5 terms.
	// The first 3 terms depend on d only, while the latter 2 terms depend on t0, t1 and d.
	//
	// ZLZ: Modify original weights
	return  weights[0] * exp(-4 * d * d)
		+ weights[1] * exp(-0.25 * (d - 3.0) * (d - 3.0))
		+ weights[2] * (d > 0 ? 0.0 : d * d)
		+ weights[3] * ((is_hydrophobic(t0) && is_hydrophobic(t1)) ? ((d >= 1.5) ? 0.0 : ((d <= 0.5) ? 1.0 : 1.5 - d)) : 0.0)
		+ weights[4] * ((is_hbond(t0, t1)) ? ((d >= 0) ? 0.0 : ((d <= -0.7) ? 1 : d * (-1.4285714285714286))): 0.0);
}
double scoring_function::score_pybind11(const size_t t0, const size_t t1, const double r){
	return scoring_function::score( t0 , t1 , r );
}


double scoring_function::score_collision_only(const size_t t0, const size_t t1, const double r)
{
	const double d = r - (vdw[t0] + vdw[t1]);

	if(d>0){
		return 0.0;
	}else{
		return d*d;
	}
}


void scoring_function::score(double* const v, const size_t t0, const size_t t1, const double r2)
{
	assert(r2 <= cutoff_sqr);

	// Calculate the surface distance d.
	const double d = sqrt(r2) - (vdw[t0] + vdw[t1]);

	// The scoring function is a weighted sum of 5 terms.
	// The first 3 terms depend on d only, while the latter 2 terms depend on t0, t1 and d.
	v[0] += exp(-4 * d * d);
	v[1] += exp(-0.25 * (d - 3.0) * (d - 3.0));
	v[2] += (d > 0 ? 0.0 : d * d);
	v[3] += ((is_hydrophobic(t0) && is_hydrophobic(t1)) ? ((d >= 1.5) ? 0.0 : ((d <= 0.5) ? 1.0 : 1.5 - d)) : 0.0);
	v[4] += ((is_hbond(t0, t1)) ? ((d >= 0) ? 0.0 : ((d <= -0.7) ? 1 : d * (-1.4285714285714286))): 0.0);
}

void scoring_function::precalculate(const size_t t0, const size_t t1)
{
	const size_t p = mr(t0, t1);
	vector<double>& ep = e[p];
	vector<double>& dp = d[p];
	assert(ep.size() == nr);
	assert(dp.size() == nr);

	// Calculate the value of scoring function evaluated at (t0, t1, d).
	for (size_t i = 0; i < nr; ++i)
	{
		ep[i] = score(t0, t1, rs[i]);
	}

	// Calculate the dor of scoring function evaluated at (t0, t1, d).
	for (size_t i = 1; i < nr - 1; ++i)
	{
		dp[i] = (ep[i + 1] - ep[i]) / ((rs[i + 1] - rs[i]) * rs[i]);
	}
	dp.front() = 0;
	dp.back() = 0;

	// calculate collision term only
	vector<double>& grid_collision_only_p = grid_collision_only_[p];
	for (size_t i = 0; i < nr; ++i)
	{
		grid_collision_only_p[i] = score_collision_only(t0, t1, rs[i]);
	}
}

void scoring_function::clear()
{
	rs.clear();
}

void scoring_function::create_grid_map() {
#pragma omp parallel for collapse(1)
    for (size_t t1 = 0; t1 < this->n; ++t1) {
        for (size_t t0 = 0; t0 <= t1; ++t0)
        {
            this->precalculate(t0, t1);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
/////
/////	Torch-related methods
/////
//////////////////////////////////////////////////////////////////////////////////////////
#include "common/common.h"
torch::Tensor scoring_function::GridTtr2(
	int xs_score_1,
	int xs_score_2,
	const torch::Tensor &r2) const
{
	auto ij_x_score_index = pyvina::CommonMp(
		xs_score_1,
		xs_score_2);

	if ((r2.item<double>()) >= cutoff_sqr_angstrom2)
	{
		return torch::zeros(1);
	}

	auto r2_indx_float = int(num_samples_per_angstrom) * r2;
	auto r2_indx_0 = int(r2_indx_float.item<double>());

	if (r2_indx_0 + 1 >= int(num_samples_within_cutoff))
	{
		return torch::zeros(1);
	}

	auto energy_r2_0 = e[ij_x_score_index][r2_indx_0];

	torch::Tensor energy_final = torch::zeros(1);
	energy_final += energy_r2_0;
	energy_final += ((e[ij_x_score_index][r2_indx_0 + 1] - energy_r2_0) *
					 (r2_indx_float - r2_indx_0));
	return energy_final;

	// return (energy_r2_0 +
	// 		((e[ij_x_score_index][r2_indx_0 + 1] - energy_r2_0) *
	// 		 (r2_indx_float - r2_indx_0)));
}

//////////////////////////////////////////////////////////////////////////////////////////
/////
/////	Vanilla-related methods
/////
//////////////////////////////////////////////////////////////////////////////////////////
double scoring_function::VanillaCore_GridTtr2(
	int xs_score_1,
	int xs_score_2,
	double r2) const
{
	auto ij_x_score_index = pyvina::CommonMp(
		xs_score_1,
		xs_score_2);

	if (r2 >= cutoff_sqr_angstrom2)
	{
		return 0.0;
	}

	auto r2_indx_float = int(num_samples_per_angstrom) * r2;
	auto r2_indx_0 = int(r2_indx_float);

	if (r2_indx_0 + 1 >= int(num_samples_within_cutoff))
	{
		return 0.0;
	}

	auto energy_r2_0 = e[ij_x_score_index][r2_indx_0];

	double energy_final = 0.0;
	energy_final += energy_r2_0;
	energy_final += ((e[ij_x_score_index][r2_indx_0 + 1] - energy_r2_0) *
					 (r2_indx_float - r2_indx_0));
	return energy_final;
}


double scoring_function::score_by_grid_collision_only(
	int xs_score_1,
	int xs_score_2,
	double r2) const
{
	auto ij_x_score_index = pyvina::CommonMp(
		xs_score_1,
		xs_score_2);

	if (r2 >= cutoff_sqr_angstrom2)
	{
		return 0.0;
	}

	auto r2_indx_float = int(num_samples_per_angstrom) * r2;
	auto r2_indx_0 = int(r2_indx_float);

	if (r2_indx_0 + 1 >= int(num_samples_within_cutoff))
	{
		return 0.0;
	}

	auto energy_r2_0 = grid_collision_only_[ij_x_score_index][r2_indx_0];

	double energy_final = 0.0;
	energy_final += energy_r2_0;
	energy_final += ((grid_collision_only_[ij_x_score_index][r2_indx_0 + 1] - energy_r2_0) *
					 (r2_indx_float - r2_indx_0));
	return energy_final;
}


//////////////////////////////////////////////////////////////////////////////////////////
/////
/////	FormulaDerivative-related methods
/////
//////////////////////////////////////////////////////////////////////////////////////////
double scoring_function::FormulaDerivativeCore_GridTtr2DerivativeToR2(
	int xs_score_1,
	int xs_score_2,
	double r2) const
{
	auto ij_x_score_index = pyvina::CommonMp(
		xs_score_1,
		xs_score_2);

	if (r2 >= cutoff_sqr_angstrom2)
	{
		return 0.0;
	}

	auto r2_indx_float = int(num_samples_per_angstrom) * r2;
	auto r2_indx_0 = int(r2_indx_float);

	if (r2_indx_0 + 1 >= int(num_samples_within_cutoff))
	{
		return 0.0;
	}

	auto energy_delta = (e[ij_x_score_index][r2_indx_0 + 1] -
						 e[ij_x_score_index][r2_indx_0]);
	return (energy_delta * int(num_samples_per_angstrom));

	// auto energy_r2_0 = e[ij_x_score_index][r2_indx_0];

	// return (
	// 	(e[ij_x_score_index][r2_indx_0 + 1] - energy_r2_0) *
	// 	int(num_samples_per_angstrom)
	// );

	// double energy_final = 0.0;
	// energy_final += energy_r2_0;
	// energy_final += ((e[ij_x_score_index][r2_indx_0 + 1] - energy_r2_0) *
	// 				 (r2_indx_float - r2_indx_0));
	// return energy_final;
}


double scoring_function::get_derivative_by_grid_collision_only(
	int xs_score_1,
	int xs_score_2,
	double r2) const
{
	auto ij_x_score_index = pyvina::CommonMp(
		xs_score_1,
		xs_score_2);

	if (r2 >= cutoff_sqr_angstrom2)
	{
		return 0.0;
	}

	auto r2_indx_float = int(num_samples_per_angstrom) * r2;
	auto r2_indx_0 = int(r2_indx_float);

	if (r2_indx_0 + 1 >= int(num_samples_within_cutoff))
	{
		return 0.0;
	}
	if (r2_indx_0 - 1 < 0)
	{
		return 0.0;
	}

	auto energy_delta = (grid_collision_only_[ij_x_score_index][r2_indx_0 + 1] -
						 grid_collision_only_[ij_x_score_index][r2_indx_0 - 1]);

	energy_delta *= 0.5;

	return (energy_delta * int(num_samples_per_angstrom));

}

