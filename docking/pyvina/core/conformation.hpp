#pragma once
#ifndef IDOCK_CONFORMATION_HPP
#define IDOCK_CONFORMATION_HPP

#include <array>
#include <vector>
using namespace std;

//! Represents a ligand conformation.
class conformation
{
public:
	array<double, 3> position; //!< Ligand origin coordinate.
	array<double, 4> orientation; //!< Ligand orientation.
	vector<double> torsions; //!< Ligand torsions.

	//! Constructs an initial conformation.
	explicit conformation(const size_t num_active_torsions) : position{}, orientation({{1, 0, 0, 0}}), torsions(num_active_torsions, 0) {}
};

//! Represents a transition from one conformation to another.
class change : public vector<double>
{
public:
	//! Constructs a zero change.
	explicit change(const size_t num_active_torsions) : vector<double>(6 + num_active_torsions, 0) {}
};

#endif
