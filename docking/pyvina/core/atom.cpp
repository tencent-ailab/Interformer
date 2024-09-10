#include <algorithm>
#include <cassert>
#include "array.hpp"
#include "atom.hpp"

//! AutoDock4 atom type strings, e.g. H, HD, C, A.
const array<string, atom::n> atom::ad_strings
{{
	"H" , //  0
	"HD", //  1
	"C" , //  2
	"A" , //  3
	"N" , //  4
	"NA", //  5
	"OA", //  6
	"S" , //  7
	"SA", //  8
	"Se", //  9
	"P" , // 10
	"F" , // 11
	"Cl", // 12
	"Br", // 13
	"I" , // 14
	"Zn", // 15
	"Fe", // 16
	"Mg", // 17
	"Ca", // 18
	"Mn", // 19
	"Cu", // 20
	"Na", // 21
	"K" , // 22
	"Hg", // 23
	"Ni", // 24
	"Co", // 25
	"Cd", // 26
	"As", // 27
	"Sr", // 28
	"U" , // 29
	"Cs", // 30
	"Mo", // 31
	"Si", // 32
}};

//! Covalent radii of AutoDock4 atom types, factorized by 1.1 for extra allowance.
//! http://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
const array<double, atom::n> atom::ad_covalent_radii
{{
	0.407, //  0 = H , 0.407 = 1.1 * 0.37
	0.407, //  1 = HD, 0.407 = 1.1 * 0.37
	0.847, //  2 = C , 0.847 = 1.1 * 0.77
	0.847, //  3 = A , 0.847 = 1.1 * 0.77
	0.825, //  4 = N , 0.825 = 1.1 * 0.75
	0.825, //  5 = NA, 0.825 = 1.1 * 0.75
	0.803, //  6 = OA, 0.803 = 1.1 * 0.73
	1.122, //  7 = S , 1.122 = 1.1 * 1.02
	1.122, //  8 = SA, 1.122 = 1.1 * 1.02
	1.276, //  9 = Se, 1.276 = 1.1 * 1.16
	1.166, // 10 = P , 1.166 = 1.1 * 1.06
	0.781, // 11 = F , 0.781 = 1.1 * 0.71
	1.089, // 12 = Cl, 1.089 = 1.1 * 0.99
	1.254, // 13 = Br, 1.254 = 1.1 * 1.14
	1.463, // 14 = I , 1.463 = 1.1 * 1.33
	1.441, // 15 = Zn, 1.441 = 1.1 * 1.31
	1.375, // 16 = Fe, 1.375 = 1.1 * 1.25
	1.430, // 17 = Mg, 1.430 = 1.1 * 1.30
	1.914, // 18 = Ca, 1.914 = 1.1 * 1.74
	1.529, // 19 = Mn, 1.529 = 1.1 * 1.39
	1.518, // 20 = Cu, 1.518 = 1.1 * 1.38
	1.694, // 21 = Na, 1.694 = 1.1 * 1.54
	2.156, // 22 = K , 2.156 = 1.1 * 1.96
	1.639, // 23 = Hg, 1.639 = 1.1 * 1.49
	1.331, // 24 = Ni, 1.331 = 1.1 * 1.21
	1.386, // 25 = Co, 1.386 = 1.1 * 1.26
	1.628, // 26 = Cd, 1.628 = 1.1 * 1.48
	1.309, // 27 = As, 1.309 = 1.1 * 1.19
	2.112, // 28 = Sr, 2.112 = 1.1 * 1.92
	2.156, // 29 = U , 2.156 = 1.1 * 1.96
	2.475, // 30 = Cs, 2.475 = 1.1 * 2.25
	1.595, // 31 = Mo, 1.595 = 1.1 * 1.45
	1.221, // 32 = Si, 1.221 = 1.1 * 1.11
}};

//! Mapping from AutoDock4 atom types to XScore atom types.
const array<size_t, atom::n> atom::ad_to_xs
{{
	 n, //  0 = H  -> dummy
	 n, //  1 = HD -> dummy
	 0, //  2 = C  -> C_H   =  0, Carbon, hydrophobic, not bonded to a hetero atom.
	 0, //  3 = A  -> C_H   =  0, Carbon, hydrophobic, not bonded to a hetero atom.
	 2, //  4 = N  -> N_P   =  2, Nitrogen, neither hydrogen bond donor nor acceptor.
	 4, //  5 = NA -> N_A   =  4, Nitrogen, hydrogen bond acceptor.
	 6, //  6 = OA -> O_A   =  6, Oxygen, hydrogen bond acceptor.
	 8, //  7 = S  -> S_P   =  8, Sulfur or Selenium.
	 8, //  8 = SA -> S_P   =  8, Sulfur or Selenium.
	 8, //  9 = Se -> S_P   =  8, Sulfur or Selenium.
	 9, // 10 = P  -> P_P   =  9, Phosphorus.
	10, // 11 = F  -> F_H   = 10, Fluorine, hydrophobic.
	11, // 12 = Cl -> Cl_H  = 11, Chlorine, hydrophobic.
	12, // 13 = Br -> Br_H  = 12, Bromine, hydrophobic.
	13, // 14 = I  -> I_H   = 13, Iodine, hydrophobic.
	14, // 15 = Zn -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 16 = Fe -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 17 = Mg -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 18 = Ca -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 19 = Mn -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 20 = Cu -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 21 = Na -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 22 = K  -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 23 = Hg -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 24 = Ni -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 25 = Co -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 26 = Cd -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 27 = As -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 28 = Sr -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 29 = U  -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 30 = Cs -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 31 = Mo -> Met_D = 14, Metal, hydrogen bond donor.
	14, // 32 = Si -> Met_D = 14, Metal, hydrogen bond donor.
}};

//! Mapping from AutoDock4 atom types to RF-Score atom types.
const array<size_t, atom::n> atom::ad_to_rf
{{
	n, //  0 = H  -> dummy
	n, //  1 = HD -> dummy
	0, //  2 = C  -> C  = 0
	0, //  3 = A  -> C  = 0
	1, //  4 = N  -> N  = 1
	1, //  5 = NA -> N  = 1
	2, //  6 = OA -> O  = 2
	3, //  7 = S  -> S  = 3
	3, //  8 = SA -> S  = 3
	n, //  9 = Se -> dummy
	4, // 10 = P  -> P  = 4
	5, // 11 = F  -> F  = 5
	6, // 12 = Cl -> Cl = 6
	7, // 13 = Br -> Br = 7
	8, // 14 = I  -> I  = 8
	n, // 15 = Zn -> dummy
	n, // 16 = Fe -> dummy
	n, // 17 = Mg -> dummy
	n, // 18 = Ca -> dummy
	n, // 19 = Mn -> dummy
	n, // 20 = Cu -> dummy
	n, // 21 = Na -> dummy
	n, // 22 = K  -> dummy
	n, // 23 = Hg -> dummy
	n, // 24 = Ni -> dummy
	n, // 25 = Co -> dummy
	n, // 26 = Cd -> dummy
	n, // 27 = As -> dummy
	n, // 28 = Sr -> dummy
	n, // 29 = U  -> dummy
	n, // 30 = Cs -> dummy
	n, // 31 = Mo -> dummy
	n, // 32 = Si -> dummy
}};

//! Constructs an atom from an ATOM/HETATM line in PDBQT format.
atom::atom(const string& line) :
	serial(stoul(line.substr(6, 5))),
	name(line.substr(12, 4)),
	coord({{stod(line.substr(30, 8)), stod(line.substr(38, 8)), stod(line.substr(46, 8))}}),
	ad(find(ad_strings.cbegin(), ad_strings.cend(), line.substr(77, isspace(line[78]) ? 1 : 2)) - ad_strings.cbegin()),
	xs(ad_to_xs[ad]),
	rf(ad_to_rf[ad]),

	// Extra atom fields to save (atom properties newly added)
	partial_charge(
		stod(line.substr(66,10))
	)

{
}

//! Returns true if the AutoDock4 atom type is not supported.
bool atom::ad_unsupported() const
{
	return ad == n;
}

//! Returns true if the XScore atom type is not supported.
bool atom::xs_unsupported() const
{
	return xs == n;
}

//! Returns true if the RF-Score atom type is not supported.
bool atom::rf_unsupported() const
{
	return rf == n;
}

//! Returns true if the atom is a nonpolar hydrogen atom.
bool atom::is_nonpolar_hydrogen() const
{
	return ad == 0;
}

//! Returns true if the atom is polar hydrogen.
bool atom::is_polar_hydrogen() const
{
	return ad == 1;
}

//! Returns true if the atom is hydrogen.
bool atom::is_hydrogen() const
{
	return ad <= 1;
}

//! Returns true if the atom is a hetero atom, i.e. non-carbon heavy atom.
bool atom::is_hetero() const
{
	return ad >= 4;
}

//! Returns the covalent radius of current AutoDock4 atom type.
double atom::covalent_radius() const
{
	return ad_covalent_radii[ad];
}

//! Returns true if the current atom is covalently bonded to a given atom.
bool atom::is_neighbor(const atom& a) const
{
	assert(this != &a);
	const double r = covalent_radius() + a.covalent_radius();
	return distance_sqr(coord, a.coord) < r * r;
}

//! For nitrogen and oxygen, revises the XScore atom type to make it a hydrogen bond donor.
void atom::donorize()
{
	switch (xs)
	{
		case 2 : xs = 3; break; // Nitrogen, hydrogen bond donor.
		case 4 : xs = 5; break; // Nitrogen, both hydrogen bond donor and acceptor.
		case 6 : xs = 7; break; // Oxygen, both hydrogen bond donor and acceptor.
	}
}

//! For carbon, revises the XScore atom type to make it non-hydrophobic.
void atom::dehydrophobicize()
{
	assert(xs <= 1);
	xs = 1; // Carbon, bonded to a hetero atom.
}
