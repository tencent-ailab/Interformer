#!/usr/bin/env python

import sys, os

from openbabel import openbabel as ob
from openbabel import pybel


# syntax:
# molml.py [files]

def atomType(mol, atomIdx):
    # get the atomic type given an atom index
    return mol.OBMol.GetAtom(atomIdx).GetType()


# repeat through all the files on the command-line
# we can change this to use the glob module as well
#  e.g., find all the files in a set of folders
for argument in sys.argv[1:]:
    filename, extension = os.path.splitext(argument)

    # read the molecule from the supplied file
    mol = next(pybel.readfile(extension[1:], argument))

    print(mol.energy)  # in kcal/mol
    # ideally, we should turn this into an atomization energy

    # iterate through all atoms
    #  .. this is commented out because Bag Of Bonds doesn't use atomic charges
    # for atom in mol.atoms:
    #    print "Atom %d, %8.4f" % (atom.type, atom.partialcharge)

    # iterate through all bonds
    bonds = []
    for bond in ob.OBMolBondIter(mol.OBMol):
        begin = atomType(mol, bond.GetBeginAtomIdx())
        end = atomType(mol, bond.GetEndAtomIdx())
        if (end < begin):
            # swap them for lexographic order
            begin, end = end, begin
        bonds.append("Bond %s-%s, %8.4f" % (begin, end, bond.GetLength()))
        print(bonds[-1])

    # iterate through all angles
    angles = []
    for angle in ob.OBMolAngleIter(mol.OBMol):
        a = (angle[0] + 1)
        b = mol.OBMol.GetAtom(angle[1] + 1)
        c = (angle[2] + 1)

        aType = atomType(mol, a)
        cType = atomType(mol, c)
        if (cType < aType):
            # swap them for lexographic order
            aType, cType = cType, aType
        angles.append("Angle %s-%s-%s, %8.3f" % (aType, b.GetType(), cType, b.GetAngle(a, c)))
        print(angles[-1])

    # iterate through all torsions
    torsions = []
    for torsion in ob.OBMolTorsionIter(mol.OBMol):
        a = (torsion[0] + 1)
        b = (torsion[1] + 1)
        c = (torsion[2] + 1)
        d = (torsion[3] + 1)

        aType = atomType(mol, a)
        bType = atomType(mol, b)
        cType = atomType(mol, c)
        dType = atomType(mol, d)

        # output in lexographic order
        if (aType < dType):
            torsions.append(
                "Torsion %s-%s-%s-%s, %8.3f" % (aType, bType, cType, dType, mol.OBMol.GetTorsion(a, b, c, d)))
        else:
            torsions.append(
                "Torsion %s-%s-%s-%s, %8.3f" % (dType, cType, bType, aType, mol.OBMol.GetTorsion(a, b, c, d)))
        print(torsions[-1])

    nb = []
    for pair in ob.OBMolPairIter(mol.OBMol):
        (first, second) = pair
        begin = atomType(mol, first)
        end = atomType(mol, second)
        if (end < begin):
            # swap them for lexographic order
            begin, end = end, begin
        dist = mol.OBMol.GetAtom(first).GetDistance(second)
        nb.append("NB %s-%s, %8.4f" % (begin, end, dist))
        print(nb[-1])
