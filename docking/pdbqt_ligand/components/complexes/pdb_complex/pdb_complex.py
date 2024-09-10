# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import os
from typing import List, Type, TypeVar

from pdbqt_ligand.components.atoms.pdb_atom import PdbAtom

T = TypeVar("T", bound="PdbComplex")


class PdbComplex(object):
    """
    """

    @classmethod
    def from_path_pdb_complex(
        #####
        cls: Type[T],
        path_pdb_complex: str,
        is_receptor_then_ligand: bool = False,
        num_ligand_atoms: int = None,
    ) -> T:
        """
        """
        abspath_pdb_complex = os.path.abspath(path_pdb_complex)
        lines_pdb_complex = None

        with open(abspath_pdb_complex) as file_to_read:
            lines_pdb_complex = file_to_read.read().splitlines()

        return cls.from_lines_pdb_complex(
            #####
            lines_pdb_complex=lines_pdb_complex,
            is_receptor_then_ligand=is_receptor_then_ligand,
            num_ligand_atoms=num_ligand_atoms,
        )

    @classmethod
    def from_lines_pdb_complex(
        #####
        cls: Type[T],
        lines_pdb_complex: List[str],
        is_receptor_then_ligand: bool = False,
        num_ligand_atoms: int = None,
    ) -> T:
        """
        """
        atoms_new: List[PdbAtom] = [
            PdbAtom.from_line_pdb_atom(x)
            for x in lines_pdb_complex
            if PdbAtom.is_line_pdb_atom(x)
        ]

        return cls(
            #####
            atoms=atoms_new,
            is_receptor_then_ligand=is_receptor_then_ligand,
            num_ligand_atoms=num_ligand_atoms,
        )

    def __init__(
        #####
        self,
        atoms: List[PdbAtom],
        is_receptor_then_ligand: bool,
        num_ligand_atoms: int,
    ) -> None:
        """
        """
        super().__init__()

        assert isinstance(atoms, List)
        for i_atom in atoms:
            assert isinstance(i_atom, PdbAtom)
        self.ATOMS = atoms

        self.IS_RECEPTOR_THEN_LIGAND = is_receptor_then_ligand
        self.NUM_LIGAND_ATOMS = num_ligand_atoms
