# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

from typing import Type, TypeVar

import numpy as np

from pdbqt_ligand.components.atoms.pdb_atom import PdbAtom

T = TypeVar("T", bound="PdbqtAtom")


class PdbqtAtom(PdbAtom):
    """
    """

    @classmethod
    def is_line_pdbqt_atom(cls: Type[T], line_pdbqt: str, ) -> bool:
        """
        """
        return cls.is_line_pdb_atom(line_pdbqt)

    @classmethod
    def from_line_pdbqt_atom(cls: Type[T], line_pdbqt_atom: str, ) -> T:
        """
        """
        assert cls.is_line_pdbqt_atom(line_pdbqt_atom)
        pdb_atom_temp = PdbAtom.from_line_pdb_atom(line_pdbqt_atom)

        #####01234567891123456789212345678931234567894123456789512345678961234567897123456789
        #####ATOM     18  H   UNL     1       7.467   7.961  26.142  0.00  0.00    +0.000 HD

        type_pdbqt = line_pdbqt_atom[77:79].strip()

        return cls(
            #####
            line=pdb_atom_temp.LINE,
            position=pdb_atom_temp.POSITION,
            type_pdbqt=type_pdbqt,
        )

    def __init__(
        #####
        self,
        line: str,
        position: np.ndarray,
        type_pdbqt: str,
    ) -> None:
        """
        """
        super().__init__(
            #####
            line=line,
            position=position,
        )

        self.TYPE_PDBQT = type_pdbqt.strip()
        if len(self.TYPE_PDBQT) <= 0:
            print(self.LINE)
        assert len(self.TYPE_PDBQT) > 0

    @property
    def is_hydrogen(self, ):
        """
        """
        return self.TYPE_PDBQT.casefold() in ["h", "hd"]
