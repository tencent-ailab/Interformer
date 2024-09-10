# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

from typing import Type, TypeVar

import numpy as np

T = TypeVar("T", bound="PdbAtom")


class PdbAtom(object):
    """
    """

    @classmethod
    def is_line_pdb_atom(cls: Type[T], line_pdb: str, ) -> bool:
        """
        """
        if line_pdb.startswith("ATOM"):
            return True
        if line_pdb.startswith("HETATM"):
            return True
        return False

    @classmethod
    def from_line_pdb_atom(cls: Type[T], line_pdb_atom: str, ) -> T:
        """
        """
        assert cls.is_line_pdb_atom(line_pdb_atom)

        return cls(
            line=line_pdb_atom,
            position=np.array(
                (
                    #####
                    float(line_pdb_atom[30:38]),
                    float(line_pdb_atom[38:46]),
                    float(line_pdb_atom[46:54]),
                )
            ),
        )

    def __init__(
        #####
        self,
        line: str,
        position: np.ndarray,
    ) -> None:
        """
        """
        super().__init__()

        self.LINE = line
        assert self.is_line_pdb_atom(self.LINE)

        self.POSITION = position
