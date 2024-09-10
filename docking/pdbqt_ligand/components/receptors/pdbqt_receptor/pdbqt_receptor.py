# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import os
from typing import NamedTuple, Sequence, Type, TypeVar

import numpy as np

from pyvina_core import ReceptorBaseCpp as CorePdbqtReceptor

from pdbqt_ligand.wrappers_for_io import IowrapperDirNewlyCreated
from pdbqt_ligand.wrappers_for_third_party_tools.wrappers_for_openbabel import (
    WrapperObabel,
)

T = TypeVar("T", bound="PdbqtReceptor")


class PdbqtReceptor(object):
    """
    """

    class Pocket(NamedTuple):
        """
        """

        CENTER: np.ndarray
        SHAPE: np.ndarray
        RESOLUTION: float

        @classmethod
        def create(
            #####
            cls,
            CENTER: Sequence[float],
            SHAPE: Sequence[float],
            RESOLUTION: float,
        ) -> "PdbqtReceptor.Pocket":
            """
            """
            a_pocket = cls(
                #####
                CENTER=np.array(CENTER),
                SHAPE=np.array(SHAPE),
                RESOLUTION=float(RESOLUTION),
            )

            assert a_pocket.CENTER.shape == (3,)
            assert a_pocket.SHAPE.shape == (3,)

            return a_pocket

    @classmethod
    def from_path_pdb_receptor(
        #####
        cls: Type[T],
        path_pdb_receptor: str,
        pocket: "PdbqtReceptor.Pocket",
        path_cwd: str = None,
    ) -> T:
        """
        """
        _, abspath_pdb_receptor = IowrapperDirNewlyCreated(path_cwd).copy_files_inward(
            path_pdb_receptor
        )

        abspath_pdbqt_receptor = abspath_pdb_receptor + "_obabel.pdbqt"
        WrapperObabel().convert_pdb2pdbqt_receptor(
            path_pdb_receptor=abspath_pdb_receptor,
            path2save_pdbqt_receptor=abspath_pdbqt_receptor,
        )

        return cls.from_path_pdbqt_receptor(
            #####
            path_pdbqt_receptor=abspath_pdbqt_receptor,
            pocket=pocket,
        )

    @classmethod
    def from_path_pdbqt_receptor(
        #####
        cls: Type[T],
        path_pdbqt_receptor: str,
        pocket: "PdbqtReceptor.Pocket",
    ) -> T:
        """
        """
        abspath_pdbqt_receptor = os.path.abspath(path_pdbqt_receptor)

        assert isinstance(pocket, PdbqtReceptor.Pocket)
        core_pdbqt_receptor = CorePdbqtReceptor(*(pocket), )
        core_pdbqt_receptor.load_from_file(abspath_pdbqt_receptor)

        return cls(
            #####
            core_pdbqt_receptor=core_pdbqt_receptor,
        )

    def __init__(
        #####
        self,
        core_pdbqt_receptor: CorePdbqtReceptor,
    ) -> None:
        """
        """
        super().__init__()

        self.CORE_PDBQT_RECEPTOR = core_pdbqt_receptor
