# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import os
from typing import Any, Dict, Sequence, Type, TypeVar

import numpy as np
from rdkit import Chem

T = TypeVar("T", bound="WrapperRdkitmol")


class WrapperRdkitmol(object):
    """
    """

    @staticmethod
    def checkremove_hydrogens_from_rdkitmol(
        rdkitmol: Chem.rdchem.Mol,
    ) -> Chem.rdchem.Mol:
        """
        """
        rdkitmol_hydrogens_removed = Chem.rdmolops.RemoveAllHs(rdkitmol)

        nums_atomic = [
            #####
            x.GetAtomicNum()
            for x in rdkitmol_hydrogens_removed.GetAtoms()
        ]

        if 1 in nums_atomic:
            raise AssertionError(
                "\n\n"
                " :: hydrogens still exist after :: \n"
                " :: .checkremove_hydrogens_from_rdkitmol() :: \n"
                "\n\n"
            )
        assert 1 not in nums_atomic

        return rdkitmol_hydrogens_removed

    @classmethod
    def from_path_sdf(
        cls: Type[T],
        path_sdf: str,
        should_sanitize: bool = False,
        should_remove_initial_hydrogens: bool = False,
        should_add_new_hydrogens: bool = False,
    ) -> T:
        """
        """
        abspath_sdf = os.path.abspath(path_sdf)

        supplier_sdf = Chem.SDMolSupplier(
            #####
            abspath_sdf,
            sanitize=should_sanitize,
            removeHs=should_remove_initial_hydrogens,
        )
        assert len(supplier_sdf) == 1

        rdkitmol = supplier_sdf[0]
        if should_remove_initial_hydrogens:
            # rdkitmol = Chem.RemoveHs(rdkitmol)
            rdkitmol = WrapperRdkitmol.checkremove_hydrogens_from_rdkitmol(rdkitmol)

        return cls(
            #####
            rdkitmol=rdkitmol,
            should_add_new_hydrogens=should_add_new_hydrogens,
        )

    @classmethod
    def from_path_mol2(
        cls: Type[T],
        path_mol2: str,
        should_sanitize: bool = False,
        should_remove_initial_hydrogens: bool = False,
        should_add_new_hydrogens: bool = False,
    ) -> T:
        """
        """
        abspath_mol2 = os.path.abspath(path_mol2)

        rdkitmol = Chem.MolFromMol2File(
            #####
            abspath_mol2,
            sanitize=should_sanitize,
            removeHs=should_remove_initial_hydrogens,
        )

        if should_remove_initial_hydrogens:
            # rdkitmol = Chem.RemoveHs(rdkitmol)
            rdkitmol = WrapperRdkitmol.checkremove_hydrogens_from_rdkitmol(rdkitmol)

        return cls(
            #####
            rdkitmol=rdkitmol,
            should_add_new_hydrogens=should_add_new_hydrogens,
        )

    def __init__(
        #####
        self,
        rdkitmol: Chem.rdchem.Mol,
        should_add_new_hydrogens: bool,
    ) -> None:
        """
        """
        super().__init__()

        ####################################################
        ##### https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.AddHs
        ##### 1. "Much of the code assumes that Hs are not
        # included in the molecular topology, so be very careful with the molecule that comes back from this function."
        #####    .removeHs() before .addHs()
        ##### 2. "(bool)addCoords=False"
        #####    remember to set hydrogen coords
        ####################################################
        if should_add_new_hydrogens:
            # rdkitmol = Chem.rdmolops.RemoveHs(rdkitmol)
            rdkitmol = WrapperRdkitmol.checkremove_hydrogens_from_rdkitmol(rdkitmol)
            rdkitmol = Chem.rdmolops.AddHs(
                #####
                rdkitmol,
                addCoords=True,
            )

        self._RDKITMOL = rdkitmol

    @property
    def num_atoms(self, ) -> int:
        """
        """
        return self._RDKITMOL.GetNumAtoms()

    def get_positions(self, ) -> np.ndarray:
        """
        """
        return self._RDKITMOL.GetConformer().GetPositions()

    def copy_with_positions_new(
        #####
        self,
        positions_new: Sequence[Sequence[float]],
        should_add_new_hydrogens: bool,
        dict_properties: Dict[str, Any] = None,
    ) -> "WrapperRdkitmol":
        """
        """
        assert self.num_atoms == len(positions_new)

        rdkitmol_copy = Chem.rdchem.Mol(self._RDKITMOL)
        for i in range(self.num_atoms):
            assert len(positions_new[i]) == 3
            rdkitmol_copy.GetConformer().SetAtomPosition(i, positions_new[i])

        if dict_properties is not None:
            for i_key, i_value in dict_properties.items():
                rdkitmol_copy.SetProp(i_key, str(i_value))

        return type(self)(
            #####
            rdkitmol=rdkitmol_copy,
            should_add_new_hydrogens=should_add_new_hydrogens,
        )

    def save_sdf(self, path_sdf_to_save: str, ) -> None:
        """
        """
        abspath_sdf_to_save = os.path.abspath(path_sdf_to_save)
        Chem.SDWriter(abspath_sdf_to_save).write(self._RDKITMOL)

    save_sdf_given_self = save_sdf

    @classmethod
    def save_sdf_given_wrappers(
        #####
        cls,
        wrappers_rdkitmol: Sequence["WrapperRdkitmol"],
        path_sdf_to_save: str,
        append2sdf: bool = False,
    ) -> None:
        """
        """
        abspath_sdf_to_save = os.path.abspath(path_sdf_to_save)
        if append2sdf:
            io_writer = open(abspath_sdf_to_save, 'a')
        else:
            io_writer = abspath_sdf_to_save
        a_sdwriter = Chem.SDWriter(io_writer)
        for i_wrapper in wrappers_rdkitmol:
            a_sdwriter.write(i_wrapper._RDKITMOL)

    @classmethod
    def extract_mol0_only_4sdf(
        cls, path_sdf_mols: str, path_sdf_mol0_only: str, pose_rank: int
    ) -> None:
        """
        """
        abspath_sdf_mols = os.path.abspath(path_sdf_mols)
        abspath_sdf_mol0_only = os.path.abspath(path_sdf_mol0_only)

        supplier_sdf = Chem.SDMolSupplier(
            #####
            abspath_sdf_mols,
            sanitize=False,
            removeHs=False,
        )
        assert len(supplier_sdf) >= 1

        Chem.SDWriter(abspath_sdf_mol0_only).write(supplier_sdf[pose_rank])
