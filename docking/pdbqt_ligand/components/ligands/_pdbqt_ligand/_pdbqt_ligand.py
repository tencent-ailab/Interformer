# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import os
from typing import Dict, List, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd

# import pyvina

from pyvina_core import LigandBaseCpp as CorePdbqtLigand

from pdbqt_ligand.wrappers_for_io import IowrapperDirNewlyCreated

from pdbqt_ligand.wrappers_for_third_party_tools import (
    WrapperRdkitmol,
    WrapperObabel,
)

from pdbqt_ligand.components.atoms.pdbqt_atom import PdbqtAtom

T = TypeVar("T", bound="_PdbqtLigand")


class _PdbqtLigand(object):
    """
    """

    THRESHOLD_IDENTICAL_POSITION = 9e-4

    KEY_POSITIONS_HEAVY_ATOMS = "positions_heavy_atoms"
    KEY_POSITIONS_HYDROGENS = "positions_hydrogens"

    KEY_GRADIENTS_HEAVYATOMPOSITIONS_TO_POSE = "gradients_heavyatompositions_to_pose"

    EXTEND4SHAPE_AUTOBOX = 8

    @classmethod
    def from_path_sdf_ligand(
        cls: Type[T],
        path_sdf_ligand: str,
        #####
        path_cwd: str = None,
        pose_rank: int = 0,
    ) -> T:
        """
        """
        _, abspath_sdf_ligand = IowrapperDirNewlyCreated(path_cwd).copy_files_inward(
            path_sdf_ligand,
        )

        if pose_rank != -1:
            _abspath_sdf_mol0_only = abspath_sdf_ligand + "_mol0_only.sdf"

            WrapperRdkitmol.extract_mol0_only_4sdf(
                path_sdf_mols=abspath_sdf_ligand,
                path_sdf_mol0_only=_abspath_sdf_mol0_only,
                pose_rank=pose_rank
            )
            abspath_sdf_ligand = _abspath_sdf_mol0_only

        wrapper_rdkitmol = WrapperRdkitmol.from_path_sdf(
            path_sdf=abspath_sdf_ligand,
            should_sanitize=False,
            should_remove_initial_hydrogens=True,
            should_add_new_hydrogens=False,
        )
        abspath_pdbqt_ligand = abspath_sdf_ligand + "_obabel.pdbqt"
        WrapperObabel().convert_by_path(
            #####
            path_input=abspath_sdf_ligand,
            path_output=abspath_pdbqt_ligand,
        )

        return cls.from_path_pdbqt_ligand(
            #####
            path_pdbqt_ligand=abspath_pdbqt_ligand,
            wrapper_rdkitmol=wrapper_rdkitmol,
        )

    @classmethod
    def from_path_mol2_ligand(
        #####
        cls: Type[T],
        path_mol2_ligand: str,
        path_cwd: str = None,
    ) -> T:
        """
        """
        _, abspath_mol2_ligand = IowrapperDirNewlyCreated(path_cwd).copy_files_inward(
            path_mol2_ligand,
        )

        wrapper_rdkitmol = WrapperRdkitmol.from_path_mol2(
            path_mol2=abspath_mol2_ligand,
            should_sanitize=False,
            should_remove_initial_hydrogens=True,
            should_add_new_hydrogens=False,
        )
        abspath_pdbqt_ligand = abspath_mol2_ligand + "_obabel.pdbqt"
        WrapperObabel().convert_by_path(
            #####
            path_input=abspath_mol2_ligand,
            path_output=abspath_pdbqt_ligand,
        )

        return cls.from_path_pdbqt_ligand(
            #####
            path_pdbqt_ligand=abspath_pdbqt_ligand,
            wrapper_rdkitmol=wrapper_rdkitmol,
        )

    @classmethod
    def from_path_pdbqt_ligand(
        #####
        cls: Type[T],
        path_pdbqt_ligand: str,
        wrapper_rdkitmol: WrapperRdkitmol = None,
    ) -> T:
        """
        """
        abspath_pdbqt_ligand = os.path.abspath(path_pdbqt_ligand)

        core_pdbqt_ligand = CorePdbqtLigand()
        core_pdbqt_ligand.load_from_file_cppbind_(abspath_pdbqt_ligand)

        return cls(
            #####
            core_pdbqt_ligand=core_pdbqt_ligand,
            wrapper_rdkitmol=wrapper_rdkitmol,
        )

    def __init__(
        #####
        self,
        core_pdbqt_ligand: CorePdbqtLigand,
        wrapper_rdkitmol: WrapperRdkitmol,
    ) -> None:
        """
        """
        super().__init__()

        self.CORE_PDBQT_LIGAND = core_pdbqt_ligand

        self.ATOMS = [
            PdbqtAtom.from_line_pdbqt_atom(x)
            for x in self.CORE_PDBQT_LIGAND.lines_cppbind_
            if PdbqtAtom.is_line_pdbqt_atom(x)
        ]

        self.ATOMS_HEAVY = [
            #####
            x
            for x in self.ATOMS
            if (not x.is_hydrogen)
        ]
        self.ATOMS_HYDROGENS = [
            #####
            x
            for x in self.ATOMS
            if (x.is_hydrogen)
        ]

        self.POSITION_MEAN = np.mean(
            [
                #####
                x.POSITION
                for x in self.ATOMS_HEAVY
            ],
            axis=0,
        )
        self.SHAPE_AUTOBOX = self._calculate_shape_autobox()

        self.WRAPPER_RDKITMOL = wrapper_rdkitmol
        if self.has_wrapper_rdkitmol:
            if self.WRAPPER_RDKITMOL.num_atoms != self.num_atoms_heavy:
                raise AssertionError(
                    (
                        "\n\n"
                        " :: rdkitmol_num_atoms :: {} :: "
                        " ::    pdbqt_num_atoms :: {} :: "
                        "\n\n"
                    ).format(
                        self.WRAPPER_RDKITMOL.num_atoms, self.num_atoms_heavy,
                    )
                )
            assert self.WRAPPER_RDKITMOL.num_atoms == self.num_atoms_heavy
            (
                self.MAPPING_INDEX_HEAVY_ATOM_PDBQT_TO_ANY,
                self.MAPPING_INDEX_HEAVY_ATOM_ANY_TO_PDBQT,
            ) = self._get_mi_ha_by_pdbqt_and_any()

    @property
    def has_wrapper_rdkitmol(self, ) -> bool:
        """
        """
        return self.WRAPPER_RDKITMOL is not None

    @property
    def num_atoms(self, ) -> int:
        """
        """
        return len(self.ATOMS)

    @property
    def num_atoms_heavy(self, ) -> int:
        """
        """
        return len(self.ATOMS_HEAVY)

    @property
    def num_atoms_hydrogens(self, ) -> int:
        """
        """
        return len(self.ATOMS_HYDROGENS)

    @property
    def num_torsions(self, ) -> int:
        """
        """
        return self.CORE_PDBQT_LIGAND.num_torsions

    def _calculate_shape_autobox(self, ) -> np.ndarray:
        """
        """
        positions_atoms = np.array(
            [
                #####
                x.POSITION
                for x in self.ATOMS
            ]
        )

        shape_autobox = (
            #####
            positions_atoms.max(axis=0, )
            - positions_atoms.min(axis=0, )
        )
        shape_autobox += self.EXTEND4SHAPE_AUTOBOX

        max_distance = 0.0
        for i_position in positions_atoms:
            for j_position in positions_atoms:
                ij_distance = np.linalg.norm(i_position - j_position)

                if max_distance < ij_distance:
                    max_distance = ij_distance

        assert shape_autobox.shape == (3,)
        for i, _ in enumerate(shape_autobox):
            if shape_autobox[i] < max_distance:
                shape_autobox[i] = max_distance

        return shape_autobox

    def _get_mi_ha_by_pdbqt_and_any(
        self,
    ) -> Tuple[
        Sequence[int], Sequence[int],
    ]:
        """
        """
        assert self.has_wrapper_rdkitmol
        assert self.WRAPPER_RDKITMOL.num_atoms == self.num_atoms_heavy

        mapping_index_heavy_atom_pdbqt_to_any = [None] * len(self.ATOMS_HEAVY)
        mapping_index_heavy_atom_any_to_pdbqt = [None] * len(self.ATOMS_HEAVY)

        positions_any = self.WRAPPER_RDKITMOL.get_positions()
        assert positions_any.shape[0] == self.num_atoms_heavy
        for i, i_position_any in enumerate(positions_any):

            i_is_matched = False
            for j, j_atom in enumerate(self.ATOMS_HEAVY):
                if (
                    #####
                    np.linalg.norm(i_position_any - j_atom.POSITION)
                    < self.THRESHOLD_IDENTICAL_POSITION
                ):
                    mapping_index_heavy_atom_pdbqt_to_any[j] = i
                    mapping_index_heavy_atom_any_to_pdbqt[i] = j
                    i_is_matched = True
                    break

            if not (i_is_matched):
                j_index_nearest = np.argmin(
                    [
                        #####
                        np.linalg.norm(i_position_any - j_atom.POSITION)
                        for j_atom in self.ATOMS_HEAVY
                    ]
                )
                print(
                    (
                        "\n\n"
                        " :: unmatched atom :: index    :: [ {} ] :: \n"
                        "                   :: position :: [ {} ] :: \n"
                        "\n"
                        " ::   nearest atom :: index    :: [ {} ] :: \n"
                        "                   :: position :: [ {} ] :: \n"
                        "\n\n"
                    ).format(
                        #####
                        i,
                        i_position_any,
                        j_index_nearest,
                        self.ATOMS_HEAVY[j_index_nearest].POSITION,
                    )
                )
            assert i_is_matched

        assert not (None in mapping_index_heavy_atom_pdbqt_to_any)
        assert not (None in mapping_index_heavy_atom_any_to_pdbqt)

        return (
            mapping_index_heavy_atom_pdbqt_to_any,
            mapping_index_heavy_atom_any_to_pdbqt,
        )

    def reorder_axis0_lha_any_to_pdbqt(
        ######
        self,
        array_to_reorder: np.ndarray,
    ) -> np.ndarray:
        """
        """
        assert isinstance(array_to_reorder, np.ndarray)
        assert array_to_reorder.shape[0] == self.num_atoms_heavy
        return array_to_reorder[self.MAPPING_INDEX_HEAVY_ATOM_PDBQT_TO_ANY]

    def reorder_axis0_lha_pdbqt_to_any(
        ######
        self,
        array_to_reorder: np.ndarray,
    ) -> np.ndarray:
        """
        """
        assert isinstance(array_to_reorder, np.ndarray)
        assert array_to_reorder.shape[0] == self.num_atoms_heavy
        return array_to_reorder[self.MAPPING_INDEX_HEAVY_ATOM_ANY_TO_PDBQT]

    def get_positions(
        #####
        self,
        pose: Sequence[float],
    ) -> Dict[str, np.ndarray]:
        """
        """
        (
            list_positions_heavy_atoms,
            list_positions_hydrogens,
            list_gradients_heavyatompositions_to_pose,
        ) = self.CORE_PDBQT_LIGAND.FormulaDerivativeCore_ConformationToCoords(pose)

        return {
            self.KEY_POSITIONS_HEAVY_ATOMS: np.array(list_positions_heavy_atoms),
            self.KEY_POSITIONS_HYDROGENS: np.array(list_positions_hydrogens),
            self.KEY_GRADIENTS_HEAVYATOMPOSITIONS_TO_POSE: np.array(
                list_gradients_heavyatompositions_to_pose
            ),
        }

    def get_pose_initial(self, ) -> List[float]:
        """
        """
        return self.CORE_PDBQT_LIGAND.GetPoseInitial()

    def get_pose_random_in_boundingbox(
        self, corner_min: Sequence[float], corner_max: Sequence[float],
    ) -> List[float]:
        """
        """
        return self.CORE_PDBQT_LIGAND.GetPoseRandomInBoundingbox(
            corner_min, corner_max,
        )

    def get_pose_random_next(self, pose_current: Sequence[float], ) -> List[float]:
        """
        """
        return self.CORE_PDBQT_LIGAND.GetPoseRandomNext(pose_current)

    def pick_first_k_unique_poses(
        self,
        poses: Sequence[Sequence[float]],
        k: int = 20,
        threshold_rmsd_heavy_atoms: float = 2.0,
        should_reduce_threshold_to_pick_k: bool = True,
    ) -> Tuple[List[List[float]], List[int]]:
        """
        """
        if not should_reduce_threshold_to_pick_k:
            return self.CORE_PDBQT_LIGAND.PickFirstKUniquePoses(
                #####
                poses,
                k,
                threshold_rmsd_heavy_atoms,
            )

        i_threshold = threshold_rmsd_heavy_atoms
        for _ in range(10):
            tuple_the_picked = self.CORE_PDBQT_LIGAND.PickFirstKUniquePoses(
                #####
                poses,
                k,
                i_threshold,
            )
            if len(tuple_the_picked[0]) >= k:
                break
            i_threshold /= 2.0

        assert len(tuple_the_picked[0]) >= k

        return tuple_the_picked

    def save_sdf(
        #####
        self,
        pose_to_save: Sequence[float],
        path_sdf_to_save: str,
        should_add_new_hydrogens: bool = True,
        dict_properties_for_1_pose: Union[Dict, pd.DataFrame,] = None,
    ) -> None:
        """
        """
        # (
        #     positions_atoms_heavy_pdbqt,
        #     _,
        #     _,
        # ) = self.CORE_PDBQT_LIGAND.FormulaDerivativeCore_ConformationToCoords(
        #     pose_to_save
        # )
        # positions_atoms_heavy_pdbqt = np.array(positions_atoms_heavy_pdbqt)

        positions_heavy_atoms_pdbqt = (
            #####
            self.get_positions(pose_to_save)
        )[self.KEY_POSITIONS_HEAVY_ATOMS]

        self.WRAPPER_RDKITMOL.copy_with_positions_new(
            positions_new=positions_heavy_atoms_pdbqt[
                self.MAPPING_INDEX_HEAVY_ATOM_ANY_TO_PDBQT
            ],
            should_add_new_hydrogens=should_add_new_hydrogens,
        ).save_sdf(path_sdf_to_save)

    save_sdf_given_1_pose = save_sdf

    def save_sdf_given_poses(
        self,
        poses_to_save: Sequence[Sequence[float]],
        path_sdf_to_save: str,
        should_add_new_hydrogens: bool = True,
        dict_properties_for_poses: Union[Dict, pd.DataFrame,] = None,
        append2sdf: bool = False,
    ) -> None:
        """
        """
        wrappers_rdkitmol_from_poses = []

        for i, i_pose in enumerate(poses_to_save):
            i_positions = (
                #####
                self.get_positions(i_pose)
            )[self.KEY_POSITIONS_HEAVY_ATOMS]

            i_dict_properties = None
            if dict_properties_for_poses is not None:
                i_dict_properties = {
                    x: dict_properties_for_poses[x][i]
                    for x in dict_properties_for_poses.keys()
                }

            wrappers_rdkitmol_from_poses.append(
                self.WRAPPER_RDKITMOL.copy_with_positions_new(
                    positions_new=i_positions[
                        self.MAPPING_INDEX_HEAVY_ATOM_ANY_TO_PDBQT
                    ],
                    should_add_new_hydrogens=should_add_new_hydrogens,
                    dict_properties=i_dict_properties,
                )
            )

        WrapperRdkitmol.save_sdf_given_wrappers(
            wrappers_rdkitmol=wrappers_rdkitmol_from_poses,
            path_sdf_to_save=path_sdf_to_save,
            append2sdf=append2sdf,
        )
