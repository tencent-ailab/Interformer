# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

from typing import Sequence

import numpy as np

import pyvina_core

from pdbqt_ligand.components.ligands._pdbqt_ligand import _PdbqtLigand
from pdbqt_ligand.components.evaluators.evaluator_base import EvaluatorBase


class EvaluatorNormalscore(EvaluatorBase):
    """
    """

    def get_weights(self, ) -> dict:
        """
        """
        return {
            "weight_intra": self._CORE_EVALUATOR_NORMALSCORE.get_weight_intra(),
            "weight_collision_inter": self._CORE_EVALUATOR_NORMALSCORE.get_weight_collision_inter(),
        }

    def __init__(
        #####
        self,
        a_pdbqt_ligand: _PdbqtLigand,
        pi_inter: np.ndarray,
        mean_inter: np.ndarray,
        sigma_inter: np.ndarray,
        vdwradius_sum_inter: np.ndarray,
        positions_inter: np.ndarray,
        corner_min: Sequence[float],
        corner_max: Sequence[float],
        reciprocal_resolution: int,
        #####
        options_bitwise_for_precalculating: int = (1 + 2 + 4),
        weight_intra: float = None,
        weight_collision_inter: float = None,
    ) -> None:
        """
        """
        self._CORE_EVALUATOR_NORMALSCORE = pyvina_core.evaluators.CoreEvaluatorNormalscore(
            a_pdbqt_ligand.CORE_PDBQT_LIGAND,
            pi_inter,
            mean_inter,
            sigma_inter,
            vdwradius_sum_inter,
            positions_inter,
            corner_min,
            corner_max,
            reciprocal_resolution,
        )

        if weight_intra is None:
            weight_intra = 0.0
        if weight_collision_inter is None:
            weight_collision_inter = 1.0

        print(
            (
                #####
                "\n\n"
                " :: SET WEIGHTS :: \n"
                " :: FROM :: \n"
                "{}\n"
                "\n\n"
            ).format(
                self.get_weights(),
            )
        )
        self._CORE_EVALUATOR_NORMALSCORE.set_weight_intra(weight_intra)
        self._CORE_EVALUATOR_NORMALSCORE.set_weight_collision_inter(weight_collision_inter)
        print(
            (
                #####
                "\n\n"
                " ::  TO  :: \n"
                "{}\n"
                "\n\n"
            ).format(
                self.get_weights(),
            )
        )

        self._CORE_EVALUATOR_NORMALSCORE.precalculate_core_grid_4d_(
            options_bitwise_for_precalculating
        )

        super().__init__(
            #####
            core_evaluator_base=self._CORE_EVALUATOR_NORMALSCORE,
        )
