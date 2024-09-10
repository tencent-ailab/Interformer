# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

# from typing import Dict, List, Sequence


import pyvina_core

from pdbqt_ligand.components.evaluators.evaluator_base import EvaluatorBase
from pdbqt_ligand.components.minimizers.minimizer_base import MinimizerBase


class MinimizerBfgs(MinimizerBase):
    """
    """

    # @staticmethod
    # def minimize_given_evaluator(
    #     #####
    #     pose: Sequence[float],
    #     a_evaluator: EvaluatorBase,
    # ) -> Dict:
    #     """
    #     """
    #     tuple_the_minimized = pyvina_core.minimizers.CoreMinimizerBfgs.MinimizeGivenEvaluator(
    #         #####
    #         pose,
    #         a_evaluator.CORE_EVALUATOR_BASE,
    #     )
    #     return {
    #         "pose": tuple_the_minimized[0],
    #         "energy": tuple_the_minimized[1],
    #     }

    def __init__(
        #####
        self,
        a_evaluator: EvaluatorBase,
    ) -> None:
        """
        """
        assert a_evaluator.has_core_evaluator_base

        self._CORE_MINIMIZER_BFGS = pyvina_core.minimizers.CoreMinimizerBfgs(
            a_evaluator.CORE_EVALUATOR_BASE
        )

        super().__init__(
            #####
            core_minimizer_base=self._CORE_MINIMIZER_BFGS
        )
