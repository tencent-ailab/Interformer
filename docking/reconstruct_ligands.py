# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import argparse
import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path
import pickle
import pprint
import subprocess
import time
import traceback
from typing import List, NamedTuple
import shelve

import numpy as np
import pandas as pd

import pdbqt_ligand
from pdbqt_ligand.components.atoms.pdb_atom import PdbAtom

###############################################
##### constants
###############################################
SHAPE_POCKET_DEFAULT = np.array(
    [
        #####
        20.0,
        20.0,
        20.0,
    ]
)
REPEATS_OF_MONTE_CARLO = 64
STEPS_FOR_EACH_MONTE_CARLO = 2000

A_ENTRY_SUBPARSER = "find"
B_ENTRY_SUBPARSER = "read_pkl"
C_ENTRY_SUBPARSER = "stat"
###############################################


LOGGER = logging.getLogger(__name__)


class PathtupleForTask(NamedTuple):
    """guessed directory structure:
    - cwd/
        - uff/
            - 1bcu_uff.sdf
        - ligand/
            - 1bcu_docked.sdf
        - complex/
            - 1bcu_complex.pdb
        - gaussian_predict/
            - 1bcu_G.db
        - ligand_reconstructing/
            - 1bcu_docked.sdf
            - 1bcu_docked.sdf_stat.csv
            - 1bcu_docked.sdf_remote/
                - 1bcu_task.sh
                - 1bcu_task.log
    """

    ABSPATH_CWD: str
    STR_PDB_ID: str
    DICT_EXTRA: dict

    @property
    def ABSPATH_SDF_LIGAND(self) -> str:
        return self.DICT_EXTRA.get(
            "abspath_sdf_ligand",
            os.path.join(
                self.ABSPATH_CWD,
                "ligand",
                "{}_docked.sdf".format(self.STR_PDB_ID),
            ),
        )

    @property
    def ABSPATH_SDF_REF(
        self,
    ) -> str:
        return os.path.join(
            #####
            self.ABSPATH_CWD,
            "ligand",
            "{}_docked.sdf".format(self.STR_PDB_ID),
        )

    @property
    def ABSPATH_DIR_OUTPUT(self) -> str:
        return self.DICT_EXTRA.get(
            "abspath_dir_output",
            os.path.join(
                self.ABSPATH_CWD,
                "ligand_reconstructing",
            ),
        )

    @property
    def ABSPATH_SDF_OUTPUT(
        self,
    ) -> str:
        return os.path.join(
            #####
            self.ABSPATH_DIR_OUTPUT,
            "{}_docked.sdf".format(self.STR_PDB_ID),
        )

    @property
    def ABSPATH_PDB_COMPLEX(
        self,
    ) -> str:
        return os.path.join(
            #####
            self.ABSPATH_CWD,
            "complex",
            "{}_complex.pdb".format(self.STR_PDB_ID),
        )

    @property
    def ABSPATH_PKL_NORMALSCORE(
        self,
    ) -> str:
        return os.path.join(
            #####
            self.ABSPATH_CWD,
            "gaussian_predict",
            "{}_G.db".format(self.STR_PDB_ID),
        )

    @property
    def ABSPATH_CSV_STAT(
        self,
    ) -> str:
        """ """
        return self.ABSPATH_SDF_OUTPUT + "_stat.csv"

    @property
    def ABSPATH_DIR_REMOTE(
        self,
    ) -> str:
        """ """
        return self.ABSPATH_SDF_OUTPUT + "_remote"

    @property
    def ABSPATH_SH_REMOTE(
        self,
    ) -> str:
        """ """
        return os.path.join(
            #####
            # self.ABSPATH_SDF_OUTPUT + "_remote",
            self.ABSPATH_DIR_REMOTE,
            "{}_task.sh".format(self.STR_PDB_ID),
        )

    @property
    def ABSPATH_PKL_REMOTE(self) -> str:
        return os.path.join(self.ABSPATH_DIR_REMOTE, "remote.pkl")

    @property
    def ABSPATH_HINT_ERROR(
        self,
    ) -> str:
        """ """
        return self.ABSPATH_SDF_OUTPUT + "_error.hint"

    @classmethod
    def is_valid_str_pdb_id(
        cls,
        str_pdb_id: str,
    ) -> bool:
        """ """
        if str_pdb_id.strip() != str_pdb_id:
            return False
        if len(str_pdb_id) != 4:
            return False
        if not str_pdb_id.isalnum():
            return False
        return True

    @classmethod
    def guess_str_pdb_id_by_paths(cls, *args) -> str:
        """ """
        filenames = [os.path.basename(x) for x in args]

        str_pdb_id = (
            #####
            os.path.commonprefix(filenames)
        ).split("_")[0]

        LOGGER.info(
            (
                "\n\n"
                " :: GUESSED pdb id => [ {} ] :: \n"
                " :: based on :: \n"
                "{}\n"
                "\n\n"
            ).format(
                #####
                str_pdb_id,
                pprint.pformat(args),
            )
        )

        assert cls.is_valid_str_pdb_id(str_pdb_id)

        return str_pdb_id

    ########################################
    ##### factory funcs
    ########################################
    @classmethod
    def from_str_pdb_id(
        cls,
        str_pdb_id: str,
        path_cwd: str,
        args: argparse.Namespace,
    ) -> "PathtupleForTask":
        """ """
        assert cls.is_valid_str_pdb_id(str_pdb_id)
        abspath_cwd = os.path.abspath(path_cwd)

        dict_extra = {}
        if args.uff_folder is not None:
            dict_extra["abspath_sdf_ligand"] = os.path.join(
                abspath_cwd,
                os.path.basename(
                    #####
                    os.path.normpath(args.uff_folder)
                ),
                "{}_uff.sdf".format(str_pdb_id),
            )
        if args.output_folder is not None:
            dict_extra["abspath_dir_output"] = os.path.join(
                abspath_cwd,
                os.path.basename(
                    #####
                    os.path.normpath(args.output_folder)
                ),
            )

        return cls(
            #####
            ABSPATH_CWD=abspath_cwd,
            STR_PDB_ID=str_pdb_id,
            DICT_EXTRA=dict_extra,
        )

    @classmethod
    def multifrom_path_cwd(
        cls,
        path_cwd: str,
        args: argparse.Namespace,
    ) -> List["PathtupleForTask"]:
        """ """
        abspath_cwd = os.path.abspath(path_cwd)

        filenames = list([x[:-4] for x in glob.glob(f'{abspath_cwd}/gaussian_predict/*.db.dat')])
        # print(f"Debug Filenames:{filenames}")
        list_strs_pdb_ids = [
            #####
            cls.guess_str_pdb_id_by_paths(x)
            for x in filenames
        ]
        list_strs_pdb_ids.sort()

        return [
            cls.from_str_pdb_id(
                #####
                str_pdb_id=x,
                path_cwd=abspath_cwd,
                args=args,
            )
            for x in list_strs_pdb_ids
        ]


def get_args_and_mainparser():
    """ """
    mainparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ########################################
    ##### 1. args for all subparsers
    ########################################
    mainparser.add_argument(
        "-r",
        "--remote",
        default=False,
        action="store_true",
        help="launch tasks on remote",
    )
    mainparser.add_argument(
        "-s",
        "--skip_finished",
        default=False,
        action="store_true",
        help="skip tasks with _stat.csv",
    )
    mainparser.add_argument(
        "-d",
        "--delete",
        default=False,
        action="store_true",
        help="Delete existing docked folder.",
    )
    mainparser.add_argument(
        "-y",
        "--yes",
        default=False,
        action="store_true",
        help="skip the confirmation",
    )
    mainparser.add_argument(
        "--uff_folder",
        default=None,
        help="input ligands from uff folder",
    )
    mainparser.add_argument(
        "--output_folder",
        default=None,
        help="output ligands to output folder",
    )
    mainparser.add_argument(
        "--weight_intra",
        type=float,
        default=30.0,
        help=" It is the weight of ligand intra steric clash",
    )
    mainparser.add_argument(
        "--weight_collision_inter",
        type=float,
        default=5.0,
        help="It is the weight of ligand and protein inter steric clash",
    )
    mainparser.add_argument(
        "--diff_weight0",
        default=False,
        action="store_true",
        help="[Time Consuming] Turn on the Inter steric clash penalty term for statistic.",
    )
    mainparser.add_argument(
        "--distance_inter_min_expected",
        type=float,
        default=None,
        help="refinement",
    )
    mainparser.add_argument(
        "--vdwdistance_inter_min_expected",
        type=float,
        default=-1.,
        help="refinement",
    )
    mainparser.add_argument(
        "--num_output_poses",
        type=int,
        default=20,
        help="num of output poses",
    )
    # calculate
    mainparser.add_argument(
        "--bust",
        default=False,
        action="store_true",
        help="turn on bust program.",
    )
    mainparser.add_argument(
        "-c",
        "--cwd",
        type=str,
        required=True,
    )
    ########## 2.1. either pdb_id or find_all
    main_a_group = mainparser.add_mutually_exclusive_group(
        required=True,
    )
    main_a_group.add_argument(
        "-i",
        "--pdb_id",
        type=str,
    )
    main_a_group.add_argument(
        "-a",
        "--find_all",
        default=False,
        action="store_true",
        help="try to find valid pdb_id based on --cwd, and reconstruct them all",
    )
    main_a_group.add_argument(
        "-b",
        "--block_pdb_ids",
        type=str,
        default=None,
    )

    subparsers = mainparser.add_subparsers(
        dest="subparser",
        required=True,
    )
    ########################################
    ##### 2. subparser A
    #####
    ##### launch tasks by pdb_id and guessing
    ########################################
    a_subparser = subparsers.add_parser(A_ENTRY_SUBPARSER)

    ########################################
    ##### 3. subparser A
    #####
    ##### launch tasks by pdb_id and guessing
    ########################################
    b_subparser = subparsers.add_parser(B_ENTRY_SUBPARSER)
    b_subparser.add_argument(
        "--pkl_remote",
        default=None,
    )

    ########################################
    ##### 4. subparser C
    #####
    ##### stat related
    ########################################
    c_subparser = subparsers.add_parser(C_ENTRY_SUBPARSER)

    args = mainparser.parse_args()

    return args, mainparser


def launch_1_task_remote(
    #####
    pathtuple: PathtupleForTask,
    args: argparse.Namespace,
) -> None:
    """ """
    raise NotImplementedError()


def launch_1_task_local(
    #####
    pathtuple: PathtupleForTask,
    args: argparse.Namespace,
) -> None:
    """ """
    reconstruct_1_ligand_given_paths(
        path_sdf_ligand=pathtuple.ABSPATH_SDF_LIGAND,
        path_sdf_ref=pathtuple.ABSPATH_SDF_REF,
        path_pdb_complex=pathtuple.ABSPATH_PDB_COMPLEX,
        path_pkl_normalscore=pathtuple.ABSPATH_PKL_NORMALSCORE,
        path_sdf_output=pathtuple.ABSPATH_SDF_OUTPUT,
        path_csv_stat=pathtuple.ABSPATH_CSV_STAT,
        str_pdb_id=pathtuple.STR_PDB_ID,
        weight_intra=args.weight_intra,
        weight_collision_inter=args.weight_collision_inter,
        #####
        diff_weight0=args.diff_weight0,
        distance_inter_min_expected=args.distance_inter_min_expected,
        vdwdistance_inter_min_expected=args.vdwdistance_inter_min_expected,
        path_pdb_pocket_4bust=os.path.join(
            pathtuple.ABSPATH_CWD,
            "pocket",
            "{}_pocket.pdb".format(pathtuple.STR_PDB_ID),
        ) if args.bust else None,
        num_output_poses=args.num_output_poses,
    )


def estimate_energy_collision(
    poses,
    a_evaluator,
    evaluator_weight0,
    weight_intra,
    weight_collision_inter,
) -> pd.DataFrame:
    """ """
    _dicts_weight0 = [
        evaluator_weight0._CORE_EVALUATOR_NORMALSCORE.evaluate_nogrid_given_weights(
            x,
            weight_intra,
            weight_collision_inter,
        )[3]
        for x in poses
    ]
    _dict_4csv = {}
    for i_key in _dicts_weight0[0].keys():
        _dict_4csv[i_key] = [x[i_key] if i_key in x else -1. for x in _dicts_weight0]

    return pd.DataFrame(_dict_4csv)


def cancel_collision_dynamic(
    pose,
    a_evaluator: pdbqt_ligand.EvaluatorNormalscore,
    a_minimizer,
    initweight_intra: float,
    initweight_collision_inter: float,
    distance_inter_min_expected: float,
    vdwdistance_inter_min_expected: float,
) -> List[float]:
    """ """
    pose_tmp = [x for x in pose]

    weight_intra = initweight_intra
    weight_collision_inter = initweight_collision_inter

    addweight_collision_inter = 1.0

    for i in range(10):
        i_result = (
            a_evaluator._CORE_EVALUATOR_NORMALSCORE.evaluate_nogrid_given_weights(
                pose_tmp,
                weight_intra,
                weight_collision_inter,
            )
        )
        # print(
        #     (
        #         #####
        #         "\n\n"
        #         "##########\n"
        #         "{}\n"
        #         "{}\n"
        #         "##########\n"
        #         "EXPECTED\n"
        #         "   DISTANCE INTER MIN [ {} ]\n"
        #         "VDWDISTANCE INTER MIN [ {} ]\n"
        #         "##########\n"
        #         "\n\n"
        #     ).format(
        #         i,
        #         i_result,
        #         distance_inter_min_expected,
        #         vdwdistance_inter_min_expected,
        #     )
        # )

        is_pose_ok = True
        for j_2check, j_lowerbound in [
            (
                i_result[3]["distance_inter_min"],
                distance_inter_min_expected,
            ),
            (
                i_result[3]["vdwdistance_inter_min"],
                vdwdistance_inter_min_expected,
            ),
        ]:
            if j_lowerbound is not None:
                if j_2check < j_lowerbound:
                    is_pose_ok = False
        if is_pose_ok:
            break

        weight_collision_inter = initweight_collision_inter + addweight_collision_inter
        addweight_collision_inter *= 2

        a_evaluator._CORE_EVALUATOR_NORMALSCORE.SetOption(
            {
                "weight_intra": str(weight_intra),
                "weight_collision_inter": str(weight_collision_inter),
            }
        )
        pose_tmp = (a_minimizer.minimize(pose_tmp))["pose"]
        a_evaluator._CORE_EVALUATOR_NORMALSCORE.ResetOption()

    a_evaluator._CORE_EVALUATOR_NORMALSCORE.SetOption(
        {
            "weight_intra": str(initweight_intra),
            "weight_collision_inter": str(initweight_collision_inter),
        }
    )
    pose_tmp = (a_minimizer.minimize(pose_tmp))["pose"]
    a_evaluator._CORE_EVALUATOR_NORMALSCORE.ResetOption()

    return pose_tmp


def reconstruct_1_ligand_given_paths(
    path_sdf_ligand: str,
    path_sdf_ref: str,
    path_pdb_complex: str,
    path_pkl_normalscore: str,
    path_sdf_output: str,
    path_csv_stat: str = None,
    str_pdb_id: str = None,
    shape_pocket: np.ndarray = None,
    weight_intra: float = None,
    weight_collision_inter: float = None,
    #####
    diff_weight0: bool = False,
    distance_inter_min_expected: float = None,
    vdwdistance_inter_min_expected: float = None,
    path_pdb_pocket_4bust: str = None,
    num_output_poses: int = None,
):
    """ """
    abspath_sdf_ligand = os.path.abspath(path_sdf_ligand)
    abspath_sdf_ref = os.path.abspath(path_sdf_ref)
    abspath_pdb_complex = os.path.abspath(path_pdb_complex)
    abspath_pkl_normalscore = os.path.abspath(path_pkl_normalscore)
    abspath_sdf_output = os.path.abspath(path_sdf_output)

    abspath_csv_stat = (
        abspath_sdf_output + "_stat.csv"
        if path_csv_stat is None
        else os.path.abspath(path_csv_stat)
    )

    ########################################
    ##### 2. input
    ########################################
    positions_complex = None
    with open(abspath_pdb_complex) as file_to_read:
        positions_complex = [
            PdbAtom.from_line_pdb_atom(x).POSITION
            for x in file_to_read
            if PdbAtom.is_line_pdb_atom(x)
        ]

    dict_normalscore = None
    # REVO: We switch to using shelve data structure,
    # index starting from 0., the index of .db file should equal to the rank of sdf file.
    with shelve.open(abspath_pkl_normalscore) as db:
        indices = [int(x) for x in list(db.keys())]
        for index in indices:
            LOGGER.info(":" * 30)
            #
            LOGGER.info(f" :: SDF Rank:{index} <- {abspath_pkl_normalscore}")
            dict_normalscore = db[str(index)]
            a_pdbqt_ligand = pdbqt_ligand.PdbqtLigand.from_path_sdf_ligand(
                abspath_sdf_ligand,
                pose_rank=index,
            )

            b_pdbqt_ref = pdbqt_ligand.PdbqtLigand.from_path_sdf_ligand(abspath_sdf_ref)

            len_ligand = a_pdbqt_ligand.num_atoms_heavy
            LOGGER.info(
                (
                    #####
                    "\n\n"
                    " :: MIN [ vdw_pair ] from .pkl :: [ {} ] :: \n"
                    " :: MAX [ vdw_pair ] from .pkl :: [ {} ] :: \n"
                    "\n\n"
                ).format(
                    np.min(dict_normalscore["vdw_pair"][:len_ligand, len_ligand:]),
                    np.max(dict_normalscore["vdw_pair"][:len_ligand, len_ligand:]),
                )
            )

            ########################################
            ##### 3. pocket
            ########################################
            if shape_pocket is None:
                shape_pocket = SHAPE_POCKET_DEFAULT
            shape_pocket = np.amax(
                [
                    #####
                    shape_pocket,
                    b_pdbqt_ref.SHAPE_AUTOBOX,
                ],
                axis=0,
            )
            LOGGER.info(
                (
                    #####
                    "\n\n"
                    " :: final shape_pocket :: \n"
                    " :: (might be extended based on ligand) :: \n"
                    "{}\n"
                    "\n\n"
                ).format(shape_pocket)
            )

            ########################################
            ##### 4. evaluator
            ########################################
            timepoint_start = time.time()

            args_4evaluator = [
                a_pdbqt_ligand,
                a_pdbqt_ligand.reorder_axis0_lha_any_to_pdbqt(
                    dict_normalscore["pi"][:len_ligand, len_ligand:]
                ),
                a_pdbqt_ligand.reorder_axis0_lha_any_to_pdbqt(
                    dict_normalscore["mean"][:len_ligand, len_ligand:]
                ),
                a_pdbqt_ligand.reorder_axis0_lha_any_to_pdbqt(
                    dict_normalscore["sigma"][:len_ligand, len_ligand:]
                ),
                a_pdbqt_ligand.reorder_axis0_lha_any_to_pdbqt(
                    dict_normalscore["vdw_pair"][:len_ligand, len_ligand:]
                ),
                positions_complex,
                b_pdbqt_ref.POSITION_MEAN - (shape_pocket / 2.0),
                b_pdbqt_ref.POSITION_MEAN + (shape_pocket / 2.0),
                8,
            ]
            kwargs_4evaluator = dict(
                weight_intra=weight_intra,
                weight_collision_inter=weight_collision_inter,
            )
            ####
            # [Revo] Interformer Input Check
            pocket_len = dict_normalscore['pocket_len'][0]
            print(" :: vdw:", dict_normalscore['vdw_pair'].shape)
            print(' :: ligand_len:', len_ligand)
            print(' :: pocket_len:', pocket_len)
            print(' :: reference:pos:mean:', b_pdbqt_ref.POSITION_MEAN)
            assert pocket_len + len_ligand == dict_normalscore['vdw_pair'].shape[
                0]  # pocket atom + ligand atom should be the same num of complex atoms
            #

            a_evaluator_normalscore = pdbqt_ligand.EvaluatorNormalscore(
                *args_4evaluator,
                **kwargs_4evaluator,
            )

            time_elapsed = time.time() - timepoint_start
            crystal_normalscore = a_evaluator_normalscore.evaluate(
                a_pdbqt_ligand.get_pose_initial()
            )

            LOGGER.info(
                (
                    "\n\n"
                    " :: time ELAPSED for evaluator precalculating  :: [ {} ] :: \n"
                    " :: num of ligand heavy atoms                  :: [ {} ] :: \n"
                    " :: the evaluated for crystal pose             :: \n"
                    "{}\n"
                    "\n\n"
                ).format(
                    #####
                    time_elapsed,
                    a_pdbqt_ligand.num_atoms_heavy,
                    pprint.pformat(crystal_normalscore),
                )
            )

            ########################################
            ##### 5. sampler
            ########################################
            a_minimizer_bfgs = pdbqt_ligand.MinimizerBfgs(
                #####
                a_evaluator=a_evaluator_normalscore,
            )
            a_sampler_monte_carlo = pdbqt_ligand.SamplerMonteCarlo(
                #####
                a_minimizer=a_minimizer_bfgs,
            )

            dict_the_sampled = a_sampler_monte_carlo.sample(
                b_pdbqt_ref.POSITION_MEAN - (shape_pocket / 2.0),
                b_pdbqt_ref.POSITION_MEAN + (shape_pocket / 2.0),
                REPEATS_OF_MONTE_CARLO,
                STEPS_FOR_EACH_MONTE_CARLO,
            )
            _, indices_unique = a_pdbqt_ligand.pick_first_k_unique_poses(
                dict_the_sampled["poses"],
                k=num_output_poses
            )

            ########################################
            ##### 6. output
            ########################################
            ########## 6.1. paths
            for i_abspath_output in [
                #####
                abspath_sdf_output,
                abspath_csv_stat,
            ]:
                Path(os.path.dirname(i_abspath_output)).mkdir(
                    parents=True,
                    exist_ok=True,
                )
            ########## 6.2. sdf
            poses_output = [dict_the_sampled["poses"][x] for x in indices_unique]
            energies_output = [dict_the_sampled["energies"][x] for x in indices_unique]

            pose_initial = a_pdbqt_ligand.get_pose_initial()
            energy_initial = a_evaluator_normalscore.evaluate(pose_initial)["energy"]
            poses_output.append(pose_initial)
            energies_output.append(energy_initial)

            evaluator_weight0 = None
            csv_collision_pre_refinement = None
            csv_collision_post_refinement = None
            if diff_weight0:
                evaluator_weight0 = pdbqt_ligand.EvaluatorNormalscore(
                    *args_4evaluator,
                    weight_intra=weight_intra,
                    weight_collision_inter=0.0,
                )

                csv_collision_pre_refinement = estimate_energy_collision(
                    poses_output,
                    a_evaluator_normalscore,
                    evaluator_weight0,
                    weight_intra,
                    weight_collision_inter,
                ).add_suffix("_0_pre_refinement")

            if (
                #####
                (distance_inter_min_expected is not None)
                or (vdwdistance_inter_min_expected is not None)
            ):
                print(":" * 10)
                print(
                    f" :: Pose-Minimization Start... "
                    f"vdwDistance_inter_min_expected:{vdwdistance_inter_min_expected} ::")
                _evaluator_collision = evaluator_weight0
                if _evaluator_collision is None:
                    _evaluator_collision = pdbqt_ligand.EvaluatorNormalscore(
                        *args_4evaluator,
                        weight_intra=weight_intra,
                        weight_collision_inter=0.0,
                    )
                _minimizer_collision = pdbqt_ligand.MinimizerBfgs(
                    a_evaluator=_evaluator_collision,
                )

                ##### skip last initial pose
                poses_output = [
                                   #####
                                   # (_minimizer_collision.minimize(x))["pose"]
                                   cancel_collision_dynamic(
                                       x,
                                       _evaluator_collision,
                                       _minimizer_collision,
                                       weight_intra,
                                       weight_collision_inter,
                                       distance_inter_min_expected,
                                       vdwdistance_inter_min_expected,
                                   )
                                   for x in poses_output[:-1]
                               ] + poses_output[-1:]

                energies_output = [
                    a_evaluator_normalscore.evaluate(x)["energy"] for x in poses_output
                ]
                print("Done")
                print(":" * 10)

            if diff_weight0:
                csv_collision_post_refinement = estimate_energy_collision(
                    poses_output,
                    a_evaluator_normalscore,
                    evaluator_weight0,
                    weight_intra,
                    weight_collision_inter,
                ).add_suffix("_1_post_refinement")

            dicts_details = [
                a_evaluator_normalscore.CORE_EVALUATOR_BASE.evaluate_no_grid(x, True)[3]
                for x in poses_output
            ]
            energies_no_grid_output = [x["loss_total"] for x in dicts_details]
            energies_inter_output = [x["loss_inter"] for x in dicts_details]
            energies_intra_output = [x["loss_intra"] for x in dicts_details]

            # save the docking pose first, in order to evaluate RMSD
            tmp_sdf_output = f'{tempfile.gettempdir()}/{os.path.basename(abspath_sdf_output)}'
            a_pdbqt_ligand.save_sdf_given_poses(
                poses_output,
                tmp_sdf_output,
            )

            ########################################
            ##### 7. RMSD
            ########################################
            str_output_obrms = subprocess.check_output(
                [
                    "obrms",
                    tmp_sdf_output,
                    abspath_sdf_ref,
                ]
            ).decode()
            list_rmsd = []
            for i_line in str_output_obrms.splitlines():
                i_line_split = i_line.split()
                if len(i_line_split) > 0:
                    list_rmsd.append(float(i_line_split[-1]))

            dict_stat = {
                "filename_input": os.path.basename(abspath_sdf_ligand),
                "pdb_id": str_pdb_id,
                "pose_rank": list(
                    #####
                    range(len(list_rmsd))
                ),
                "num_atoms_heavy": a_pdbqt_ligand.num_atoms_heavy,
                "num_torsions": a_pdbqt_ligand.CORE_PDBQT_LIGAND.num_torsions,
                "pose": poses_output,
                "energy": energies_output,
                # "energy_grid": energies_output,
                # "energy_nogrid": energies_no_grid_output,
                "inter_energy": energies_inter_output,
                "intra_energy": energies_intra_output,
                "rmsd": list_rmsd,
            }

            dataframe_stat = pd.DataFrame(dict_stat)

            if diff_weight0:
                csv_collision = pd.concat(
                    [
                        csv_collision_pre_refinement,
                        csv_collision_post_refinement,
                    ],
                    axis=1,
                )
                csv_collision = csv_collision.reindex(
                    sorted(csv_collision.columns),
                    axis=1,
                )

                dataframe_stat = pd.concat(
                    [
                        dataframe_stat,
                        csv_collision,
                    ],
                    axis=1,
                )

            ####
            # Output Dataframe
            # mv rmsd column to the end
            rmsd = dataframe_stat.pop('rmsd')
            dataframe_stat['rmsd'] = rmsd
            # assign sdf_rank
            dataframe_stat['sdf_rank'] = index
            if os.path.exists(abspath_csv_stat):
                dataframe_stat.to_csv(abspath_csv_stat, mode='a', header=False, index=False)
            else:
                dataframe_stat.to_csv(abspath_csv_stat, index=False)
            # Output SDF
            a_pdbqt_ligand.save_sdf_given_poses(
                poses_output,
                abspath_sdf_output,
                dict_properties_for_poses=dataframe_stat,
                append2sdf=True,
            )
            LOGGER.info(
                (
                    #####
                    "\n\n"
                    " :: :: _stat.csv previewed :: :: \n"
                    "{}\n"
                    "\n\n"
                ).format(dataframe_stat)
            )
            ########################################
            ##### 8. bust
            ########################################
            if path_pdb_pocket_4bust is not None:
                print(' :: Runing Bust Program...')
                subprocess.check_call(
                    [
                        "bust",
                        tmp_sdf_output,
                        "-l",
                        abspath_sdf_ref,
                        "-p",
                        os.path.abspath(path_pdb_pocket_4bust),
                        "--outfmt",
                        "csv",
                        "--output",
                        abspath_sdf_output + "_bust.csv",
                    ]
                )


def launch_tasks(
    pathtuples: List[PathtupleForTask],
    args: argparse.Namespace,
) -> None:
    """ """
    if args.remote:
        for i_pathtuple in pathtuples:
            launch_1_task_remote(
                #####
                pathtuple=i_pathtuple,
                args=args,
            )
        return

    # take the first path to retrieve outputFolder
    i_pathtuple = pathtuples[0]
    if os.path.exists(i_pathtuple.ABSPATH_DIR_OUTPUT) and args.delete:
        print(" :: Output Ligand Folder exists, remove!")
        shutil.rmtree(i_pathtuple.ABSPATH_DIR_OUTPUT)

    Path(i_pathtuple.ABSPATH_DIR_OUTPUT).mkdir(
        parents=True,
        exist_ok=True,
    )

    for i_pathtuple in pathtuples:
        # if args.remote:
        #     launch_1_task_remote(pathtuple=i_pathtuple,)
        #     continue
        ########################################
        ##### 1. hint error
        ########################################

        if (
            #####
            (args.skip_finished)
            and os.path.exists(i_pathtuple.ABSPATH_CSV_STAT)
        ):
            print(
                (
                    #####
                    "\n\n ::   FINISHED :: [ {} ] :: \n\n"
                ).format(i_pathtuple.ABSPATH_SDF_LIGAND)
            )
            continue

        with open(i_pathtuple.ABSPATH_HINT_ERROR, "w+") as file_to_write:
            file_to_write.write(str(i_pathtuple))

        try:
            launch_1_task_local(
                #####
                pathtuple=i_pathtuple,
                args=args,
            )

            assert os.path.exists(i_pathtuple.ABSPATH_HINT_ERROR)
            os.remove(i_pathtuple.ABSPATH_HINT_ERROR)

        except Exception as err:
            str_exception = "\n\n" + traceback.format_exc()

            LOGGER.info(str_exception)
            with open(i_pathtuple.ABSPATH_HINT_ERROR, "a+") as file_to_append:
                file_to_append.write(str_exception)


def get_should_continue_to_handle(
    #####
    str_what: str = "continue to handle?",
) -> bool:
    """ """
    return (
        #####
        input(str_what + " [Y/N]:\n").strip().casefold()
        in [
            "y",
            "yes",
            "ok",
        ]
    )


def handle_a_entry_by_args(
    pathtuples: List[PathtupleForTask],
    args: argparse.Namespace,
) -> None:
    """ """
    LOGGER.info(
        (
            #####
            "\n\n"
            " :: all pdb_id FOUND :: \n"
            "{}\n\n"
            " :: COUNT :: [ {} ] :: \n"
            "\n\n"
        ).format(
            [
                #####
                x.STR_PDB_ID
                for x in pathtuples
            ],
            len(pathtuples),
        )
    )
    if len(pathtuples) > 1:
        if not args.yes:
            if not get_should_continue_to_handle("continue to restruct all?"):
                return

    launch_tasks(
        pathtuples=pathtuples,
        args=args,
    )


def handle_c_entry_by_args(
    pathtuples: List[PathtupleForTask],
    args: argparse.Namespace,
) -> None:
    """ """
    NUM_DETAILED_FINISHED_UNFINISHED = 10

    abspath_cwd = os.path.abspath(args.cwd)

    finished_pathtuples: List[PathtupleForTask] = []
    unfinished_pathtuples: List[PathtupleForTask] = []

    dataframes_for_concat: List[pd.DataFrame] = []

    for i_pathtuple in pathtuples:
        i_message = None
        if os.path.exists(i_pathtuple.ABSPATH_CSV_STAT):
            try:
                i_message = " :: {} :: finished :: [ {} ] :: ".format(
                    i_pathtuple.STR_PDB_ID,
                    os.path.relpath(
                        #####
                        i_pathtuple.ABSPATH_CSV_STAT,
                        abspath_cwd,
                    ),
                )
                finished_pathtuples.append(i_pathtuple)

                csvs = [pd.read_csv(
                    i_pathtuple.ABSPATH_CSV_STAT,
                    dtype={
                        #####
                        "pdb_id": str,
                    },
                )]
                # add bust
                bust_f = i_pathtuple.ABSPATH_SDF_OUTPUT + "_bust.csv"
                if os.path.exists(bust_f):
                    csvs.append(pd.read_csv(bust_f).add_prefix("bust_"))
                #
                i_csv_stat = pd.concat(
                    csvs,
                    axis=1,
                )
                dataframes_for_concat.append(i_csv_stat)
            except Exception:
                print(traceback.format_exc())

        else:
            i_message = " :: {} :: ".format(i_pathtuple.STR_PDB_ID)
            unfinished_pathtuples.append(i_pathtuple)

        LOGGER.info(i_message)

    LOGGER.info(
        (
            "\n\n"
            "\n\n :: UNFINISHED :: \n\n"
            "{}\n"
            "\n\n :: UNFINISHED :: [:{}] :: \n\n"
            "{}\n"
            #####
            "\n\n ::   FINISHED :: \n\n"
            "{}\n"
            "\n\n ::   FINISHED :: [:{}] :: \n\n"
            "{}\n"
            "\n\n"
        ).format(
            [x.STR_PDB_ID for x in unfinished_pathtuples],
            NUM_DETAILED_FINISHED_UNFINISHED,
            pprint.pformat(
                [
                    x._asdict()
                    for x in unfinished_pathtuples[:NUM_DETAILED_FINISHED_UNFINISHED]
                ]
            ),
            #####
            [x.STR_PDB_ID for x in finished_pathtuples],
            NUM_DETAILED_FINISHED_UNFINISHED,
            pprint.pformat(
                [
                    x._asdict()
                    for x in finished_pathtuples[:NUM_DETAILED_FINISHED_UNFINISHED]
                ]
            ),
        )
    )

    if len(finished_pathtuples) > 0:
        abspath_csv_concated = os.path.join(
            os.path.dirname(finished_pathtuples[0].ABSPATH_CSV_STAT),
            "stat_concated.csv",
        )
        dataframe_concated = pd.concat(
            dataframes_for_concat,
            axis=0,
            ignore_index=True,
        )
        LOGGER.info(
            (
                #####
                "\n\n"
                " :: DATAFRAME :: CONCATED :: \n"
                "{}\n"
                "\n\n"
            ).format(dataframe_concated)
        )

        # dataframe_concated = dataframe_concated.astype(
        #     {
        #         #####
        #         "pdb_id": str,
        #         "pose_rank": int,
        #     }
        # )
        dataframe_concated.to_csv(
            #####
            abspath_csv_concated,
            # index=False,
        )
        shutil.copyfile(
            abspath_csv_concated,
            os.path.join(
                abspath_cwd,
                "stat_{}.csv".format(
                    os.path.basename(
                        #####
                        os.path.dirname(abspath_csv_concated)
                    )
                ),
            ),
        )

        rows_rank0 = dataframe_concated[
            #####
            dataframe_concated["pose_rank"]
            == 0
            ]

        is_rmsd_ok = rows_rank0["rmsd"] < 2.0
        is_bust_ok = rows_rank0[
            "bust_minimum_distance_to_protein"] if "bust_minimum_distance_to_protein" in rows_rank0 else np.array(
            [0] * len(rows_rank0))
        is_rmsd_bust_ok = is_rmsd_ok & is_bust_ok

        LOGGER.info(
            (
                #####
                "\n\n"
                " :: _concated.csv saved to :: [ {} ] :: \n"
                " :: Top1 :: [ {} ]/[ {} ] :: \n"
                " :: [ {} ] :: \n"
                "\n\n"
                " ::      RMSD OK [ {} ]\n"
                " ::      BUST OK [ {} ] = {}%\n"
                " :: RMSD BUST OK [ {} ]\n"
                "\n\n"
            ).format(
                abspath_csv_concated,
                is_rmsd_ok.sum(),
                is_rmsd_ok.count(),
                1.0 * is_rmsd_ok.sum() / is_rmsd_ok.count(),
                is_rmsd_ok.sum(),
                is_bust_ok.sum(), round(is_bust_ok.sum() * 100. / is_rmsd_ok.count(), 3),
                is_rmsd_bust_ok.sum(),
            )
        )

    return


def main() -> None:
    """ """
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.INFO)

    args, mainparser = get_args_and_mainparser()

    ########################################
    ##### 1. pathtuples
    ########################################
    pathtuples = None

    if args.subparser == B_ENTRY_SUBPARSER:
        _ABSPATH_PKL_TASK = os.path.abspath(args.pkl_remote)
        assert os.path.exists(_ABSPATH_PKL_TASK)

        with open(_ABSPATH_PKL_TASK, "rb") as f2read:
            pathtuples, args = pickle.load(f2read)
    else:
        abspath_cwd = os.path.abspath(args.cwd)

        if args.find_all:
            pathtuples = PathtupleForTask.multifrom_path_cwd(abspath_cwd, args)
        elif args.block_pdb_ids is not None:
            abspath_block_pdb_ids = os.path.abspath(args.block_pdb_ids)
            with open(abspath_block_pdb_ids) as file_2read:
                pathtuples = [
                    PathtupleForTask.from_str_pdb_id(
                        #####
                        str_pdb_id=x,
                        path_cwd=abspath_cwd,
                        args=args,
                    )
                    for x in file_2read.read().split()
                ]
        else:
            pathtuples = [
                PathtupleForTask.from_str_pdb_id(
                    #####
                    str_pdb_id=args.pdb_id,
                    path_cwd=abspath_cwd,
                    args=args,
                ),
            ]

    ########################################
    ##### 2. handlers
    ########################################
    if args.subparser == A_ENTRY_SUBPARSER:
        handle_a_entry_by_args(pathtuples, args)
    elif args.subparser == C_ENTRY_SUBPARSER:
        handle_c_entry_by_args(pathtuples, args)
    else:
        mainparser.print_help()


if __name__ == "__main__":
    main()
