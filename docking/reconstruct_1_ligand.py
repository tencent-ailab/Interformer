# -*- coding: utf-8 -*-
#
# Copyright @ 2022 Tencent.com


"""
"""

import argparse
import logging

from reconstruct_ligands import reconstruct_1_ligand_given_paths

LOGGER = logging.getLogger(__name__)


def get_args_and_mainparser():
    """
    """
    mainparser = argparse.ArgumentParser()

    mainparser.add_argument(
        "--sdf_ligand", type=str, required=True,
    )
    mainparser.add_argument(
        "--sdf_ref", type=str, required=True,
    )
    mainparser.add_argument(
        "--pdb_complex", type=str, required=True,
    )
    mainparser.add_argument(
        "--pkl_normalscore", type=str, required=True,
    )
    mainparser.add_argument(
        "--sdf_output", type=str, required=True,
    )
    mainparser.add_argument("--csv_stat", type=str, default=None)
    mainparser.add_argument("--pdb_id", type=str, default=None)
    mainparser.add_argument(
        "--weight_intra",
        type=float,
        default=None,
        help="use --weight_intra=0.0 to reproduce the paper result (top1=0.56);"
             " use --weight_intra=30.0 to get a much better performance (top1=0.70)",
    )
    mainparser.add_argument(
        "--weight_collision_inter",
        type=float,
        default=None,
        help="if None, --weight_collision_inter=1.0",
    )

    args = mainparser.parse_args()

    return args, mainparser


def main() -> None:
    """
    """
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.INFO)

    args, mainparser = get_args_and_mainparser()

    reconstruct_1_ligand_given_paths(
        path_sdf_ligand=args.sdf_ligand,
        path_sdf_ref=args.sdf_ref,
        path_pdb_complex=args.pdb_complex,
        path_pkl_normalscore=args.pkl_normalscore,
        path_sdf_output=args.sdf_output,
        path_csv_stat=args.csv_stat,
        str_pdb_id=args.pdb_id,
        weight_intra=args.weight_intra,
        weight_collision_inter=args.weight_collision_inter,
    )


if __name__ == "__main__":
    main()
