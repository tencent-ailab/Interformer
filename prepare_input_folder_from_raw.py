import os
import subprocess
import argparse
import glob
import shutil  # For moving files

def create_directories(output_folder):
    """Create necessary directories for output."""
    os.makedirs(os.path.join(output_folder, "ligand"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "raw", "pocket"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "pocket"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "uff"), exist_ok=True)  # Create the uff folder
    print(f"Created directories in {output_folder}")

def find_required_files(raw_folder):
    """Find required .sdf and .pdb files in the raw folder."""
    sdf_files = glob.glob(os.path.join(raw_folder, "*.sdf"))
    pdb_files = glob.glob(os.path.join(raw_folder, "*.pdb"))

    if not sdf_files:
        raise FileNotFoundError("No .sdf file found in the raw folder.")
    if not pdb_files:
        raise FileNotFoundError("No .pdb file found in the raw folder.")

    print(f"Found ligand file: {sdf_files[0]}")
    print(f"Found protein file: {pdb_files[0]}")
    return sdf_files[0], pdb_files[0]  # Return the first found .sdf and .pdb

def get_output_filenames(ligand_sdf, protein_pdb, output_folder):
    """Generate output filenames based on input filenames."""
    ligand_base = os.path.splitext(os.path.basename(ligand_sdf))[0]  # e.g., 2qbr_ligand
    protein_base = os.path.splitext(os.path.basename(protein_pdb))[0]  # e.g., 2qbr

    docked_ligand_sdf = os.path.join(output_folder, "ligand", f"{ligand_base.replace('_ligand', '_docked')}.sdf")
    reduced_protein_pdb = os.path.join(output_folder, "raw", "pocket", f"{protein_base}_reduce.pdb")
    output_pocket_pdb = os.path.join(output_folder, "pocket", f"{protein_base}_pocket.pdb")
    uff_folder = os.path.join(output_folder, "uff")  # Path for UFF folder

    print(f"Output filenames generated:")
    print(f"  Docked ligand: {docked_ligand_sdf}")
    print(f"  Reduced protein: {reduced_protein_pdb}")
    print(f"  Pocket file: {output_pocket_pdb}")
    return docked_ligand_sdf, reduced_protein_pdb, output_pocket_pdb, uff_folder

def process_ligand(ligand_sdf, docked_ligand_sdf, uff_folder):
    """Process the ligand: protonate, add hydrogen, and generate conformations."""
    print(f"Processing ligand: {ligand_sdf}")
    cmd_protonate = f"obabel {ligand_sdf} -p 7.4 -O {docked_ligand_sdf}"
    subprocess.run(cmd_protonate, shell=True, check=True)
    print(f"Protonated ligand saved as: {docked_ligand_sdf}")

    # Generate initial ligand conformation using UFF
    cmd_gen_conformation = f"python tools/rdkit_ETKDG_3d_gen.py {os.path.dirname(docked_ligand_sdf)} {uff_folder}"
    subprocess.run(cmd_gen_conformation, shell=True, check=True)
    print(f"Initial ligand conformation generated in UFF folder: {uff_folder}")

def process_protein(protein_pdb, reduced_protein_pdb, output_pocket_pdb, docked_ligand_folder, remove_ccd):
    """Process the protein: reduce and extract the pocket."""
    print(f"Processing protein: {protein_pdb}")
    cmd_reduce = f"reduce {protein_pdb} > {reduced_protein_pdb}"
    subprocess.run(cmd_reduce, shell=True, check=True)
    print(f"Reduced protein saved as: {reduced_protein_pdb}")

    # Extract the pocket within 10 Å around the reference ligand
    print(f"Extracting pocket using ligand ...")
    cmd_extract_pocket = f"python tools/extract_pocket_by_ligand.py {os.path.dirname(reduced_protein_pdb)} {docked_ligand_folder} {remove_ccd}"
    subprocess.run(cmd_extract_pocket, shell=True, check=True)

    # Move the extracted pocket file to the output folder
    shutil.move(os.path.join(os.path.dirname(reduced_protein_pdb), "output", f"{os.path.basename(output_pocket_pdb)}"), output_pocket_pdb)
    print(f"Extracted pocket saved as: {output_pocket_pdb}")

def main():
    parser = argparse.ArgumentParser(description="Process ligand and protein files.")
    parser.add_argument('--raw_folder', '-r', required=True, help='Path to the raw folder containing input files.')
    parser.add_argument('--output_folder', '-o', required=True, help='Path to the output folder for results.')
    parser.add_argument('--remove_ccd', '-ccd', type=int, choices=[0, 1], default=1, help='Remove CCD ligand (1) or not (0).')

    args = parser.parse_args()

    # Find required files
    ligand_sdf, protein_pdb = find_required_files(args.raw_folder)

    # Generate output filenames based on input filenames
    docked_ligand_sdf, reduced_protein_pdb, output_pocket_pdb, uff_folder = get_output_filenames(ligand_sdf, protein_pdb, args.output_folder)

    create_directories(args.output_folder)
    process_ligand(ligand_sdf, docked_ligand_sdf, uff_folder)
    process_protein(protein_pdb, reduced_protein_pdb, output_pocket_pdb, os.path.dirname(docked_ligand_sdf), args.remove_ccd)

    print("Processing completed successfully!")

if __name__ == "__main__":
    main()