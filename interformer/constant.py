# CONSTANT
#########
# Molecule setting
affinity_max_amino_nodes = 230  # lower this value in order to decrease gpu memory to make a larger batch size
energy_max_amino_nodes = 400
max_ligand_atoms = 100
min_ligand_atoms = 3
####
# Graph setting
attention_dist_cutoff = 5.3  # It is for affinity & pose-selection,
# the unit is VdW radius distance, two C atoms distance should be 2.0 + 2.0 + d (VdW radisus),
# so it is equivalent to Euclidean distance 8
subgraph0_dist = 10.  # default=10.  # control the maximum distance to residues atoms
extract_pocket_max_dist = 7  # default=7.  # control the pocket size
consider_ligand_cofactor = True
#########
