######
# Energy
# PYTHONPATH=interformer python interformer/pre.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
# -work_path /opt/home/revoli/data_worker/interformer/poses \
# -filter_type normal \
# -dataset sbdd \
# -ligand_folder ligand/rcsb \
# -reload
######
# Affinity
export TMPDIR=~/tmp
export TEMP=~/tmp
export TMP=~/tmp
PYTHONPATH=interformer python interformer/pre.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020_round0_full.csv \
-work_path /opt/home/revoli/data_worker/interformer/poses \
-filter_type full \
-dataset sbdd \
-reload \
-affinity_pre
###
# Affinity-Normal
#PYTHONPATH=interformer python interformer/pre.py -data_path /opt/home/revoli/data_worker/interformer/train/general_PL_2020.csv \
#-work_path /opt/home/revoli/data_worker/interformer/poses \
#-ligand ligand/final \
#-filter_type normal \
#-dataset sbdd \
#-reload \
#-affinity_pre
