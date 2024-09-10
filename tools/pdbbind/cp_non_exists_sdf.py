# it is a simple copy files tools, for copying sdf to another folder
import glob
import os.path

full_set = '/opt/home/revoli/data_worker/interformer/poses/ligand/20240620/pdbbank'
gt_set = '/opt/home/revoli/data_worker/pdbbind/2020/v2020-other-PL/index/pdb/ligand'
uff_set = '/opt/home/revoli/data_worker/pdbbind/2020/v2020-other-PL/index/pdb/ligand/uff'

full = glob.glob(full_set + '/*.sdf')

for sdf_f in full:
    sdf_name = os.path.basename(sdf_f)
    pdb = sdf_name[:4]
    gt_f = f'{gt_set}/{sdf_name}'
    if not os.path.exists(gt_f):
        print(f'missing->{sdf_name}')
        os.system(f'cp {sdf_f} {gt_f}')
    # uff
    uff_f = f'{uff_set}/{pdb}_uff.sdf'
    if not os.path.exists(uff_f):
        print(f'missing uff->{sdf_name}')
        os.system(f'cp {sdf_f} {uff_f}')
