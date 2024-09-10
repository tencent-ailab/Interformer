import glob
import os.path

from deepdock.prepare_target.computeTargetMesh import compute_inp_surface
import trimesh

root = './'

for pdb_f in glob.glob(f'{root}/*.pdb'):
    pdb = os.path.basename(pdb_f)[:4]
    out_f = f'{pdb}_pocket.ply'
    print(pdb)
    if os.path.exists(out_f):
        continue
    ligand_filename = f'{root}/{pdb}_docked.sdf'
    compute_inp_surface(pdb_f, ligand_filename, dist_threshold=10)
    if os.path.exists(out_f):
        mesh = trimesh.load_mesh(pdb + '_pocket.ply')
        print('Number of nodes: ', len(mesh.vertices))
    # mesh.show()
