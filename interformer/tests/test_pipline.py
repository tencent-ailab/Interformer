from feats.gnina_types.obabel_api import merge_sdf_pdb_by_rdkit
from feats.dgl_feat.mol_construct import complex_to_data
from feats.third_rd_lib import load_by_rdkit
from utils.configure import get_exp_configure

if __name__ == '__main__':
    # training
    pdb_id = '2rjp'
    root = '/opt/home/revoli/data_worker/raw_data/redock'
    test_sdf = f'{root}/ligands/{pdb_id}_docked.sdf'
    pdb_file = f'{root}/pocket/{pdb_id}_pocket.pdb'
    # inference
    # root = '/opt/home/revoli/data_worker/v2019-docking'
    # # test_sdf = f'{root}/infer/{pdb_id}_docked.sdf'
    # test_sdf = f'{root}/ligands/{pdb_id}_docked.sdf'
    # pdb_file = f'{root}/pocket/{pdb_id}_pocket.pdb'
    #
    ligand_mols = load_by_rdkit(test_sdf)
    pdb_mol = load_by_rdkit(pdb_file, format='pdb')
    # function test
    min_key = [x for x in list(ligand_mols.keys()) if 'min' in x]
    if len(min_key):
        min_key = min_key[0]
    else:
        min_key = list(ligand_mols.keys())[0]
    l_first = ligand_mols[min_key][0]
    ###
    # Unit Test
    merge_data = merge_sdf_pdb_by_rdkit([pdb_id, l_first, pdb_mol])
    merge1_data = merge_sdf_pdb_by_rdkit([pdb_id, l_first, pdb_mol])
    ###
    graph4 = get_exp_configure('1_Graph4')
    feats = complex_to_data(merge_data, graph4['node_featurizer'], graph4['edge_featurizer'])
