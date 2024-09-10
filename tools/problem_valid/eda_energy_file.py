import pickle

if __name__ == '__main__':
    root = '/opt/home/revoli/eva/Interformer/energy_output'
    pdb = '5S8I'
    data = pickle.load(open(f'{root}/gaussian_predict/{pdb}_G.pkl', 'rb'))

    d, ligand_len = data['d'], int(data['ligand_len'][0])

    inter_d = d[:ligand_len, ligand_len:]  # min=-0.24

    print('done')
