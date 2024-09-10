from feats.gnina_types.gnina_featurizer import obabel_mol_parser, PLIPAtomFeaturizer, PLIPEdgeFeaturizer, \
    obabel_ppi_parser
from feats.residue.residue_featurizer import ReisdueCAFeaturizer, residue_parser

model_list = ['Interformer', 'PPIScorer']


def get_exp_configure(exp):
    print(f"$$$${exp}$$$$")
    model_name, query_feat = exp.split('_')

    # Obabel
    gnina_conf = {
        'node_featurizer': PLIPAtomFeaturizer(),
        'edge_featurizer': None,
        'complex_to_data': obabel_mol_parser,
        'angle_feat_size': 0
    }
    gnina2_conf = {
        'node_featurizer': PLIPAtomFeaturizer(),
        'edge_featurizer': PLIPEdgeFeaturizer(covalent_bond=False),
        'complex_to_data': obabel_mol_parser,
        'angle_feat_size': 0
    }
    gnina3_conf = {
        'node_featurizer': PLIPAtomFeaturizer(),
        'edge_featurizer': PLIPEdgeFeaturizer(covalent_bond=False, interaction_angle=True),
        'complex_to_data': obabel_mol_parser,
        'angle_feat_size': 1
    }
    # PPI
    ppi_gnina_conf = gnina2_conf.copy()
    ppi_gnina_conf['complex_to_data'] = obabel_ppi_parser
    # PPI-Reisdue CA only
    ppi_residue_conf = {
        'node_featurizer': ReisdueCAFeaturizer(),
        'edge_featurizer': ReisdueCAFeaturizer(),
        'complex_to_data': residue_parser,
    }
    #####
    # Description
    feature_set = {'Gnina1': gnina_conf, 'Gnina2': gnina2_conf, 'Gnina3': gnina3_conf, 'PPI-Gnina': ppi_gnina_conf,
                   'PPI-Residue': ppi_residue_conf}
    exp_dict = feature_set[query_feat]
    # refresh feature size
    exp_dict['node_feat_size'] = exp_dict['node_featurizer'].feat_size('hv')
    exp_dict['edge_feat_size'] = exp_dict['edge_featurizer'].feat_size() if exp_dict['edge_featurizer'] else 0
    # refresh angle feat size
    exp_dict['angle_feat_size'] = exp_dict.get('angle_feat_size', 6)
    return exp_dict
