import os.path
from collections import defaultdict

import pandas as pd
import glob
import numpy as np


def grep_pdbs():
    all = []
    df = pd.read_csv(root + '/train/general_PL_2020.csv')

    all.extend(df['Target'].unique().tolist())
    all = list(set(all))
    return all


def grep_seqs(chain_ids):
    # only take those seqs in chains_ids with the responding chain
    seqs = open(f'{output_folder}/pdb_seqres.txt').readlines()
    data = {}
    for i in range(0, len(seqs), 2):
        if 'protein' in seqs[i]:
            pdb, curr_chain = seqs[i][1:5], seqs[i][6]
            if pdb in chain_ids and curr_chain == chain_ids[pdb]:
                data[pdb] = seqs[i + 1].strip()
            # grep the maximum seqs as the pdb seq
            # if pdb in data and len(seqs[i + 1]) < len(data[pdb]):
            #   continue
    return data


def write(pdbs, seqs_map, name):
    f = open(f'{output_folder}/{name}.fasta', 'w')
    for pdb in pdbs:
        if pdb in seqs_map:
            f.write(f'>tr|{pdb}\n')
            f.write(seqs_map[pdb] + '\n')
        else:
            print(pdb, end=',')
    f.close()


def grep_chain_ids():
    pocket_root = f'{output_folder}/pocket'
    chain_ids = {}
    for pdb_f in glob.glob(f'{pocket_root}/*.pdb'):
        pdb = os.path.basename(pdb_f)[:4]
        chain_id = open(f'{pocket_root}/{pdb}_pocket.pdb').readlines()[100][21]  # 100th line
        chain_ids[pdb] = chain_id
    return chain_ids


def grep_avg_sim_by_m8_file(test_f):
    data = open(test_f).readlines()
    stats = defaultdict(float)
    for line in data:
        line = line.split('\t')
        query_pdb, score = line[0], line[2]
        stats[query_pdb] = max(float(score), stats[query_pdb])
    #
    avg = np.mean(list(stats.values()))
    return len(stats), avg


def mmseqs2_cal(list_of_test):
    for test_f in list_of_test:
        # (1,2) identifiers for query and target sequences/profiles, (3) sequence identity, (4) alignment length,
        # (5) number of mismatches, (6) number of gap openings, (7-8, 9-10) domain start and end-position in query
        # and in target, (11) E-value, and (12) bit score.
        output_f = f'{output_folder}/a8_files/{test_f}.m8'
        os.system(
            f"mmseqs easy-search --min-aln-len 100 {output_folder}/{test_f}.fasta {output_folder}/train.fasta"
            f" {output_f} /tmp >/dev/null")
        n, score = grep_avg_sim_by_m8_file(output_f)
        print(test_f, f'avg_sim:{score}, n:{n}')
        pass


if __name__ == '__main__':
    # pre-requirement
    # conda install -c conda-forge -c bioconda mmseqs2  # install  mmseqs2
    # wget https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz  # download all seqs from rcsb
    # put all pocket pdb into {output_folder}/pocket
    #
    root = '/opt/home/revoli/data_worker/interformer'
    output_folder = '/opt/home/revoli/data_worker/paper/benchmark/docking/similarity/sequence'
    # gather pdbs list
    core = [x.strip() for x in open(f'{root}/train/diffdock_splits/coresetlist').readlines()]
    test = [x.strip() for x in open(f'{root}/train/diffdock_splits/timesplit_test').readlines()]
    train = [x.strip() for x in open(f'{root}/train/diffdock_splits/timesplit_no_lig_overlap_train').readlines()]
    train = [x for x in train if x not in core]  # exclude coreset
    test_no_rec = [x.strip() for x in open(f'{root}/train/diffdock_splits/timesplit_test_no_rec_overlap').readlines()]
    kinase = pd.read_csv(f'/opt/home/revoli/data_worker/paper/benchmark/affinity/kinase_test.csv')['Target'].unique().tolist()
    print(f"num kinase targets:{len(kinase)}")
    covid = ['7rfs']
    lsd1 = ['6w4k']
    # load data
    pdbs = core + test + train + kinase + covid + lsd1
    w = open(f'{output_folder}/pdb_list', 'w')
    w.write('\n'.join(pdbs))
    w.close()
    #
    chain_ids = grep_chain_ids()
    seqs_map = grep_seqs(chain_ids)
    #
    write(train, seqs_map, name='train')
    write(core, seqs_map, name='core')
    write(test, seqs_map, name='test')
    write(test_no_rec, seqs_map, name='test_no_rec')
    write(covid, seqs_map, name='covid')
    write(lsd1, seqs_map, name='lsd1')
    write(kinase, seqs_map, name='kinase')
    #
    os.makedirs(f'{output_folder}/a8_files', exist_ok=True)
    list_of_test = ['core', 'test', 'test_no_rec', 'covid', 'lsd1', 'kinase']
    mmseqs2_cal(list_of_test)
    print('done')
