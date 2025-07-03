import sys

from rdkit import Chem

if __name__ == '__main__':
    output_txt = 'smiles.csv'
    input_sdf = sys.argv[1]
    #
    with open(output_txt, "w") as f_out:
        suppl = Chem.SDMolSupplier(input_sdf)
        f_out.write('SMILES,Name\n')
        for mol in suppl:
            if mol is not None:
                smiles = mol.GetProp('SMILES')
                name = mol.GetProp("Name")  # 获取分子名称（需 SDF 中有此属性）
                f_out.write(f"{smiles},{name}\n")  # 输出 SMILES 和名称，用制表符分隔