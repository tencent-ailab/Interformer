import sys

import pandas as pd
from rdkit import Chem


if __name__ == '__main__':
    output_txt = 'smiles.csv'
    input_sdf = sys.argv[1]
    has_catalog = False
    #
    with open(output_txt, "w") as f_out:
        suppl = Chem.SDMolSupplier(input_sdf)
        data = []
        for mol in suppl:
            if mol is not None:
                smiles = mol.GetProp('SMILES')
                name = mol.GetProp("Name")  # 获取分子名称（需 SDF 中有此属性）
                row = [smiles, name]
                if has_catalog:
                    no = mol.GetProp('Catalog_NO')
                    row.append(no)
                data.append(row)

        columns = ["SMILES", "Name"]
        if has_catalog:
            columns.append("Catalog_NO")
        df = pd.DataFrame(data, columns=columns).to_csv(output_txt, index=False)
