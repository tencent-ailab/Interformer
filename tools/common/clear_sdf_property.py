from rdkit import Chem
import sys

# 读取SDF文件
suppl = Chem.SDMolSupplier(sys.argv[1])

# 创建一个新的SDF写入器
writer = Chem.SDWriter(sys.argv[2])

# 遍历SDF文件中的分子
for mol in suppl:
    if mol is not None:  # 检查分子是否有效
        # 保留分子名称
        name = mol.GetProp('_Name')
        prop_names = mol.GetPropNames()
        # 删除所有属性
        for name in prop_names:
          mol.ClearProp(name)
        # mol.ClearPropNames()
        # mol.SetProp('_Name', name)
        # 将分子写入新的SDF文件
        writer.write(mol)

# 关闭写入器
writer.close()
