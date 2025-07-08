This is a virtual screening demonstration where numerous ligands are docked into a rigid protein pocket, and their affinity values are scored. For this example, we are using a dataset located in the `vs_examples/` folder.
```
vs_example/
├── ligand
│   └── 2qbr_docked.sdf (Ensure that it is a bound ligand conformation inside the binding pocket, as it is the best choice. This sdf file can be obtained from RCSB website.)
├── pocket
│   └── 2qbr_pocket.pdb (A prepared protein structure, it can be the entire protein or pocket structure)
└── uff
    └── 2qbr_uff.sdf (Prepare a force filed minimized SDF file containing all the ligands you wish to dock.)
```
`vs_inference.sh` is the script for virtual screening on one protein with many ligands.
Please note that the docking program will skip ligands that fail processing or contain more than 100 atoms.
```
# copy vs_example/ and script to the Interformer/ soure code folder
cp -r vs_example/ vs_inference.sh ../../
cd ../../
# start virtual screening using Interformer
sh vs_inference.sh
```
