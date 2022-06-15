import os
from openeye import oechem
from openeye import oedocking
import numpy as np

for complex in os.listdir('data/PDBBind'):
    lig_path = os.path.join('data/PDBBind', complex, f'{complex}_ligand.sdf')
    rec_path = os.path.join('data/PDBBind', complex, f'{complex}_protein_processed.pdb')
    # score default pdbbind docking pose and record
    # get receptor as graph mol(no receptor data)
    receptor = oechem.OEGraphMol()
    rec_istream = oechem.oemolistream(rec_path)
    if not oechem.OEReadPDBFile(rec_istream, receptor):
        oechem.OEThrow.Fatal("Unable to read receptor")
    # get lig as graph mol
    lig_mol = oechem.OEGraphMol()
    istream = oechem.oemolistream(lig_path)
    if not oechem.OEReadMolecule(istream, lig_mol):
        oechem.OEThrow.Fatal("Unable to read ligand")
    # find bounding box +-some flexibility around ligand
    lig_coords_list = [[x[0], x[1], x[2]] for x in lig_mol.GetCoords().values()]
    mins = np.min(lig_coords_list, axis=0)
    maxes = np.max(lig_coords_list, axis=0)

    bounding_box = oedocking.OEBox(*mins, *maxes)
    # initialize score obj 
    score = oedocking.OEScore()
    score.Initialize(receptor, bounding_box)
    #score.SystematicSolidBodyOptimize(lig_mol)
    rescore = score.ScoreLigand(lig_mol)
    print(f'complex {complex}')
    print(rescore)