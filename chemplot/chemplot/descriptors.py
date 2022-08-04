# Authors: Murat Cihan Sorkun <mcsorkun@gmail.com>, Dajt Mullaj <dajt.mullai@gmail.com>
# Descriptor operation methods
#
# License: BSD 3 clause
from __future__ import print_function

import os

import pandas as pd
import math
import mordred
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import AllChem
from openeye import oechem
from mordred import Calculator, descriptors #Dont remove these imports
import prolif as plf
import oddt
from oddt.fingerprints import PLEC


def get_mordred_descriptors_from_smiles(smiles_list, target_list):
    """
    Calculates the Mordred descriptors for given smiles list

    :param smiles_list: List of smiles
    :type smiles_list: list
    :returns: The calculated descriptors list for the given smiles
    :rtype: Dataframe
    """

    return generate_mordred_descriptors(smiles_list, target_list, Chem.MolFromSmiles, 'SMILES')


def get_mordred_descriptors_from_inchi(inchi_list, target_list):
    """
    Calculates the Mordred descriptors for given InChi list

    :param inchi_list: List of InChi
    :type inchi_list: list
    :returns: The calculated descriptors list for the given smiles
    :rtype: Dataframe
    """

    return generate_mordred_descriptors(inchi_list, target_list, Chem.MolFromInchi, 'InChi')

def get_mordred_descriptors_from_sdf():
    pass

def get_mordred_descriptors_from_oechem(oechem_files, target_list):
    #default target_list is energies

    _, lig_path = oechem_files

    smiles_list = []

    ligands = oechem.oemolistream(lig_path)
    for idx, mol in enumerate(tqdm(ligands.GetOEGraphMols())):
        smiles_list.append(oechem.OEMolToSmiles(mol))

    return generate_mordred_descriptors(smiles_list, target_list, Chem.MolFromSmiles, 'SMILES')


def generate_mordred_descriptors(encoding_list, target_list, encoding_function, encoding_name):
    """
    Calculates the Mordred descriptors for list of molecules encodings

    :param smiles_list: List of molecules encodings
    :type smiles_list: list
    :returns: The calculated descriptors list for the given molecules encodings
    :rtype: Dataframe
    """

    if len(target_list) == 0:
        raise Exception("Target values missing")

    if len(target_list) > 0:
        if len(target_list) != len(encoding_list):
            raise Exception("If target is provided its length must match the instances of molecules")

    calc = mordred.Calculator()

    calc.register(mordred.AtomCount)        #16
    calc.register(mordred.RingCount)        #139
    calc.register(mordred.BondCount)        #9
    calc.register(mordred.HydrogenBond)     #2
    calc.register(mordred.CarbonTypes)      #10
    calc.register(mordred.SLogP)            #2
    calc.register(mordred.Constitutional)   #16
    calc.register(mordred.TopoPSA)          #2
    calc.register(mordred.Weight)           #2
    calc.register(mordred.Polarizability)   #2
    calc.register(mordred.McGowanVolume)    #1

    name_list=[]
    for desc_name in calc.descriptors:
        name_list.append(str(desc_name))

    mols=[]
    descriptors_list=[]
    erroneous_encodings=[]
    encodings_none_descriptors=[]
    for encoding in tqdm(encoding_list):
        mol=encoding_function(encoding)
        if mol is None:
            descriptors_list.append([None]*len(name_list))
            erroneous_encodings.append(encoding)
        else:
            mol=Chem.AddHs(mol)
            calculated_descriptors = calc(mol)
            for i in range(len(calculated_descriptors._values)):
                if math.isnan(calculated_descriptors._values[i]):
                    calculated_descriptors._values = [None]*len(name_list)
                    encodings_none_descriptors.append(encoding)
                    break
                if i == len(calculated_descriptors._values) - 1:
                    mols.append(mol)
            descriptors_list.append(calculated_descriptors._values)

    if len(erroneous_encodings)>0:
        print("The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(encoding_name, '\n'.join(map(str, erroneous_encodings)), encoding_name))

    if len(encodings_none_descriptors)>0:
        print("For the following {} not all descriptors can be computed:\n{}.\nThese {} will be removed from the data.".format(encoding_name, '\n'.join(map(str, encodings_none_descriptors)), encoding_name))

    df_descriptors=pd.DataFrame(descriptors_list, index=encoding_list, columns=name_list)
    df_descriptors = df_descriptors.select_dtypes(exclude=['object'])

    # Remove erroneous data
    if not isinstance(target_list,list): target_list = target_list.values
    df_descriptors = df_descriptors.assign(target=target_list)
    df_descriptors = df_descriptors.dropna(how='any')
    target_list = df_descriptors['target'].to_list()
    df_descriptors = df_descriptors.drop(columns=['target'])

    print(df_descriptors)
    print(target_list)

    return mols, df_descriptors, target_list

def select_descriptors_lasso(df_descriptors, target_list, R_select=0.05, C_select=0.3, kind="R"):
    """
    Selects descriptors by LASSO

    :param df_descriptors: descriptors of molecules
    :param target_list: list of target values
    :param R_select: alpha value for Lasso
    :param C_select: C value for LogisticRegression
    :param kind: kind of target R->Regression C->Classification
    :type df_descriptors: Dataframe
    :type target_list: list
    :type R_select: float
    :type C_select: float
    :type kind: string
    :returns: The selected descriptors
    :rtype: Dataframe
    """

    df_descriptors_scaled = StandardScaler().fit_transform(df_descriptors)

    if kind=="C":
        model = LogisticRegression(C=C_select,penalty='l1', solver='liblinear',random_state=1).fit(df_descriptors_scaled, target_list)
    else:
        model = Lasso(alpha=R_select,max_iter=10000,random_state=1).fit(df_descriptors_scaled, target_list)

    selected = SelectFromModel(model, prefit=True)
    X_new_lasso = selected.transform(df_descriptors)
    if X_new_lasso.size > 0:
        # Get back the kept features as a DataFrame with dropped columns as all 0s
        selected_features = pd.DataFrame(selected.inverse_transform(X_new_lasso), index=df_descriptors.index, columns=df_descriptors.columns)
        # Dropped columns have values of all 0s, keep other columns
        selected_columns_lasso = selected_features.columns[selected_features.var() != 0]
        selected_data = df_descriptors[selected_columns_lasso]
    else:
        # No features were selected
        selected_data = df_descriptors

    return selected_data, target_list


def get_ecfp_from_smiles(smiles_list, target_list, **kwargs):
    """
    Calculates the ECFP fingerprint for given SMILES list

    :param smiles_list: List of SMILES
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type radius: int
    :type smiles_list: list
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """

    return generate_ecfp(smiles_list, Chem.MolFromSmiles, 'SMILES', target_list, **kwargs)


def get_ecfp_from_inchi(inchi_list, target_list, **kwargs):
    """
    Calculates the ECFP fingerprint for given InChi list

    :param inchi_list: List of InChi
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type inchi_list: list
    :type radius: int
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given InChi
    :rtype: Dataframe
    """

    return generate_ecfp(inchi_list, Chem.MolFromInchi, 'InChi', target_list, **kwargs)


def get_ecfp_from_sdf(data_path, target_list, **kwargs):

    radius = kwargs['radius'] if 'radius' in kwargs else 2
    n_bits = kwargs['n_bits'] if 'n_bits' in kwargs else 2048

    file_names = os.listdir(data_path)
    lig_names = [i for i in file_names if '.sdf' in i].sort()

    if len(target_list) > 0:
        if len(target_list) != len(lig_names):
            raise Exception("If target is provided its length must match the instances of molecules")

    # Generate ECFP fingerprints
    mols=[]
    smiles=[]
    ecfp_fingerprints=[]
    erroneous_encodings=[]
    for lig_name in tqdm(lig_names):

        lig_path = os.path.join(data_path, lig_name)
        mol=Chem.SDMolSupplier(os.path.abspath(lig_path), sanitize=True, removeHs=False)[0]

        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_encodings.append(encoding)
        else:
            mols.append(mol)
            smiles.append(Chem.MolToSmiles(mol))
            list_bits_fingerprint = []
            list_bits_fingerprint[:0] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            ecfp_fingerprints.append(list_bits_fingerprint)



    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = smiles)

    # Remove erroneous data
    if len(erroneous_encodings)>0:
        print("The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(encoding_name, '\n'.join(map(str, erroneous_encodings)), encoding_name))

    if len(target_list)>0:
        if not isinstance(target_list,list): target_list = target_list.values
        df_ecfp_fingerprints = df_ecfp_fingerprints.assign(target=target_list)

    df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')

    if len(target_list)>0:
        target_list = df_ecfp_fingerprints['target'].to_list()
        df_ecfp_fingerprints = df_ecfp_fingerprints.drop(columns=['target'])

    # Remove bit columns with no variablity (all "0" or all "1")
    # df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 0).any(axis=0)]
    # df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 1).any(axis=0)]


    return mols, df_ecfp_fingerprints, target_list

def get_ecfp_from_inchi(inchi_list, target_list, **kwargs):
    """
    Calculates the ECFP fingerprint for given InChi list

    :param inchi_list: List of InChi
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type inchi_list: list
    :type radius: int
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given InChi
    :rtype: Dataframe
    """

    return generate_ecfp(inchi_list, Chem.MolFromInchi, 'InChi', target_list, **kwargs)


def get_ecfp_from_sdf_subdir(data_path, target_list, **kwargs):

    radius = kwargs['radius'] if 'radius' in kwargs else 2
    n_bits = kwargs['n_bits'] if 'n_bits' in kwargs else 2048

    # complex_names = os.listdir(data_path)
    f = open('./data/timesplit_test', 'r')
    complex_names = [line.rstrip() for line in f.readlines()]

    # Generate ECFP fingerprints
    mols=[]
    smiles=[]
    ecfp_fingerprints=[]
    erroneous_encodings=[]
    for c in tqdm(complex_names):
        lig_path_sdf = os.path.join(data_path, c, f'{c}_ligand.sdf')
        lig_path_mol2 = os.path.join(data_path, c, f'{c}_ligand.mol2')

        mol=Chem.SDMolSupplier(os.path.abspath(lig_path_sdf), sanitize=True, removeHs=False)[0]
        if (mol is None):
            mol = Chem.MolFromMol2File(os.path.abspath(lig_path_mol2), sanitize=False, removeHs=False)

        if mol is None:
            ecfp_fingerprints.append([None]*n_bits)
            erroneous_encodings.append(c)
        else:
            mols.append(mol)
            smiles.append(Chem.MolToSmiles(mol))
            list_bits_fingerprint = []
            list_bits_fingerprint[:0] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            ecfp_fingerprints.append(list_bits_fingerprint)



    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = smiles)

    # Remove erroneous data
    if len(erroneous_encodings)>0:
        print("The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(encoding_name, '\n'.join(map(str, erroneous_encodings)), encoding_name))

    if len(target_list)>0:
        if not isinstance(target_list,list): target_list = target_list.values
        df_ecfp_fingerprints = df_ecfp_fingerprints.assign(target=target_list)

    df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')

    if len(target_list)>0:
        target_list = df_ecfp_fingerprints['target'].to_list()
        df_ecfp_fingerprints = df_ecfp_fingerprints.drop(columns=['target'])

    # Remove bit columns with no variablity (all "0" or all "1")
    # df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 0).any(axis=0)]
    # df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 1).any(axis=0)]


    return mols, df_ecfp_fingerprints, target_list


def get_ecfp_from_oechem(oechem_files, target_list, **kwargs):

    _, lig_path = oechem_files

    smiles_list = []
    energies_list = []

    ligands = oechem.oemolistream(lig_path)
    for idx, mol in enumerate(tqdm(ligands.GetOEGraphMols())):
        smiles_list.append(oechem.OEMolToSmiles(mol))
        energies_list.append(mol.GetEnergy())

    if len(target_list) == 0:
        target_list = energies_list

    return generate_ecfp(smiles_list, Chem.MolFromSmiles, 'SMILES', target_list, **kwargs)

def generate_ecfp(encoding_list, encoding_function, encoding_name, target_list, **kwargs):
    """
    Calculates the ECFP fingerprint for given list of molecules encodings

    :param encoding_list: List of molecules encodings
    :param encoding_function: Function used to extract the molecules from the encodings
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type encoding_list: list
    :type encoding_function: fun
    :type radius: int
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given molecules encodings
    :rtype: Dataframe
    """

    radius = kwargs['radius'] if 'radius' in kwargs else 2
    n_bits = kwargs['n_bits'] if 'n_bits' in kwargs else 2048

    if len(target_list) > 0:
        if len(target_list) != len(encoding_list):
            raise Exception("If target is provided its length must match the instances of molecules")

    # Generate ECFP fingerprints
    mols=[]
    ecfp_fingerprints=[]
    erroneous_encodings=[]
    for encoding in tqdm(encoding_list):
        mol=encoding_function(encoding)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_encodings.append(encoding)
        else:
            mol=Chem.AddHs(mol)
            mols.append(mol)
            list_bits_fingerprint = []
            list_bits_fingerprint[:0] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            ecfp_fingerprints.append(list_bits_fingerprint)

    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = encoding_list)

    # Remove erroneous data
    if len(erroneous_encodings)>0:
        print("The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(encoding_name, '\n'.join(map(str, erroneous_encodings)), encoding_name))

    if len(target_list)>0:
        if not isinstance(target_list,list): target_list = target_list.values
        df_ecfp_fingerprints = df_ecfp_fingerprints.assign(target=target_list)

    df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')

    if len(target_list)>0:
        target_list = df_ecfp_fingerprints['target'].to_list()
        df_ecfp_fingerprints = df_ecfp_fingerprints.drop(columns=['target'])

    # Remove bit columns with no variablity (all "0" or all "1")
    # df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 0).any(axis=0)]
    # df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 1).any(axis=0)]

    return mols, df_ecfp_fingerprints, target_list

def get_plec_from_oechem(oechem_files, target_list, **kwargs):
    # Default target from oechem

    depth_ligand = kwargs['depth_ligand'] if 'depth_ligand' in kwargs else 2
    depth_protein = kwargs['depth_protein'] if 'depth_protein' in kwargs else 4
    distance_cutoff = kwargs['distance_cutoff'] if 'distance_cutoff' in kwargs else 4.5
    size = kwargs['size'] if 'size' in kwargs else 16384


    rec_path, lig_path = oechem_files
    temp_filename = 'temp.sdf'

    data = []
    mols = []
    smiles = []
    energies = []

    protein = next(oddt.toolkit.readfile('pdb', rec_path))
    protein.protein = True

    ligands = oechem.oemolistream(lig_path)
    temp = oechem.oemolostream()
    for idx, mol in enumerate(tqdm(ligands.GetOEGraphMols())):

        temp.open(temp_filename)
        oechem.OEWriteMolecule(temp, mol)
        ligand = next(oddt.toolkit.readfile('sdf', temp_filename))

        fp = PLEC(protein, ligand, depth_ligand=depth_ligand, depth_protein=depth_protein, distance_cutoff=distance_cutoff, size=size, sparse=False)
        data.append(fp)

        rdkit_mol = Chem.SDMolSupplier(temp_filename, sanitize=True, removeHs=False)[0]
        mols.append(rdkit_mol)

        smi = Chem.MolToSmiles(rdkit_mol)
        smiles.append(smi)

        energy = mol.GetEnergy()
        energies.append(energy)

    df_plec_fingerprints = pd.DataFrame(data=data, index=smiles)

    if len(target_list) == 0:
        target_list = energies
    else:
        if len(target_list) != len(df_plec_fingerprints):
            raise Exception("If target is provided its length must match the instances of molecules")

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return mols, df_plec_fingerprints, target_lists

def get_plec_from_sdf(data_path, target_list, **kwargs):

    depth_ligand = kwargs['depth_ligand'] if 'depth_ligand' in kwargs else 2
    depth_protein = kwargs['depth_protein'] if 'depth_protein' in kwargs else 4
    distance_cutoff = kwargs['distance_cutoff'] if 'distance_cutoff' in kwargs else 4.5
    size = kwargs['size'] if 'size' in kwargs else 16384

    data = []
    smiles = []
    mols = []

    file_names = os.listdir(data_path)

    rec_name = [i for i in file_names if '.pdb' in i or 'protein' in i][0]
    rec_path = os.path.join(data_path, rec_name)

    protein = next(oddt.toolkit.readfile('pdb', rec_path))
    protein.protein = True

    lig_names = [i for i in file_names if '.sdf' in i].sort()
    for idx, lig_name in enumerate(tqdm(lig_names)):

        lig_path = os.path.join(data_path, lig_name)
        ligand = next(oddt.toolkit.readfile('sdf', lig_path))

        fp = PLEC(protein, ligand, depth_ligand=depth_ligand, depth_protein=depth_protein, distance_cutoff=distance_cutoff, size=size, sparse=False)
        data.append(fp)

        rdkit_mol = Chem.SDMolSupplier(os.path.abspath(lig_path), sanitize=True, removeHs=False)[0]
        mols.append(rdkit_mol)

        smi = Chem.MolToSmiles(rdkit_mol)
        smiles.append(smi)

    df_plec_fingerprints = pd.DataFrame(data=data, index=smiles)

    return mols, df_plec_fingerprints, target_list

def get_plec_from_sdf_subdir(data_path, target_list, **kwargs):

    depth_ligand = kwargs['depth_ligand'] if 'depth_ligand' in kwargs else 2
    depth_protein = kwargs['depth_protein'] if 'depth_protein' in kwargs else 4
    distance_cutoff = kwargs['distance_cutoff'] if 'distance_cutoff' in kwargs else 4.5
    size = kwargs['size'] if 'size' in kwargs else 16384

    data = []
    smiles = []
    mols = []

    # complex_names = os.listdir(data_path)
    f = open('./data/timesplit_test', 'r')
    complex_names = [line.rstrip() for line in f.readlines()]

    for c in tqdm(complex_names):
        try:
            lig_path_sdf = os.path.join('./data/results/output/screen', c, f'{c}_lig_equibind_corrected.sdf')
            # lig_path_mol2 = os.path.join(data_path, c, f'{c}_ligand.mol2')
            rec_path = os.path.join(data_path, c, f'{c}_protein_processed.pdb')

            ligand = next(oddt.toolkit.readfile('sdf', lig_path_sdf))
            # if (ligand is None):
            #     ligand = next(oddt.toolkit.readfile('mol2', lig_path_mol2))

            protein = next(oddt.toolkit.readfile('pdb', rec_path))
            protein.protein = True

            fp = PLEC(ligand, protein, depth_ligand=depth_ligand, depth_protein=depth_protein, distance_cutoff=distance_cutoff, size=size, sparse=False)
            data.append(fp)

            rdkit_mol = Chem.SDMolSupplier(os.path.abspath(lig_path_sdf), sanitize=True, removeHs=False)[0]
            # if (rdkit_mol is None):
            #     rdkit_mol = Chem.MolFromMol2File(os.path.abspath(lig_path_mol2), sanitize=False, removeHs=False)
            mols.append(rdkit_mol)

            smi = Chem.MolToSmiles(rdkit_mol)
            smiles.append(smi)
        except:
            print(f'error on complex {c}')

    df_plec_fingerprints = pd.DataFrame(data=data, index=smiles)

    return mols, df_plec_fingerprints, target_list

def get_prolif_from_oechem(oechem_files, target_list):

    rec_path, lig_path = oechem_files
    temp_filename = 'temp.sdf'

    data = []
    mols = []
    smiles = []
    energies = []

    mol = Chem.MolFromPDBFile(rec_path, removeHs=False)
    protein = plf.Molecule(mol)

    ligands = oechem.oemolistream(lig_path)
    temp = oechem.oemolostream()
    for idx, mol in enumerate(tqdm(ligands.GetOEGraphMols())):

        temp.open(temp_filename)
        oechem.OEWriteMolecule(temp, mol)
        ligand = plf.sdf_supplier(temp_filename)

        fp = plf.Fingerprint()
        fp.run_from_iterable(ligand, protein, progress=False, residues='all')
        fp = fp.to_dataframe(drop_empty=False)
        fp = fp.iloc[0].values.astype(int)
        data.append(fp)

        rdkit_mol = Chem.SDMolSupplier(temp_filename, sanitize=True, removeHs=False)[0]
        mols.append(rdkit_mol)

        smi = Chem.MolToSmiles(rdkit_mol)
        smiles.append(smi)

        energy = mol.GetEnergy()
        energies.append(energy)

    df_prolif_fingerprints = pd.DataFrame(data=data, index=smiles)

    if len(target_list) == 0:
        target_list = energies
    else:
        if len(target_list) != len(df_prolif_fingerprints):
            raise Exception("If target is provided its length must match the instances of molecules")

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return mols, df_prolif_fingerprints, target_list


def get_prolif_from_sdf(data_path, target_list):

    data = []
    mols = []
    smiles = []

    file_names = os.listdir(data_path)

    rec_name = [i for i in file_names if '.pdb' in i or 'protein' in i][0]
    rec_path = os.path.join(data_path, rec_name)

    mol = Chem.MolFromPDBFile(rec_path, removeHs=False)
    protein = plf.Molecule(mol)

    lig_names = [i for i in file_names if '.sdf' in i].sort()
    for idx, lig_name in enumerate(tqdm(lig_names, leave=False)):

        lig_path = os.path.join(data_path, lig_name)
        ligand = plf.sdf_supplier(lig_path)

        fp = plf.Fingerprint()
        fp.run_from_iterable(ligand, protein, progress=False, residues='all')
        fp = fp.to_dataframe(drop_empty=False)
        fp = fp.iloc[0].values.astype(int)
        data.append(fp)

        rdkit_mol = Chem.SDMolSupplier(os.path.abspath(lig_path), sanitize=True, removeHs=False)[0]
        mols.append(rdkit_mol)

        smi = Chem.MolToSmiles(rdkit_mol)
        smiles.append(smi)

    df_prolif_fingerprints = pd.DataFrame(data=data, index=smiles)

    if len(target_list) > 0:
        if len(target_list) != len(df_prolif_fingerprints):
            raise Exception("If target is provided its length must match the instances of molecules")

    return mols, df_prolif_fingerprints, target_list

def get_prolif_from_sdf_subdir(data_path, target_list):

    data = []
    mols = []
    smiles = []

    # complex_names = os.listdir(data_path)
    f = open('./data/timesplit_test', 'r')
    complex_names = [line.rstrip() for line in f.readlines()]

    for c in tqdm(complex_names, leave=False):
        lig_path_sdf = os.path.join('./data/results/output/screen', c, f'{c}_lig_equibind_corrected.sdf')
        # lig_path_mol2 = os.path.join(data_path, c, f'{c}_ligand.mol2')
        rec_path = os.path.join(data_path, c, f'{c}_protein_processed.pdb')

        ligand = Chem.SDMolSupplier(os.path.abspath(lig_path_sdf), sanitize=True, removeHs=False)[0]
        # if (ligand is None):
        #     ligand = Chem.MolFromMol2File(os.path.abspath(lig_path_mol2), sanitize=False, removeHs=False)
        if (ligand is None):
            print(f'error processing ligand for complex {c}')
            continue
        try:
            lig = plf.Molecule.from_rdkit(ligand)
        except:
            continue
        
        mol = Chem.MolFromPDBFile(rec_path, removeHs=False)
        try:
            protein = plf.Molecule.from_rdkit(mol)
        except:
            print(f'error processing protein for complex {c}')
            continue

        fp = plf.Fingerprint()
        fp.run_from_iterable([lig], protein, progress=False, residues='all')

        fp = fp.to_dataframe(drop_empty=False)
        fp = fp.iloc[0].values.astype(int)
        data.append(fp)

        rdkit_mol = Chem.SDMolSupplier(os.path.abspath(lig_path_sdf), sanitize=True, removeHs=False)[0]
        # if (rdkit_mol is None):
        #     rdkit_mol = Chem.MolFromMol2File(os.path.abspath(lig_path_mol2), sanitize=False, removeHs=False)

        mols.append(rdkit_mol)

        smi = Chem.MolToSmiles(rdkit_mol)
        smiles.append(smi)

    df_prolif_fingerprints = pd.DataFrame(data=data, index=smiles)
    df_prolif_fingerprints = df_prolif_fingerprints.fillna(0)
    print(df_prolif_fingerprints.shape)

    if len(target_list) > 0:
        if len(target_list) != len(df_prolif_fingerprints):
            raise Exception("If target is provided its length must match the instances of molecules")

    return mols, df_prolif_fingerprints, target_list
