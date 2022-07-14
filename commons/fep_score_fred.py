import os
from openeye import oechem, oedocking

fep_base_path = './data/FEPBenchmark' #replace with fepbenchmark base path
benchmark_sets = os.listdir(fep_base_path)
scores = []

for set in benchmark_sets:
    dir = os.path.join(fep_base_path, set)
    for file in os.listdir(dir):
        if 'prepared.pdb' in file:
            rec_file = os.path.join(dir, file)
        elif file == 'all.sdf':
            lig_file = os.path.join(dir, file)
            
    # input prep for fred(oedu receptor construction) as fred needs protein in oedu form
    rec = oechem.OEGraphMol()
    rec_istream = oechem.oemolistream(rec_file)
    if not oechem.OEReadPDBFile(rec_istream, rec):
        oechem.OEThrow.Fatal("Unable to read receptor")

    ligs =  oechem.OEGraphMol()
    istream = oechem.oemolistream(lig_file)
    if not oechem.OEReadMolecule(istream, ligs):
        oechem.OEThrow.Fatal("Unable to read ligands")

    oerec = oechem.OEDesignUnit(rec, ligs)
    oerecoptions = oedocking.OEMakeReceptorOptions()
    oedocking.OEMakeReceptor(oerec, oerecoptions)

    if not oechem.OEWriteDesignUnit(f'{set}_rec.oedu', oerec):
        oechem.OEThrow.Fatal("Unable to write receptor to oedu")
    
    # calls fred and saves scores to {set}_fred_scores.txt
    # there doesnt seem to be a score_only option like smina so this runs docking before hand
    os.system(f'fred -receptor {set}_rec.oedu -dbase {lig_file} -score_file {set}_fred_scores.txt -no_extra_output_files -nostructs')
