import os

fep_base_path = './data/FEPBenchmark' #replace with fepbenchmark base path
smina_path = './smina.static' # replace with path to smina.static binary
benchmark_sets = os.listdir(fep_base_path)
scores = []

for set in benchmark_sets:
    dir = os.path.join(fep_base_path, set)
    for file in os.listdir(dir):
        if 'prepared.pdb' in file:
            rec_file = os.path.join(dir, file)
        elif file == 'ligands.sdf':
            lig_file = os.path.join(dir, file)
    # scores saved to {set}_scores.txt as a log file, may need to do some additional parsing
    os.system(f'{smina_path} -r {rec_file} -l {lig_file} --score_only --log {set}_smina_scores.txt')
