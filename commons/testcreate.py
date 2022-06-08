import os

f = open("litpcba_test", "w")
data_dir = '../data/LitPCBA'

seen = set()
for subdir in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, subdir)):
        if file == 'actives.smi' or file == 'inactives.smi':
            continue
        else:
            name = file.split('_')[0]
            if not f'{subdir}/{name}' in seen:
                seen.add(f'{subdir}/{name}')

for lppair in seen:
    f.write(lppair)
    f.write('\n')
