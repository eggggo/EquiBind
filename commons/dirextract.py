import os

f = open('posgtcomplex.txt', 'r')
for line in f:
    complex = line.split(':')[0].strip()
    os.system(f'cp -R ./data/PDBBind/{complex} ./poscomplex')