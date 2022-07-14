import os

exit_code = os.system('python train.py --config=configs_clean/RDKitCoords_flexible_self_docking.yml')
while (exit_code != 0):
    exit_code = os.system('python train.py --config=configs_clean/RDKitCoords_flexible_self_docking.yml')
