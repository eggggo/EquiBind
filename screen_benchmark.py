import argparse
import sys

from copy import deepcopy

import os

from dgl import load_graphs

from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm

from commons.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, apply_changes
from commons.logger import Logger
from commons.process_mols import read_molecule, get_lig_graph_revised, \
    get_rec_graph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference

from train import load_model

from datasets.pdbbind import PDBBind
from datasets.litpcba import LitPCBA
from datasets.fepbenchmark import FEPBenchmark

from commons.utils import seed_all, read_strings_from_txt

import yaml

from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader

from trainer.metrics import Rsquared, MeanPredictorLoss, MAE, PearsonR, RMSD, RMSDfraction, CentroidDist, \
    CentroidDistFraction, RMSDmedian, CentroidDistMedian

# turn on for debugging C code like Segmentation Faults
import faulthandler

from openeye import oechem
from openeye import oedocking

from openbabel import openbabel

faulthandler.enable()


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/inference.yml')
    p.add_argument('--checkpoint', type=str, help='path to .pt file in a checkpoint directory')
    p.add_argument('--output_directory', type=str, default=None, help='path where to put the predicted results')
    p.add_argument('--run_corrections', type=bool, default=False,
                   help='whether or not to run the fast point cloud ligand fitting')
    p.add_argument('--run_dirs', type=list, default=[], help='path directory with saved runs')
    p.add_argument('--fine_tune_dirs', type=list, default=[], help='path directory with saved finetuning runs')
    p.add_argument('--ligand_to_screen', type=str, help='path to ligand to screen')
    p.add_argument('--protein_set', type=str, help='path to list of proteins')
    p.add_argument('--mode', type=str, help='n for none, q for qvina2, s for smina')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset_params', type=dict, default={},
                   help='parameters with keywords of the dataset')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=1, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--clip_grad', type=float, default=None, help='clip gradients if magnitude is greater')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--trainer', type=str, default='binding', help='')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--check_se3_invariance', type=bool, default=False, help='check it instead of generating files')
    p.add_argument('--num_confs', type=int, default=1, help='num_confs if using rdkit conformers')
    p.add_argument('--use_rdkit_coords', type=bool, default=None,
                   help='override the rkdit usage behavior of the used model')

    return p.parse_args()

def screen_ligand(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = None
    all_ligs_coords_corrected = []
    all_intersection_losses = []
    all_intersection_losses_untuned = []
    all_ligs_coords_pred_untuned = []
    all_ligs_coords = []
    all_ligs_keypts = []
    all_recs_keypts = []
    all_names = []
    dp = args.dataset_params
    use_rdkit_coords = args.use_rdkit_coords if args.use_rdkit_coords != None else args.dataset_params[
        'use_rdkit_coords']

    rescores = []

    #get screening dataset
    with open(args.protein_set) as file:
        lines = file.readlines()
        protein_set = [line.rstrip() for line in lines]

    #loop through pdbbind dataset and predict bindings
    for complex in protein_set:
        #preprocess ligand
        lig_path = os.path.join('data/PDBBind', complex, f'{complex}_ligand.sdf')
        if not os.path.exists(lig_path):
            raise ValueError(f'Path does not exist: {lig_path}')
        print(f'Trying to load {lig_path}')
        lig = read_molecule(lig_path, sanitize=True)
        if lig == None:
            lig = read_molecule(os.path.join('data/PDBBind', complex, f'{complex}_ligand.mol2'), sanitize=True)
        if lig != None:  # read mol2 file if sdf file cannot be sanitized
            used_lig = lig_path
        if lig == None: raise ValueError(f'The ligand file could not be read')
        lig_graph = get_lig_graph_revised(lig, lig_path, max_neighbors=dp['lig_max_neighbors'],
                                                use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
        if 'geometry_regularization' in dp and dp['geometry_regularization']:
            geometry_graph = get_geometry_graph(lig)
        elif 'geometry_regularization_ring' in dp and dp['geometry_regularization_ring']:
            geometry_graph = get_geometry_graph_ring(lig)
        else:
            geometry_graph = None
        start_lig_coords = lig_graph.ndata['x']

        rec_path = os.path.join('data/PDBBind', complex, f'{complex}_protein_processed.pdb')
        if (not os.path.exists(rec_path)):
            print(f'Protein at {rec_path} does not exist')
            continue
        print(f'Docking the receptor {complex}\nTo the ligand {used_lig}')
        rec, rec_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(rec_path)
        rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                  use_rec_atoms=dp['use_rec_atoms'], rec_radius=dp['rec_graph_radius'],
                                  surface_max_neighbors=dp['surface_max_neighbors'],
                                  surface_graph_cutoff=dp['surface_graph_cutoff'],
                                  surface_mesh_cutoff=dp['surface_mesh_cutoff'],
                                  c_alpha_max_neighbors=dp['c_alpha_max_neighbors'])

        # Randomly rotate and translate the ligand.
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        if (use_rdkit_coords):
            lig_coords_to_move = lig_graph.ndata['new_x']
        else:
            lig_coords_to_move = lig_graph.ndata['x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        input_coords = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        lig_graph.ndata['new_x'] = input_coords

        if model == None:
            model = load_model(args, data_sample=(lig_graph, rec_graph), device=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

        with torch.no_grad():
            geometry_graph = geometry_graph.to(device) if geometry_graph != None else None
            ligs_coords_pred_untuned, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss = model(
                deepcopy(lig_graph.to(device)), rec_graph.to(device), geometry_graph, complex_names=[args.ligand_to_screen], epoch=0)

            for lig_coords_pred_untuned, lig_coords, lig_keypts, rec_keypts, rotation, translation in zip(
                    ligs_coords_pred_untuned, [start_lig_coords], ligs_keypts, recs_keypts, rotations,
                    translations, ):
                all_intersection_losses_untuned.append(
                    compute_revised_intersection_loss(lig_coords_pred_untuned.detach().cpu(), rec_graph.ndata['x'],
                                                      alpha=0.2, beta=8, aggression=0))
                all_ligs_coords_pred_untuned.append(lig_coords_pred_untuned.detach().cpu())
                all_ligs_coords.append(lig_coords.detach().cpu())
                all_ligs_keypts.append(((rotation @ (lig_keypts).T).T + translation).detach().cpu())
                all_recs_keypts.append(rec_keypts.detach().cpu())

            if args.run_corrections:
                prediction = ligs_coords_pred_untuned[0].detach().cpu()
                lig_input = deepcopy(lig)
                conf = lig_input.GetConformer()
                for i in range(lig_input.GetNumAtoms()):
                    x, y, z = input_coords.numpy()[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

                lig_equibind = deepcopy(lig)
                conf = lig_equibind.GetConformer()
                for i in range(lig_equibind.GetNumAtoms()):
                    x, y, z = prediction.numpy()[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

                coords_pred = lig_equibind.GetConformer().GetPositions()

                Z_pt_cloud = coords_pred
                rotable_bonds = get_torsions([lig_input])
                new_dihedrals = np.zeros(len(rotable_bonds))
                for idx, r in enumerate(rotable_bonds):
                    new_dihedrals[idx] = get_dihedral_vonMises(lig_input, lig_input.GetConformer(), r, Z_pt_cloud)
                optimized_mol = apply_changes(lig_input, new_dihedrals, rotable_bonds)

                coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
                R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
                coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()
                all_ligs_coords_corrected.append(coords_pred_optimized)

                if args.output_directory:
                    if not os.path.exists(f'{args.output_directory}/screen/{complex}'):
                        os.makedirs(f'{args.output_directory}/screen/{complex}')
                    conf = optimized_mol.GetConformer()
                    for i in range(optimized_mol.GetNumAtoms()):
                        x, y, z = coords_pred_optimized[i]
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                    prediction_path = f'{args.output_directory}/screen/{complex}/{complex}_lig_equibind_corrected.sdf'

                    block_optimized = Chem.MolToMolBlock(optimized_mol)
                    print(f'Writing prediction to {args.output_directory}/screen/{complex}/{complex}_lig_equibind_corrected.sdf')
                    with open(prediction_path, "w") as newfile:
                        newfile.write(block_optimized)
                    if (args.mode != 'n'):
                        obConversion  = openbabel.OBConversion()
                        obConversion.SetInAndOutFormats('sdf', 'pdbqt')
                        mol = openbabel.OBMol()
                        obConversion.ReadFile(mol, prediction_path)
                        pdbqt_file = f'{args.output_directory}/screen/{complex}/{complex}_lig_equibind_corrected_ad.pdbqt'
                        obConversion.WriteFile(mol, pdbqt_file)
                        autodock_out_path = f'{args.output_directory}/screen/{complex}/{complex}_lig_autodock_corrected.pdbqt'

                        search_mins = np.min(coords_pred_optimized, axis=0)
                        search_maxes = np.max(coords_pred_optimized, axis=0)
                        search_dims = np.add(np.divide(np.subtract(search_maxes, search_mins), 2), [5, 5, 5])
                        search_center = np.add(search_mins, search_dims)

                        #correct with qvina2/smina if mode
                        if (args.mode == 'q'):
                            os.system(f'./qvina2.1 --receptor {rec_path} --ligand {pdbqt_file} --center_x {search_center[0]} --center_y {search_center[1]} --center_z {search_center[2]} --size_x {search_dims[0]} --size_y {search_dims[1]} --size_z {search_dims[2]} --out {autodock_out_path}')
                        elif (args.mode == 's'):
                            os.system(f'./smina.static --receptor {rec_path} --ligand {pdbqt_file} --center_x {search_center[0]} --center_y {search_center[1]} --center_z {search_center[2]} --size_x {search_dims[0]} --size_y {search_dims[1]} --size_z {search_dims[2]} --out {autodock_out_path}')

                        obConversion.SetInAndOutFormats('pdbqt', 'sdf')
                        obConversion.ReadFile(mol, autodock_out_path)
                        prediction_path = f'{args.output_directory}/screen/{complex}/{complex}_lig_autodock_corrected.sdf'
                        obConversion.WriteFile(mol, prediction_path)
                
                
                # rescore docking pose and record
                # get receptor as graph mol(no receptor data)
                receptor = oechem.OEGraphMol()
                rec_istream = oechem.oemolistream(rec_path)
                if not oechem.OEReadPDBFile(rec_istream, receptor):
                    oechem.OEThrow.Fatal("Unable to read receptor")
                # get lig as graph mol
                lig_mol = oechem.OEGraphMol()
                istream = oechem.oemolistream(prediction_path)
                if not oechem.OEReadMolecule(istream, lig_mol):
                    oechem.OEThrow.Fatal("Unable to read ligand")
                # find bounding box +-some flexibility around ligand
                lig_coords_list = [[x[0], x[1], x[2]] for x in lig_mol.GetCoords().values()]
                mins = np.min(lig_coords_list, axis=0)
                mins -= 0 # angstrom flexibility in bounding box (apparently does not change anything)
                maxes = np.max(lig_coords_list, axis=0)
                maxes += 0
                bounding_box = oedocking.OEBox(*mins, *maxes)
                # initialize score obj 
                score = oedocking.OEScore()
                score.Initialize(receptor, bounding_box)
                rescore = score.ScoreLigand(lig_mol)
                rescores.append(rescore)
                print(rescore)

            f = open('qvina2test.txt', 'w')
            for score in rescores:
                f.write(str(score))
                f.write('\n')
            f.close()
            all_names.append(args.ligand_to_screen)

    path = os.path.join(os.path.dirname(args.checkpoint), f'predictions_RDKit{use_rdkit_coords}.pt')
    print(f'Saving predictions to {path}')
    results = {'corrected_predictions': all_ligs_coords_corrected, 'initial_predictions': all_ligs_coords_pred_untuned,
               'targets': all_ligs_coords, 'lig_keypts': all_ligs_keypts, 'rec_keypts': all_recs_keypts,
               'names': all_names, 'intersection_losses_untuned': all_intersection_losses_untuned,
               'intersection_losses': all_intersection_losses, 'scores': rescores}
    torch.save(results, path)
 


if __name__ == '__main__':
    args = parse_arguments()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    for run_dir in args.run_dirs:
        args.checkpoint = f'runs/{run_dir}/best_checkpoint.pt'
        config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
        # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        args.model_parameters['noise_initial'] = 0
        screen_ligand(args)
