import os
import random
from copy import deepcopy, copy

from dgl import save_graphs, load_graphs

from joblib import Parallel, delayed, cpu_count
import torch
import dgl
from biopandas.pdb import PandasPdb
from joblib.externals.loky import get_reusable_executor

from rdkit import Chem
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from commons.geometry_utils import random_rotation_translation, rigid_transform_Kabsch_3D_torch
from commons.process_mols import get_rdkit_coords, get_receptor, get_pocket_coords, \
    read_molecule, get_rec_graph, get_lig_graph_revised, get_receptor_atom_subgraph, get_lig_structure_graph, \
    get_geometry_graph, get_lig_graph_multiple_conformer, get_geometry_graph_ring
from commons.utils import pmap_multi, read_strings_from_txt, log


class PDBBind_Lazy(Dataset):
    """"""

    def __init__(self, device='cuda:0',
                 complex_names_path='data/',
                 bsp_proteins=False,
                 bsp_ligands=False,
                 pocket_cutoff=8.0,
                 use_rec_atoms=False,
                 n_jobs=None,
                 chain_radius=7,
                 c_alpha_max_neighbors=10,
                 lig_max_neighbors=20,
                 translation_distance=5.0,
                 lig_graph_radius=30,
                 rec_graph_radius=30,
                 surface_max_neighbors=5,
                 surface_graph_cutoff=5,
                 surface_mesh_cutoff=1.7,
                 deep_bsp_preprocessing=True,
                 only_polar_hydrogens=False,
                 use_rdkit_coords=False,
                 pocket_mode='match_terminal_atoms',
                 dataset_size=None,
                 remove_h=False,
                 rec_subgraph=False,
                 is_train_data=False,
                 min_shell_thickness=2,
                 subgraph_radius=10,
                 subgraph_max_neigbor=8,
                 subgraph_cutoff=4,
                 lig_structure_graph= False,
                 random_rec_atom_subgraph= False,
                 subgraph_augmentation=False,
                 lig_predictions_name=None,
                 geometry_regularization= False,
                 multiple_rdkit_conformers = False,
                 random_rec_atom_subgraph_radius= 10,
                 geometry_regularization_ring= False,
                 num_confs=10,
                 transform=None, **kwargs):
        # subset name is either 'pdbbind_filtered' or 'casf_test'
        self.chain_radius = chain_radius
        self.pdbbind_dir = 'data/PDBBind'
        self.bsp_dir = 'data/deepBSP'
        self.only_polar_hydrogens = only_polar_hydrogens
        self.complex_names_path = complex_names_path
        self.pocket_cutoff = pocket_cutoff
        self.use_rec_atoms = use_rec_atoms
        self.deep_bsp_preprocessing = deep_bsp_preprocessing
        self.device = device
        self.lig_graph_radius = lig_graph_radius
        self.rec_graph_radius = rec_graph_radius
        self.surface_max_neighbors = surface_max_neighbors
        self.surface_graph_cutoff = surface_graph_cutoff
        self.surface_mesh_cutoff = surface_mesh_cutoff
        self.dataset_size = dataset_size
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.lig_max_neighbors = lig_max_neighbors
        self.n_jobs = cpu_count() - 1 if n_jobs == None else n_jobs
        self.translation_distance = translation_distance
        self.pocket_mode = pocket_mode
        self.use_rdkit_coords = use_rdkit_coords
        self.bsp_proteins = bsp_proteins
        self.bsp_ligands = bsp_ligands
        self.remove_h = remove_h
        self.is_train_data = is_train_data
        self.subgraph_augmentation = subgraph_augmentation
        self.min_shell_thickness = min_shell_thickness
        self.rec_subgraph = rec_subgraph
        self.subgraph_radius = subgraph_radius
        self.subgraph_max_neigbor=subgraph_max_neigbor
        self.subgraph_cutoff=subgraph_cutoff
        self.random_rec_atom_subgraph = random_rec_atom_subgraph
        self.lig_structure_graph =lig_structure_graph
        self.random_rec_atom_subgraph_radius = random_rec_atom_subgraph_radius
        self.lig_predictions_name = lig_predictions_name
        self.geometry_regularization = geometry_regularization
        self.geometry_regularization_ring = geometry_regularization_ring
        self.multiple_rdkit_conformers = multiple_rdkit_conformers
        self.num_confs = num_confs
        self.conformer_id = 0

        self.processed_dir = f'data/processed/size{self.dataset_size}_INDEX{os.path.splitext(os.path.basename(self.complex_names_path))[0]}_Hpolar{int(self.only_polar_hydrogens)}_H{int(not self.remove_h)}_BSPprot{int(self.bsp_proteins)}_BSPlig{int(self.bsp_ligands)}_surface{int(self.use_rec_atoms)}_pocketRad{self.pocket_cutoff}_ligRad{self.lig_graph_radius}_recRad{self.rec_graph_radius}_recMax{self.c_alpha_max_neighbors}_ligMax{self.lig_max_neighbors}_chain{self.chain_radius}_POCKET{self.pocket_mode}'
        print(f'using processed directory: {self.processed_dir}')
        if not os.path.exists('data/processed/'):
            os.mkdir('data/processed/')
        self.complex_names = read_strings_from_txt(self.complex_names_path)
        # remove previously errored complexes
        f = open('errored_complexes.txt', 'r')
        error_set = f.read().strip().split('\n')
        for l in error_set:
            print(l)
            if l in self.complex_names:
                self.complex_names.remove(l)
        self.cache = {}


    def __len__(self):
        return len(self.complex_names)

    def __getitem__(self, idx):
        print(idx)
        print(len(self.complex_names))
        complex = self.complex_names[idx]
        # read from file if exists, otherwise process first
        if (not os.path.exists(f'{complex}_pocket_and_rec_coords.pt') 
        or (not os.path.exists(f'{complex}_lig_structure_graphs.pt') and self.lig_structure_graph) 
        or (not os.path.exists(f'{complex}_lig_graph.pt')) 
        or (not os.path.exists(f'{complex}_rec_graphs.pt')) 
        or (self.rec_subgraph and not os.path.exists(f'{complex}_rec_subgraph.pt')) 
        or (self.lig_structure_graph and not os.path.exists(f'{complex}_torsion_masks_and_angles.pt')) 
        or (self.geometry_regularization and not os.path.exists(f'{complex}_geometry_regularization.pt')) 
        or (self.geometry_regularization_ring and not os.path.exists(f'{complex}_geometry_regularization_ring.pt'))):
            while not self.process(complex): # if it fails we delete the lig from set and process a new one at that index
                print(f'{complex} errored')
                self.complex_names.remove(complex)
                if (idx >= len(self.complex_names)):
                    return
                print(idx)
                print(len(self.complex_names))
                complex = self.complex_names[idx] #reindex after deletion

        pocket_coords = torch.load(os.path.join(self.processed_dir, f'{complex}_pocket_and_rec_coords.pt'))['pockets_coords']
        if self.lig_structure_graph:
            lig_graph, _ = load_graphs(os.path.join(self.processed_dir, f'{complex}_lig_structure_graphs.pt'))
            lig_graph = lig_graph[0]
        else:
            if self.multiple_rdkit_conformers:
                lig_graph, _ = load_graphs(os.path.join(self.processed_dir, f'{complex}_lig_graph.pt'))
                lig_graph = lig_graph[0][self.conformer_id]
            else:
                lig_graph, _ = load_graphs(os.path.join(self.processed_dir, f'{complex}_lig_graph.pt'))
                lig_graph = lig_graph[0]
        lig_coords = lig_graph.ndata['x']
        rec_graph, _ = load_graphs(os.path.join(self.processed_dir, f'{complex}_rec_graphs.pt'))
        rec_graph = rec_graph[0]
        if self.rec_subgraph:
            rec_subgraph, _ = load_graphs(os.path.join(self.processed_dir, f'{complex}_rec_subgraph.pt'))
            rec_subgraph = rec_subgraph[0]
        if self.lig_structure_graph:
            masks_angles = torch.load(os.path.join(self.processed_dir, f'{complex}_torsion_masks_and_angles.pt'))
            angles = masks_angles['angles']
            masks = masks_angles['masks']
        if self.geometry_regularization:
            print(os.path.join(self.processed_dir, 'geometry_regularization.pt'))
            geometry_graph, _ =  load_graphs(os.path.join(self.processed_dir, f'{complex}_geometry_regularization.pt'))
            geometry_graph = geometry_graph[0]
        if self.geometry_regularization_ring:
            print(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt'))
            geometry_graph, _ =  load_graphs(os.path.join(self.processed_dir, f'{complex}_geometry_regularization_ring.pt'))
            geometry_graph = geometry_graph[0]


        # Randomly rotate and translate the ligand.
        rot_T, rot_b = random_rotation_translation(translation_distance=self.translation_distance)
        if self.use_rdkit_coords:
            lig_coords_to_move =lig_graph.ndata['new_x']
        else:
            lig_coords_to_move = lig_coords
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        lig_graph.ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        new_pocket_coords = (rot_T @ (pocket_coords - mean_to_remove).T).T + rot_b

        if self.subgraph_augmentation and self.is_train_data:
            with torch.no_grad():
                if idx in self.cache:
                    max_distance, min_distance, distances = self.cache[idx]
                else:
                    lig_centroid = lig_graph.ndata['x'].mean(dim=0)
                    distances = torch.norm(rec_graph.ndata['x'] - lig_centroid, dim=1)
                    max_distance = torch.max(distances)
                    min_distance = torch.min(distances)
                    self.cache[idx] = (min_distance.item(), max_distance.item(), distances)
                radius = min_distance + self.min_shell_thickness + random.random() * abs((
                            max_distance - min_distance - self.min_shell_thickness))
                rec_graph = dgl.node_subgraph(rec_graph, distances <= radius)
                assert rec_graph.num_nodes() > 0
        if self.rec_subgraph:
            rec_graph = rec_subgraph
            if self.random_rec_atom_subgraph:
                rot_T, rot_b = random_rotation_translation(translation_distance=2)
                translated_lig_coords = lig_coords + rot_b
                min_distances, _ = torch.cdist(rec_graph.ndata['x'],translated_lig_coords).min(dim=1)
                rec_graph = dgl.node_subgraph(rec_graph, min_distances < self.random_rec_atom_subgraph_radius)
                assert rec_graph.num_nodes() > 0

        geometry_graph = geometry_graph if self.geometry_regularization or self.geometry_regularization_ring else None
        if self.lig_structure_graph:
            return lig_graph.to(self.device), rec_graph.to(self.device), masks, angles, lig_coords, rec_graph.ndata['x'], new_pocket_coords, pocket_coords,geometry_graph, self.complex_names[idx], idx
        else:
            return lig_graph.to(self.device), rec_graph.to(self.device), lig_coords, rec_graph.ndata['x'], new_pocket_coords, pocket_coords, geometry_graph, self.complex_names[idx], idx

    def process(self, name):
        # processes a single complex into file
        log(f'Processing complex {name} and saving it to [{self.processed_dir}]')
        # process lig
        lig = read_molecule(os.path.join(self.pdbbind_dir, name, f'{name}_ligand.sdf'), sanitize=True,
                                    remove_hs=self.remove_h)
        if lig == None:  # read mol2 file if sdf file cannot be sanitized
            lig = read_molecule(os.path.join(self.pdbbind_dir, name, f'{name}_ligand.mol2'), sanitize=True,
                                remove_hs=self.remove_h)
        if self.only_polar_hydrogens:
            for atom in lig.GetAtoms():
                if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                    atom.SetAtomicNum(0)
            lig = Chem.DeleteSubstructs(lig, Chem.MolFromSmarts('[#0]'))
            Chem.SanitizeMol(lig)
        if (lig == None): # errored mol, remove from set
            print(f'ERRORED COMPLEX: {name}')
            if (not os.path.exists('errored_complexes.txt')):
                f = open('errored_complexes.txt', 'w')
                f.close()
            f = open('errored_complexes.txt', 'a')
            f.write(f'{name}\n')
            return False
        
        rec_path = os.path.join(self.pdbbind_dir, name, f'{name}_protein_processed.pdb')

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

        # process rec
        receptor_representatives = get_receptor(rec_path, lig, cutoff=self.chain_radius)
        rec, rec_coords, c_alpha_coords, n_coords, c_coords = receptor_representatives
        pockets_coords = get_pocket_coords(lig, rec_coords, cutoff=self.pocket_cutoff, pocket_mode=self.pocket_mode)
        rec_coords_tensor = torch.tensor(np.concatenate(rec_coords, axis=0))
        torch.save({'pockets_coords': pockets_coords,
                        'all_rec_coords': rec_coords_tensor,
                        # coords of all atoms and not only those included in graph
                        'complex_name': name,
                        }, os.path.join(self.processed_dir, f'{name}_pocket_and_rec_coords.pt'))

        rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, 
                                    use_rec_atoms=self.use_rec_atoms, 
                                    rec_radius=self.rec_graph_radius,
                                    surface_max_neighbors=self.surface_max_neighbors,
                                    surface_graph_cutoff=self.surface_graph_cutoff,
                                    surface_mesh_cutoff=self.surface_mesh_cutoff,
                                    c_alpha_max_neighbors=self.c_alpha_max_neighbors,)
        save_graphs(os.path.join(self.processed_dir, f'{name}_rec_graphs.pt'), rec_graph)

        if (self.rec_subgraph):
            rec_subgraph = get_receptor_atom_subgraph(rec, rec_coords, lig, max_neighbor=self.subgraph_max_neigbor, subgraph_radius=self.subgraph_radius,
                                        graph_cutoff=self.subgraph_cutoff,)
            save_graphs(os.path.join(self.processed_dir, f'{name}_rec_subgraph.pt'), rec_subgraph)

        if self.multiple_rdkit_conformers:
            lig_graph = get_lig_graph_multiple_conformer(lig, name, max_neighbors=self.lig_max_neighbors, use_rdkit_coords=self.use_rdkit_coords,
                                radius=self.lig_graph_radius, num_confs=self.num_confs)
        else:
            lig_graph = get_lig_graph_revised(lig, name, max_neighbors=self.lig_max_neighbors, use_rdkit_coords=self.use_rdkit_coords,
                                radius=self.lig_graph_radius)

        save_graphs(os.path.join(self.processed_dir, f'{name}_lig_graph.pt'), lig_graph)

        if (self.lig_structure_graph):
            graphs_masks_angles = get_lig_structure_graph(lig)
            graphs, masks, angles = graphs_masks_angles
            torch.save({'masks': masks,
                        'angles': angles,
                        }, os.path.join(self.processed_dir, f'{name}_torsion_masks_and_angles.pt'))
            save_graphs(os.path.join(self.processed_dir, f'{name}_lig_structure_graph.pt'), graphs)

        try: 
            geometry_graph = get_geometry_graph(lig)
            save_graphs(os.path.join(self.processed_dir, f'{name}_geometry_regularization.pt'), geometry_graph)

            geometry_graph = get_geometry_graph_ring(lig)
            save_graphs(os.path.join(self.processed_dir, f'{name}_geometry_regularization_ring.pt'), geometry_graph)
        except:
            print(f'complex {name} errors on rdkit conformer generation')
            if (not os.path.exists('errored_complexes.txt')):
                f = open('errored_complexes.txt', 'w')
                f.close()
            f = open('errored_complexes.txt', 'a')
            f.write(f'{name}\n')
            return False
        return True