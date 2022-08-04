# Authors: Murat Cihan Sorkun <mcsorkun@gmail.com>, Dajt Mullaj <dajt.mullai@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import umap
import base64
import functools

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import sklearn.cluster as cst

from pandas.api.types import is_numeric_dtype

from rdkit import Chem
from openeye import oechem
from rdkit.Chem import Draw
from bokeh.plotting import figure
from bokeh.transform import transform, factor_cmap
from bokeh.palettes import Category10, Inferno, Spectral4
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import ColorBar, HoverTool, Panel, Tabs
from bokeh.io import output_file, save, show
from scipy import stats
from io import BytesIO

import chemplot.descriptors as desc
import chemplot.parameters as parameters

import prolif as plf
import oddt
from oddt.fingerprints import PLEC

from faerun import Faerun
from faerun import host
from cmcrameri import cm

def calltracker(func):
    @functools.wraps(func)
    def wrapper(*args):
        wrapper.has_been_called = True
        return func(*args)
    wrapper.has_been_called = False
    return wrapper

class Plotter(object):
    """
    A class used to plot the ECFP fingerprints of the molecules used to
    instantiate it.

    :param __sim_type: similarity type structural or tailored
    :param __target_type: target type R (regression) or C (classificatino)
    :param __target: list containing the target values. Is empty if a target does not exist
    :param __mols: list of valid molecules that can be plotted
    :param __df_descriptors: datatframe containing the descriptors representation of each molecule
    :param __df_2_components: dataframe containing the two-dimenstional representation of each molecule
    :param __plot_title: title of the plot reflecting the dimensionality reduction algorithm used
    :param __data: list of the scaled descriptors to which the dimensionality reduction algorithm is applied
    :param pca_fit: PCA object created when the corresponding algorithm is applied to the data
    :param tsne_fit: t-SNE object created when the corresponding algorithm is applied to the data
    :param umap_fit: UMAP object created when the corresponding algorithm is applied to the data
    :param df_plot_xy: dataframe containing the coordinates that have been plotted
    :type __sim_type: string
    :type __target_type: string
    :type __target: list
    :type __mols: rdkit.Chem.rdchem.Mol
    :type __df_descriptors: Dataframe
    :type __df_2_components: Dataframe
    :type __plot_title: string
    :type __data: list
    :type pca_fit: sklearn.decomposition.TSNE
    :type tsne_fit: sklearn.manifold.TSNE
    :type umap_fit: umap.umap_.UMAP
    :type df_plot_xy: Dataframe
    """

    # _static_plots = {'scatter', 'hex', 'kde'}
    #
    # _interactive_plots = {'scatter', 'hex'}

    _sim_types = {'tailored', 'ecfp', 'prolif', 'plec'}

    _target_types = {'R', 'C'}


    def __init__(self, data, target, target_type, sim_type,
        get_desc=None,
        get_ecfp_fingerprints=None,
        get_plec_fingerprints=None,
        get_prolif_fingerprints=None, **kwargs):

        cutoff_score = kwargs['cutoff_score'] if 'cutoff_score' in kwargs else None

        # Instantiate Plotter class
        self.__target_type = target_type
        self.__sim_type = sim_type
        self.__mols = None

        if len(target) > 0:
            df_target = pd.DataFrame(data=target)
            unique_targets_ratio = 1.*df_target.iloc[:, 0].nunique()/df_target.iloc[:, 0].count() < 0.05
            numeric_target = is_numeric_dtype(df_target.dtypes[0])
            if self.__target_type == 'R' and (unique_targets_ratio or not numeric_target):
                print('Input received is \'R\' for target values that seem not continuous.')
            if self.__target_type not in self._target_types:
                if not unique_targets_ratio and numeric_target:
                    self.__target_type = 'R'
                    print('target_type indicates if the target is a continuous variable or a class label.\n'+
                          'R stands for regression and C for classification. Input R as target type for continuous variables and C for class labels.\n'+
                          'From analysis of the target, R has been selected for target_type.')
                else:
                    self.__target_type = 'C'
                    print('target_type indicates if the target is a continuous variable or a class label.\n'+
                          'R stands for regression and C for classification. Input R as target type for continuous variables and C for class labels.\n'+
                          'From analysis of the target, C has been selected for target_type.')
        else:
            self.__target_type = None

        if len(target) > 0 and self.__target_type == 'C':
            df_target = pd.DataFrame(data=target)
            if df_target.iloc[:, 0].nunique() == 1:
                target = []
                self.__sim_type = "ecfp"
                print("Only one class found in the targets")

        if len(target) > 0 and cutoff_score is not None:
            target = [score if score < cutoff_score else 0 for score in target]

        if self.__sim_type == "tailored":

            if get_desc is None:
                raise Exception("get_desc must be provided for a tailored sim")

            _, df_descriptors, target = get_desc(data, target)
            if df_descriptors.empty:
                raise Exception("Descriptors could not be computed for given molecules")

            self.__df_descriptors, self.__target = desc.select_descriptors_lasso(df_descriptors, target, kind=self.__target_type)
        elif self.__sim_type == "ecfp":

            if get_ecfp_fingerprints is None:
                raise Exception("get_ecfp_fingerprints must be provided for an ecfp sim")

            _, self.__df_descriptors, self.__target = get_ecfp_fingerprints(data, target, **kwargs)
        elif self.__sim_type == "plec":

            if get_plec_fingerprints is None:
                raise Exception("get_plec_fingerprints must be provided for a plec sim")

            _, self.__df_descriptors, self.__target = get_plec_fingerprints(data, target, **kwargs)
        elif self.__sim_type == "prolif":
            if get_prolif_fingerprints is None:
                raise Exception("get_prolif_fingerprints must be provided for a prolif sim")

            _, self.__df_descriptors, self.__target = get_prolif_fingerprints(data, target, **kwargs)
        else:
            raise Exception("Invalid sim_type")



        if len(self.__df_descriptors) < 2 or len(self.__df_descriptors.columns) < 2:
            raise Exception("Plotter object cannot be instantiated for given molecules")



        self.__df_2_components = None
        self.__plot_title = None

        self.__test_target = None
        self.__test_mols = None
        self.__test_df_descriptors = None
        self.__test_df_2_components = None

    @classmethod
    def from_smiles(cls, smiles_list, target=[], target_type=None, sim_type='ecfp', **kwargs):
        """
        Class method to construct a Plotter object from a list of SMILES.

        :param smile_list: List of the SMILES representation of the molecules to plot.
        :param target: target values
        :param target_type: target type R (regression) or C (classificatino)
        :param sim_type: similarity type structural or tailored
        :type smile_list: list
        :type target: list
        :type target_type: string
        :type sim_type: string
        :returns: A Plotter object for the molecules given as input.
        :rtype: Plotter
        """

        if sim_type not in {'tailored', 'ecfp'}:
            raise Exception("Invalid sim_type")

        return cls(smiles_list, target, target_type, sim_type,
            get_desc=desc.get_mordred_descriptors_from_smiles,
            get_ecfp_fingerprints=desc.get_ecfp_from_smiles, **kwargs)


    @classmethod
    def from_inchi(cls, inchi_list, target=[], target_type=None, sim_type='ecfp', **kwargs):
        """
        Class method to construct a Plotter object from a list of InChi.

        :param inchi_list: List of the InChi representation of the molecules to plot.
        :type inchi_list: dict
        :param target: target values
        :type target: dict
        :param target_type: target type R (regression) or C (classificatino)
        :type target_type: string
        :param sim_type: similarity type structural or tailored
        :type sim_type: string
        :returns: A Plotter object for the molecules given as input.
        :rtype: Plotter
        """

        if sim_type not in {'tailored', 'ecfp'}:
            raise Exception("Invalid sim_type")

        return cls(inchi_list, target, target_type, sim_type,
            get_desc=desc.get_mordred_descriptors_from_inchi,
            get_ecfp_fingerprints=desc.get_ecfp_from_inchi, **kwargs)

    @classmethod
    def from_sdf(cls, data_path, target=[], target_type=None, sim_type='plec', **kwargs):
        """
        Class method to construct a Plotter object path with a protein pdb file and many ligand sdf files.
        Uses PLEC fingerprints as features

        :param data_path: folder containing the pdb and sdf files
        :type data_path: string
        :returns: A Plotter object for the molecules given as input.
        :rtype: Plotter
        """

        if sim_type not in {'ecfp', 'plec', 'prolif'}:
            raise Exception("Invalid sim_type")

        # get_desc is None since only structural
        return cls(data_path, target, target_type, sim_type,
            get_ecfp_fingerprints=desc.get_ecfp_from_sdf,
            get_plec_fingerprints=desc.get_plec_from_sdf,
            get_prolif_fingerprints=desc.get_prolif_from_sdf, **kwargs)

    @classmethod
    def from_sdf_dir(cls, data_path, target=[], target_type=None, sim_type='plec', **kwargs):
        """
        Class method to construct a Plotter object path with a data dir(pdb sdf pairs)

        :param data_path: folder containing the subdirs of pdb and sdf files
        :type data_path: string
        :returns: A Plotter object for the molecules given as input.
        :rtype: Plotter
        """

        if sim_type not in {'ecfp', 'plec', 'prolif'}:
            raise Exception("Invalid sim_type")

        # get_desc is None since only structural
        return cls(data_path, target, target_type, sim_type,
            get_ecfp_fingerprints=desc.get_ecfp_from_sdf_subdir,
            get_plec_fingerprints=desc.get_plec_from_sdf_subdir,
            get_prolif_fingerprints=desc.get_prolif_from_sdf_subdir, **kwargs)

    @classmethod
    def from_oechem(cls, lig_path, rec_path=None, target=[], target_type=None, sim_type='plec', **kwargs):
        """
        Class method to construct a Plotter object path with a protein pdb file and a ligand oeb.gz file
        Uses PLEC fingerprints as features

        :param rec_path: path to protein pdb
        :type rec_path: string
        :param lig_path: path to ligand oeb.gz
        :type rec_path: string
        :returns: A Plotter object for the molecules given as input.
        :rtype: Plotter
        """

        if sim_type not in {'ecfp', 'plec', 'prolif', 'tailored'}:
            raise Exception("Invalid sim_type")

        if (sim_type == 'plec' or sim_type == 'prolif') and rec_path is None:
            raise Exception(f'{sim_type} fingerprints require receptor')

        if len(target) == 0:
            ligands = oechem.oemolistream(lig_path)
            target = [mol.GetEnergy() for mol in ligands.GetOEGraphMols()]

        # get_desc is None since only structural
        return cls((rec_path, lig_path), target, target_type, sim_type,
            get_desc=desc.get_mordred_descriptors_from_oechem,
            get_ecfp_fingerprints=desc.get_ecfp_from_oechem,
            get_plec_fingerprints=desc.get_plec_from_oechem,
            get_prolif_fingerprints=desc.get_prolif_from_oechem, **kwargs)

    @classmethod
    def from_dict(cls, dict, target=[], target_type=None, sim_type='ecfp'):
         self = cls.__new__(cls)

         # Instantiate Plotter class
         self.__target_type = target_type
         self.__sim_type = sim_type
         self.__mols = None

         self.__df_descriptors = dict
         self.__target = target

         if len(self.__target) > 0:
             df_target = pd.DataFrame(data=self.__target)
             unique_targets_ratio = 1.*df_target.iloc[:, 0].nunique()/df_target.iloc[:, 0].count() < 0.05
             numeric_target = is_numeric_dtype(df_target.dtypes[0])
             if self.__target_type == 'R' and (unique_targets_ratio or not numeric_target):
                 print('Input received is \'R\' for target values that seem not continuous.')
             if self.__target_type not in self._target_types:
                 if not unique_targets_ratio and numeric_target:
                     self.__target_type = 'R'
                     print('target_type indicates if the target is a continuous variable or a class label.\n'+
                           'R stands for regression and C for classification. Input R as target type for continuous variables and C for class labels.\n'+
                           'From analysis of the target, R has been selected for target_type.')
                 else:
                     self.__target_type = 'C'
                     print('target_type indicates if the target is a continuous variable or a class label.\n'+
                           'R stands for regression and C for classification. Input R as target type for continuous variables and C for class labels.\n'+
                           'From analysis of the target, C has been selected for target_type.')
         else:
             self.__target_type = None


         self.__df_2_components = None
         self.__plot_title = None

         self.__test_target = None
         self.__test_mols = None
         self.__test_df_descriptors = None
         self.__test_df_2_components = None

         return self

    def data_split(self, test_size=0.1, random_state=None):

        if self.__test_df_descriptors is not None:
            print('Data already split')
            return

        if len(self.__target) > 0:

            self.__df_descriptors, self.__test_df_descriptors, self.__target, self.__test_target = train_test_split(self.__df_descriptors, self.__target, test_size=test_size, random_state=random_state)
        else:
            self.__df_descriptors, self.__test_df_descriptors = train_test_split(self.__df_descriptors, test_size=test_size, random_state=random_state)

    def __add_test_data(self, test_data, test_target,
        get_desc=None,
        get_ecfp_fingerprints=None,
        get_plec_fingerprints=None,
        get_prolif_fingerprints=None):

        if self.__sim_type == "tailored":

            return

            if get_desc is None:
                raise Exception("get_desc must be provided for a tailored sim")

            _, test_df_descriptors, test_target = get_desc(test_data, test_target)
            if df_descriptors.empty:
                raise Exception("Descriptors could not be computed for given molecules")

            self.__test_df_descriptors = test_df_descriptors[self.__df_descriptors.columns]
            self.__target = test_target
        elif self.__sim_type == "ecfp":

            if get_ecfp_fingerprints is None:
                raise Exception("get_ecfp_fingerprints must be provided for an ecfp sim")

            _, self.__test_df_descriptors, self.__test_target = get_ecfp_fingerprints(test_data, test_target)
        elif self.__sim_type == "prolif":
            if get_prolif_fingerprints is None:
                raise Exception("get_prolif_fingerprints must be provided for a prolif sim")

            _, self.__test_df_descriptors, self.__test_target = get_prolif_fingerprints(test_data, test_target)
        else:
            raise Exception("Invalid sim_type")

        if len(self.__test_target) > 0 and len(self.__target) == 0:
            print("Test data cannot have target values if embedding does not")
            print("Dropping test target")
            self.__test_target = []

        if len(self.__test_target) == 0 and len(self.__target) > 0:
            print("Test data cannot have target values if embedding does not")
            print("Dropping embedding target")
            self.__target = []

    def add_test_data_from_smiles(self, smiles_list, target=[]):

        if self.__sim_type not in {'tailored', 'ecfp'}:
            raise Exception("Invalid sim_type for smiles data")

        self.__add_test_data(smiles_list, target,
            get_desc=desc.get_mordred_descriptors_from_smiles,
            get_ecfp_fingerprints=desc.get_ecfp_from_smiles)

    def add_test_data_from_inchi(self, inchi_list, target=[]):

        if self.__sim_type not in {'tailored', 'ecfp'}:
            raise Exception("Invalid sim_type for inchi data")

        self.__add_test_data(inchi_list, target,
            get_desc=desc.get_mordred_descriptors_from_inchi,
            get_ecfp_fingerprints=desc.get_ecfp_from_inchi)

    def add_test_data_from_sdf(self, data_path, target=[]):

        if self.__sim_type not in {'ecfp', 'plec', 'prolif'}:
            raise Exception("Invalid sim_type for sdf data")

        self.__add_test_data(data_path, target,
            get_ecfp_fingerprints=desc.get_ecfp_from_sdf,
            get_plec_fingerprints=desc.get_plec_from_sdf,
            get_prolif_fingerprints=desc.get_prolif_from_sdf)

    def add_test_data_from_oechem(self, lig_path, rec_path=None, target=[]):

        if self.__sim_type not in {'ecfp', 'plec', 'prolif', 'tailored'}:
            raise Exception("Invalid sim_type for oechem data")

        if (self.__sim_type == 'plec' or self.__sim_type == 'prolif') and rec_path is None:
            raise Exception(f'{sim_type} fingerprints require receptor')

        self.__add_test_data((rec_path, lig_path), target,
            get_desc=desc.get_mordred_descriptors_from_oechem,
            get_ecfp_fingerprints=desc.get_ecfp_from_oechem,
            get_plec_fingerprints=desc.get_plec_from_oechem,
            get_prolif_fingerprints=desc.get_prolif_from_oechem)

    # rm later
    def get_pca_var(self):
        return self.ev

    def pca(self, **kwargs):
        """
        Calculates the first 2 PCA components of the molecular descriptors.

        :param kwargs: Other keyword arguments are passed down to sklearn.decomposition.PCA
        :type kwargs: key, value mappings
        :returns: The dataframe containing the PCA components.
        :rtype: Dataframe
        """
        self.__data = self.__data_scaler(self.__df_descriptors)

        if self.__test_df_descriptors is not None:
            self.__test_data = self.__data_scaler(self.__test_df_descriptors)

        # Linear dimensionality reduction to 2 components by PCA
        self.pca_fit = PCA(**kwargs)
        pca_embedding = self.pca_fit.fit_transform(self.__data)

        if self.__test_df_descriptors is not None:
            test_pca_embedding = self.pca_fit.transform(self.__test_data)

        coverage_components = self.pca_fit.explained_variance_ratio_
        # rm later
        self.ev = coverage_components
        # Create labels for the plot
        first_component = "PC-1 (" + "{:.0%}".format(coverage_components[0]) + ")"
        second_component = "PC-2 (" + "{:.0%}".format(coverage_components[1]) + ")"
        # Create a dataframe containinting the first 2 PCA components of ECFP
        #HERE
        # self.__df_2_components = pd.DataFrame(data = pca_embedding
        #     , columns = [first_component, second_component])

        # if self.__test_df_descriptors is not None:
        #     self.__test_df_2_components = pd.DataFrame(data = test_pca_embedding
        #          , columns = [first_component, second_component])

        # self.__plot_title = "PCA plot"

        # if len(self.__target) > 0:
        #     self.__df_2_components['target'] = self.__target

        #     if self.__test_df_descriptors is not None:
        #         self.__test_df_2_components['target'] = self.__test_target


        # return self.__df_2_components.copy()


    def tsne(self, perplexity=None, pca=False, reduction=0.5, random_state=None, **kwargs):
        """
        Calculates the first 2 t-SNE components of the molecular descriptors.

        :param perplexity: perplexity value for the t-SNE model
        :param pca: indicates if the features must be preprocessed by PCA
        :param reduction: multiplied by numer of features to give target number of features
        :param random_state: random seed that can be passed as a parameter for reproducing the same results
        :param kwargs: Other keyword arguments are passed down to sklearn.manifold.TSNE
        :type perplexity: int
        :type pca: boolean
        :type reduction: float
        :type random_state: int
        :type kwargs: key, value mappings
        :returns: The dataframe containing the t-SNE components.
        :rtype: Dataframe
        """
        self.__data = self.__data_scaler(self.__df_descriptors)

        if self.__test_df_descriptors is not None:
            self.__test_data = self.__data_scaler(self.__test_df_descriptors)


        if pca and self.__sim_type != "tailored" and reduction > 0:
            n_components = round(len(self.__df_descriptors.columns) * reduction)

            if n_components > len(self.__df_descriptors):
                n_components = len(self.__df_descriptors)

            print(f'PCA Reducing from {len(self.__df_descriptors.columns)} to {n_components} features, {(n_components/len(self.__df_descriptors.columns) * 100):.2f}%')

            pca = PCA(n_components=n_components, random_state=random_state)
            self.__data = pca.fit_transform(self.__data)

            if self.__test_df_descriptors is not None:
                self.__test_data = pca.transform(self.__test_data)

            self.__plot_title = "t-SNE plot from components with cumulative variance explained " + "{:.0%}".format(sum(pca.explained_variance_ratio_))
        else:
            self.__plot_title = "t-SNE plot"

        # Get the perplexity of the model
        if perplexity is None:
            if self.__sim_type != "tailored":
                if pca:
                    perplexity = parameters.perplexity_structural_pca(len(self.__data))
                else:
                    perplexity = parameters.perplexity_structural(len(self.__data))
            else:
                perplexity = parameters.perplexity_tailored(len(self.__data))
        else:
            if perplexity<5 or perplexity>50:
                print('Robust results are obtained for values of perplexity between 5 and 50')

        # Embed the data in two dimensions
        self.tsne_fit = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, **kwargs)
        tsne_embedding = self.tsne_fit.fit_transform(self.__data)
        # Create a dataframe containinting the first 2 TSNE components of ECFP
        self.__df_2_components = pd.DataFrame(data = tsne_embedding
             , columns = ['t-SNE-1', 't-SNE-2'])

        if self.__test_df_descriptors is not None:
            test_tsne_embedding = self.tsne_fit.transform(self.__test_data)
            self.__test_df_2_components = pd.DataFrame(data=test_tsne_embedding
                 , columns=['UMAP-1', 'UMAP-2'])

        if len(self.__target) > 0:
            self.__df_2_components['target'] = self.__target

            if self.__test_df_descriptors is not None:
                self.__test_df_2_components['target'] = self.__test_target


        return self.__df_2_components.copy()


    def umap(self, n_neighbors=None, min_dist=None, pca=False, reduction=0.5, random_state=None, **kwargs):
        """
        Calculates the first 2 UMAP components of the molecular descriptors.

        :param num_neighbors: Number of neighbours used in the UMAP madel.
        :param min_dist: Value between 0.0 and 0.99, indicates how close to each other the points can be displayed.
        :param pca: indicates if the features must be preprocessed by PCA
        :param reduction: multiplied by numer of features to give target number of features
        :param random_state: random seed that can be passed as a parameter for reproducing the same results
        :param kwargs: Other keyword arguments are passed down to umap.UMAP
        :type num_neighbors: intl
        :type min_dist: float
        :type pca: boolean
        :type reduction: float
        :type random_state: int
        :type kwargs: key, value mappings
        :returns: The dataframe containing the UMAP components.
        :rtype: Dataframe
        """
        self.__data = self.__data_scaler(self.__df_descriptors)

        if self.__test_df_descriptors is not None:
            self.__test_data = self.__data_scaler(self.__test_df_descriptors)


        # Preprocess the data with PCA
        if pca and self.__sim_type != "tailored" and reduction > 0:
            n_components = round(len(self.__df_descriptors.columns) * reduction)

            if n_components > len(self.__df_descriptors):
                n_components = len(self.__df_descriptors)

            print(f'PCA Reducing from {len(self.__df_descriptors.columns)} to {n_components} features, {(n_components/len(self.__df_descriptors.columns) * 100):.2f}%')

            pca = PCA(n_components=n_components, random_state=random_state)
            self.__data = pca.fit_transform(self.__data)

            if self.__test_df_descriptors is not None:
                self.__test_data = pca.transform(self.__test_data)

            self.__plot_title = "UMAP plot from components with cumulative variance explained " + "{:.0%}".format(sum(pca.explained_variance_ratio_))
        else:
            self.__plot_title = "UMAP plot"

        if n_neighbors is None:
            if self.__sim_type != "tailored":
                if pca:
                    n_neighbors = parameters.n_neighbors_structural_pca(len(self.__data))
                else:
                    n_neighbors = parameters.n_neighbors_structural(len(self.__data))
            else:
                n_neighbors = parameters.n_neighbors_tailored(len(self.__data))

        if min_dist is None or min_dist < 0.0 or min_dist > 0.99:
            if min_dist is not None and (min_dist < 0.0 or min_dist > 0.99):
                print('min_dist must range from 0.0 up to 0.99. Default used.')
            if self.__sim_type != "tailored":
                if pca:
                    min_dist = parameters.MIN_DIST_STRUCTURAL_PCA
                else:
                    min_dist = parameters.MIN_DIST_STRUCTURAL
            else:
                min_dist = parameters.MIN_DIST_TAILORED

        # Embed the data in two dimensions
        self.umap_fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, n_components=2, **kwargs)
        umap_embedding = self.umap_fit.fit_transform(self.__data)
        self.__df_2_components = pd.DataFrame(data=umap_embedding
             , columns=['UMAP-1', 'UMAP-2'])

        if self.__test_df_descriptors is not None:
            test_umap_embedding = self.umap_fit.transform(self.__test_data)
            self.__test_df_2_components = pd.DataFrame(data=test_umap_embedding
                 , columns=['UMAP-1', 'UMAP-2'])

        if len(self.__target) > 0:
            self.__df_2_components['target'] = self.__target

            if self.__test_df_descriptors is not None:
                self.__test_df_2_components['target'] = self.__test_target

        return self.__df_2_components.copy()


    def cluster(self, n_clusters=5, **kwargs):
        """
        Computes the clusters presents in the embedded chemical space.

        :param n_clusters: Number of clusters that will be computed
        :param kwargs: Other keyword arguments are passed down to sklearn.cluster.KMeans
        :type n_clusters: int
        :type kwargs: key, value mappings
        :returns: The dataframe containing the 2D embedding.
        :rtype: Dataframe
        """

        if self.__df_2_components is None:
            print('Reduce the dimensions of your molecules before clustering.')
            return None


        x = self.__df_2_components.columns[0]
        y = self.__df_2_components.columns[1]

        cluster = cst.KMeans(n_clusters)
        # cluster = cst.Birch(n_clusters)

        cluster_list = cluster.fit_predict(self.__df_2_components[[x,y]])

        print(self.__df_2_components)
        print(cluster_list)
        self.__df_2_components['clusters'] = cluster_list

        if self.__test_df_2_components is not None:


            test_x = self.__test_df_2_components.columns[0]
            test_y = self.__test_df_2_components.columns[1]

            test_cluster_list = cluster.predict(self.__test_df_2_components[[test_x,test_y]])

            print(self.__test_df_2_components)
            print(test_cluster_list)


            self.__test_df_2_components['clusters'] = test_cluster_list

            return self.__test_df_2_components.copy()
        else:
            return self.__df_2_components.copy()


    # def visualize_plot(self, size=20, kind="scatter", remove_outliers=False, is_colored=True, colorbar=False, clusters=False, filename=None, title=None):
    #     """
    #     Generates a plot for the given molecules embedded in two dimensions.
    #
    #     :param size: Size of the plot
    #     :param kind: Type of plot
    #     :param remove_outliers: Boolean value indicating if the outliers must be identified and removed
    #     :param is_colored: Indicates if the points must be colored according to target
    #     :param colorbar: Indicates if the plot legend must be represented as a colorbar. Only considered when the target_type is "R".
    #     :param clusters: If True the clusters are shown instead of possible targets. Pass a list or a int to only show selected clusters (indexed by int).
    #     :param filename: Indicates the file where to save the plot
    #     :param title: Title of the plot.
    #     :type size: int
    #     :type kind: string
    #     :type remove_outliers: boolean
    #     :type is_colored: boolean
    #     :type colorbar: boolean
    #     :type clusters: boolean or list or int
    #     :type filename: string
    #     :type title: string
    #     :returns: The matplotlib axes containing the plot.
    #     :rtype: Axes
    #     """
    #     if self.__df_2_components is None:
    #         print('Reduce the dimensions of your molecules before creating a plot.')
    #         return None
    #
    #     if clusters is not False and 'clusters' not in self.__df_2_components.columns + self.__test_df_2_components.columns:
    #         print('Call cluster() before visualizing a plot with clusters.')
    #
    #     if title is None:
    #         title = self.__plot_title
    #
    #     if kind not in self._static_plots:
    #         kind = 'scatter'
    #         print('kind indicates which type of plot must be visualized. Currently supported static visualization are:\n'+
    #               '-scatter plot (scatter)\n'+
    #               '-hexagon plot (hex)\n'+
    #               '-kernel density estimation plot (kde)\n'+
    #               'Please input one between scatter, hex or kde for parameter kind.\n'+
    #               'As default scatter has been taken.')
    #
    #     x, y, df_data = self.__parse_dataframe(self.__df_2_components)
    #
    #     # Define colors
    #     hue = None
    #     hue_order = None
    #     palette = None
    #     if clusters is not False and 'clusters' in self.__df_2_components.columns:
    #         hue = 'clusters'
    #         palette = 'deep'
    #         if not isinstance(clusters, bool):
    #             if isinstance(clusters, int): clusters = [clusters]
    #             df_data['clusters'] = df_data['clusters'].isin(clusters)
    #             # Labels cluster
    #             total = df_data['clusters'].value_counts()
    #             t_s = total.get(True) if total.get(True) else 0
    #             p_s = t_s / total.sum()
    #             p_o = 1 - p_s
    #             labels = {
    #                 True: f'Selected - {p_s:.0%}',
    #                 False: f'Other - {p_o:.0%}'
    #                 }
    #             df_data.clusters.replace(labels, inplace=True)
    #             hue_order = list(labels.values())
    #         else:
    #             hue_order = self.__percentage_clusters(df_data)
    #             hue_order.sort()
    #     else:
    #         if len(self.__target) == 0:
    #             is_colored = False;
    #         else:
    #             if is_colored:
    #                 df_data = df_data.assign(target=self.__target)
    #                 hue = 'target'
    #                 palette = 'deep'
    #                 if self.__target_type == "R":
    #                     palette = sns.color_palette("inferno", as_cmap=True)
    #
    #
    #     # Remove outliers (using Z-score)
    #     if remove_outliers:
    #         df_data = self.__remove_outliers(x, y, df_data)
    #
    #     # Define plot aesthetics parameters
    #     sns.set_style("dark")
    #     sns.set_context("notebook", font_scale=size*0.15)
    #     fig, ax = plt.subplots(figsize=(size,size))
    #
    #     # Create a plot based on the reduced components
    #     if kind == "scatter":
    #         plot = sns.scatterplot(x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, data=df_data, s=size*3)
    #         plot.set_label("scatter")
    #         axis = plot
    #         plot.legend(markerscale=size*0.19)
    #         # Add colorbar
    #         if self.__target_type == "R" and colorbar:
    #             plot.get_legend().remove()
    #             norm = plt.Normalize(df_data['target'].min(), df_data['target'].max())
    #             cm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    #             cm.set_array([])
    #             plot.figure.colorbar(cm)
    #     elif kind == "hex":
    #         plot = ax.hexbin(df_data[x], df_data[y], gridsize=40, cmap='Blues')
    #         fig.colorbar(plot, ax=ax)
    #         ax.set_label("hex")
    #         axis = ax
    #     elif kind == "kde":
    #         plot = sns.kdeplot(x=x, y=y, shade=True, data=df_data)
    #         plot.set_label("kde")
    #         axis = plot
    #
    #     # Remove units from axis
    #     axis.set(yticks=[])
    #     axis.set(xticks=[])
    #     # Add labels
    #     axis.set_title(title,fontsize=size*2)
    #     axis.set_xlabel(x,fontsize=size*2)
    #     axis.set_ylabel(y,fontsize=size*2)
    #
    #     # Save plot
    #     if filename is not None:
    #         fig.savefig(filename)
    #
    #     self.df_plot_xy = df_data[[x,y]]
    #
    #     return axis

    # def interactive_plot(self, size=700, kind="scatter", remove_outliers=False, is_colored=True, clusters=False, filename=None, show_plot=False, title=None, imgs=True):
    #     """
    #     Generates an interactive Bokeh plot for the given molecules embedded in two dimensions.
    #
    #     :param size: Size of the plot
    #     :param kind: Type of plot
    #     :param remove_outliers: Boolean value indicating if the outliers must be identified and removed
    #     :param is_colored: Indicates if the points must be colored according to target
    #     :param clusters: Indicates if to add a tab with the clusters if these have been computed
    #     :param filename: Indicates the file where to save the Bokeh plot
    #     :param show_plot: Immediately display the current plot.
    #     :param title: Title of the plot.
    #     :type size: int
    #     :type kind: string
    #     :type remove_outliers: boolean
    #     :type is_colored: boolean
    #     :type cluster: boolean
    #     :type filename: string
    #     :type show_plot: boolean
    #     :type title: string
    #     :returns: The bokeh figure containing the plot.
    #     :rtype: Figure
    #     """
    #     if self.__df_2_components is None:
    #         print('Reduce the dimensions of your molecules before creating a plot.')
    #         return None
    #
    #     if clusters and 'clusters' not in self.__df_2_components.columns + self.__test_df_2_components.columns:
    #         print('Call cluster() before visualizing a plot with clusters.')
    #
    #     if title is None:
    #         title = self.__plot_title
    #
    #     if kind not in self._interactive_plots:
    #         kind = 'scatter'
    #         print('kind indicates which type of plot must be visualized. Currently supported interactive visualization are:\n'+
    #               '-scatter plot (scatter)\n'+
    #               '-hexagon plot (hex)\n'+
    #               'Please input one between scatter, hex or kde for parameter kind.\n'+
    #               'As default scatter has been taken.')
    #
    #     x, y, df_data = self.__parse_dataframe(self.__df_2_components)
    #     df_data['mols'] = self.__mols
    #
    #
    #
    #     if self.__test_df_2_components is not None:
    #         test_x, test_y, test_df_data = self.__parse_dataframe(self.__test_df_2_components)
    #         test_df_data['mols'] = self.__test_mols
    #     else:
    #         test_x = test_y = test_df_data = None
    #
    #     if len(self.__target) > 0:
    #         # Target exists
    #         if self.__target_type == 'C':
    #             df_data['target'] = list(map(str, self.__target))
    #         else:
    #             df_data['target'] = self.__target
    #
    #     # Remove outliers (using Z-score)
    #     if remove_outliers:
    #         df_data = self.__remove_outliers(x, y, df_data)
    #
    #     tabs = None
    #     if kind == "scatter":
    #         p, tabs = self.__interactive_scatter(x, y, df_data, test_x, test_y, test_df_data, size, is_colored, clusters, title, imgs=imgs)
    #     else:
    #         p = self.__interactive_hex(x, y, df_data, size, title)
    #
    #     p.xaxis[0].axis_label = x
    #     p.yaxis[0].axis_label = y
    #
    #     p.xaxis.major_tick_line_color = None
    #     p.xaxis.minor_tick_line_color = None
    #     p.yaxis.major_tick_line_color = None
    #     p.yaxis.minor_tick_line_color = None
    #     p.xaxis.major_label_text_font_size = '0pt'
    #     p.yaxis.major_label_text_font_size = '0pt'
    #
    #     if tabs is not None:
    #         p = tabs
    #
    #     # Save plot
    #     if filename is not None:
    #         output_file(filename, title=title)
    #         save(p)
    #
    #     # Show plot
    #     if show_plot:
    #         self.__open_plot(p)
    #
    #     self.df_plot_xy = df_data[[x,y]]
    #
    #     if self.__test_df_2_components is not None:
    #         self.test_df_plot_xy = test_df_data[[test_x, test_y]]
    #
    #     return p

    def faerun_interactive_plot(self, filename):

        if self.__df_2_components is None or (self.__test_df_descriptors is not None and self.__test_df_2_components is None):
                print('Reduce the dimensions of your molecules before creating a plot.')
                return None

        x, y, df_data = self.__parse_dataframe(self.__df_2_components)
        smiles = self.__df_descriptors.index.values

        if self.__test_df_descriptors is not None:
            test_x, test_y, test_df_data = self.__parse_dataframe(self.__test_df_2_components)
            test_smiles = self.__test_df_descriptors.index.values

        if self.__target_type == 'R':
            categorical = False
            c = self.__target
            if self.__test_df_descriptors is not None:
                test_c = self.__test_target
        else:
            categorical = True
            c = np.ones(len(df_data))
            if self.__test_df_descriptors is not None:
                test_c = np.ones(len(test_df_data))



        print(smiles)

        legend_style = {
            "bottom": "10px",
            "right": "10px",
            "padding": "10px",
            "border": "1px solid #262626",
            "border-radius": "2px",
            "background-color": "#111111",
            "filter": "drop-shadow(0px 0px 10px rgba(0, 0, 0, 0.5))",
            "color": "#eeeeee",
            "font-family": "'Open Sans'",
            "max-height": "100%",
            "overflow": "scroll"
        }

        f = Faerun(
            clear_color="#222222",
            x_title=x, y_title=y,
            view="front",
            style={"legend": legend_style}
        )

        global_colormap = "autumn"
        test_colormap = "winter"
        cluster_colormap = cm.batlowS

        if "clusters" in self.__df_2_components.columns:
            clusters = self.__df_2_components['clusters']

            f.add_scatter(
                name="global",
                data={
                    "x": df_data[x],
                    "y": df_data[y],
                    "c": [c, clusters],
                    "labels": smiles
                },
                colormap=[global_colormap, cluster_colormap],
                max_point_size=10.0,
                categorical=[categorical, True],
                has_legend=True,
                series_title=["All", "Clusters"]
            )
        else:
            f.add_scatter(
                name="global",
                data={
                    "x": df_data[x],
                    "y": df_data[y],
                    "c": c,
                    "labels": smiles
                },
                colormap=global_colormap,
                max_point_size=10.0,
                categorical=categorical,
                has_legend=True
            )


        if self.__test_df_descriptors is not None:

            if "clusters" in self.__test_df_2_components.columns:
                test_clusters = self.__test_df_2_components['clusters']

                f.add_scatter(
                    name="test",
                    data={
                        "x": test_df_data[x],
                        "y": test_df_data[y],
                        "c": [test_c, test_clusters],
                        "labels": test_smiles
                    },
                    colormap=[test_colormap, cluster_colormap],
                    max_point_size=10.0,
                    categorical=[categorical, True],
                    has_legend=True,
                    series_title=["All", "Clusters"]
                )
            else:

                f.add_scatter(
                    name="test",
                    data={
                        "x": test_df_data[x],
                        "y": test_df_data[y],
                        "c": test_c,
                        "labels": test_smiles
                    },
                    colormap=test_colormap,
                    max_point_size=10.0,
                    categorical=categorical,
                    has_legend=True
                )

        with open(f'{filename}.faerun', 'wb+') as handle:
            pickle.dump(f.create_python_data(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        f.plot(filename, template="smiles")

        # host(f'{filename}.faerun', theme='dark', legend=True)

        self.df_plot_xy = df_data[[x,y]]
        if self.__test_df_descriptors is not None:
            self.test_df_plot_xy = test_df_data[[test_x,test_y]]

    def to_pickle(self, filename='plotter.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename='plotter.pkl'):
        with open(filename, 'rb') as f:
            self = pickle.load(f)

        return self


    def __data_scaler(self, data):
        # Scale the data
        if self.__sim_type == "tailored":
            scaled_data = StandardScaler().fit_transform(data.values.tolist())
        else:
            scaled_data = data.values.tolist()

        return scaled_data

    def __parse_dataframe(self, data):
        x = data.columns[0]
        y = data.columns[1]

        return x, y, data.copy()

    def __remove_outliers(self, x, y, df):
        # Remove outliers (using Z-score)
        z_scores = stats.zscore(df[[x,y]])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)

        return df[filtered_entries]

    def __percentage_clusters(self, df_data):
        total = df_data['clusters'].value_counts()
        sum_tot = total.sum()
        labels = {}
        count = 0
        for key, value in total.items():
            p = float(f"{(value/sum_tot)*100:.0f}")
            labels[key] = p
            count += p
        # Solve possible rounding errors
        if 100 - count > 0:
            labels[0] = labels[0] + 100 - count
        for key, value in labels.items():
            labels[key] = f"Cluster {key} - {value:.0f}%"
        # Edit df_data and return labels
        df_data.clusters.replace(labels, inplace=True)
        return list(labels.values())

    # def __interactive_scatter(self, x, y, df_data, test_x, test_y, test_df_data, size, is_colored, clusters, title, imgs=True):
    #     tabs = None
    #     p_c = None
    #     p_test = None
    #
    #     if imgs:
    #         # Add images column
    #         df_data['imgs'] = self.__mol_to_2Dimage(list(df_data['mols']))
    #
    #         if test_df_data is not None:
    #             test_df_data['imgs'] = self.__mol_to_2Dimage(list(test_df_data['mols']))
    #
    #     df_data.drop(columns=['mols'], inplace=True)
    #     if test_df_data is not None:
    #         test_df_data.drop(columns=['mols'], inplace=True)
    #
    #     # Set tools
    #     tools = "pan, lasso_select, wheel_zoom, hover, save, reset"
    #
    #     if len(self.__target) == 0:
    #         TOOLTIPS = parameters.TOOLTIPS_NO_TARGET
    #     else:
    #         TOOLTIPS = parameters.TOOLTIPS_TARGET
    #
    #     # Create plot
    #     p = figure(title=title, plot_width=size, plot_height=size, tools=tools, tooltips=TOOLTIPS)
    #     if test_df_data is not None:
    #         p_test = figure(title=title, plot_width=size, plot_height=size, tools=tools, tooltips=TOOLTIPS)
    #
    #         p_test.xaxis[0].axis_label = test_x
    #         p_test.yaxis[0].axis_label = test_y
    #
    #         p_test.xaxis.major_tick_line_color = None
    #         p_test.xaxis.minor_tick_line_color = None
    #         p_test.yaxis.major_tick_line_color = None
    #         p_test.yaxis.minor_tick_line_color = None
    #         p_test.xaxis.major_label_text_font_size = '0pt'
    #         p_test.yaxis.major_label_text_font_size = '0pt'
    #
    #     if len(self.__target) == 0 or not(is_colored):
    #         p.circle(x=x, y=y, size=2.5, alpha=0.8, source=df_data)
    #         if test_df_data is not None:
    #             p_test.circle(x=test_x, y=test_y, size=2.5, alpha=0.8, source=test_df_data)
    #     else:
    #         if test_df_data is not None:
    #             targets = pd.concat([test_df_data['target'], df_data['target']])
    #
    #         else:
    #             targets = df_data['target']
    #
    #         print(test_df_data['target'])
    #         print(df_data['target'])
    #         print(targets)
    #
    #
    #         # Target exists
    #         if self.__target_type == 'C':
    #             index_cmap = factor_cmap('target', Category10[10], list(set(targets)))
    #
    #             p.circle(x=x, y=y, size=2.5, alpha=0.8, line_color=index_cmap, fill_color=index_cmap,
    #                  legend_group="target", source=df_data)
    #             p.legend.location = "top_left"
    #             p.legend.title = "Target"
    #
    #             if test_df_data is not None:
    #                 p_test.circle(x=test_x, y=test_y, size=2.5, alpha=0.8, line_color=index_cmap, fill_color=index_cmap,
    #                      legend_group="target", source=test_df_data)
    #                 p_test.legend.location = "top_left"
    #                 p_test.legend.title = "Target"
    #
    #         else:
    #             color_mapper = LinearColorMapper(Inferno[256], low=min(targets), high=max(targets))
    #             index_cmap = transform('target', color_mapper)
    #
    #             p.circle(x=x, y=y, size=2.5, alpha=0.8, line_color=index_cmap, fill_color=index_cmap,
    #                  source=df_data)
    #             color_bar = ColorBar(color_mapper=color_mapper, location=(0,0))
    #             p.add_layout(color_bar, 'right')
    #
    #             if test_df_data is not None:
    #                 p_test.circle(x=test_x, y=test_y, size=2.5, alpha=0.8, line_color=index_cmap, fill_color=index_cmap,
    #                      source=test_df_data)
    #                 color_bar = ColorBar(color_mapper=color_mapper, location=(0,0))
    #                 p_test.add_layout(color_bar, 'right')
    #
    #
    #     if clusters and 'clusters' in df_data.columns + test_df_data.columns:
    #         if 'clusters' in df_data.columns:
    #             p_c = figure(title=title, plot_width=size, plot_height=size, tools=tools, tooltips=parameters.TOOLTIPS_CLUSTER)
    #             # Get percentages
    #             self.__percentage_clusters(df_data)
    #             clusters = df_data.groupby(['clusters'])
    #             for cluster, color in zip(clusters, Category10[10]):
    #                 p_c.circle(x=x, y=y, size=2.5, alpha=1, line_color=color, fill_color=color,
    #                      legend_label=f'{cluster[0]}', muted_color=('#717375'), muted_alpha=0.2,
    #                      source=cluster[1])
    #         elif 'clusters' in test_df_data.columns:
    #             p_c = figure(title=title, plot_width=size, plot_height=size, tools=tools, tooltips=parameters.TOOLTIPS_CLUSTER)
    #             # Get percentages
    #             self.__percentage_clusters(test_df_data)
    #             clusters = test_df_data.groupby(['clusters'])
    #             for cluster, color in zip(clusters, Category10[10]):
    #                 p_c.circle(x=test_x, y=test_y, size=2.5, alpha=1, line_color=color, fill_color=color,
    #                      legend_label=f'{cluster[0]}', muted_color=('#717375'), muted_alpha=0.2,
    #                      source=cluster[1])
    #
    #         p_c.legend.location = "top_left"
    #         p_c.legend.title = "Clusters"
    #         p_c.legend.click_policy = "mute"
    #
    #         p_c.xaxis[0].axis_label = x
    #         p_c.yaxis[0].axis_label = y
    #
    #         p_c.xaxis.major_tick_line_color = None
    #         p_c.xaxis.minor_tick_line_color = None
    #         p_c.yaxis.major_tick_line_color = None
    #         p_c.yaxis.minor_tick_line_color = None
    #         p_c.xaxis.major_label_text_font_size = '0pt'
    #         p_c.yaxis.major_label_text_font_size = '0pt'
    #
    #     if p_c or p_test:
    #         tabs = [Panel(child=p, title="Plot")]
    #
    #         if p_c:
    #             tabs.append(Panel(child=p_c, title="Clusters"))
    #         if p_test:
    #             tabs.append(Panel(child=p_test, title="Test"))
    #
    #         tabs = Tabs(tabs=tabs)
    #
    #     return p, tabs
    #
    # def __interactive_hex(self, x, y, df_data, size, title):
    #     # Hex Plot
    #     df_data.drop(columns=['mols'], inplace=True)
    #
    #     tools = "pan, wheel_zoom, save, reset"
    #
    #     p = figure(title=title, plot_width=size, plot_height=size, match_aspect=True,
    #        tools=tools)
    #     p.background_fill_color = '#440154'
    #     p.grid.visible = False
    #
    #     max_x = max(df_data[x])
    #     min_x = min(df_data[x])
    #     max_y = max(df_data[y])
    #     min_y = min(df_data[y])
    #
    #     diff_x = max_x - min_x
    #     diff_y = max_y - min_y
    #     size = max(diff_y, diff_x) / 20
    #
    #     p.hexbin(df_data[x], df_data[y], size=size, hover_color="pink", hover_alpha=0.8)
    #
    #     hover = HoverTool(tooltips=[("count", "@c")])
    #     p.add_tools(hover)
    #
    #     return p

    def __mol_to_2Dimage(self, list_mols):
        # Create molecule images
        images_mol=[]
        for mol in tqdm(list_mols):

            try:
                png = Draw.MolToImage(mol,size=(200,130))
                out = BytesIO()
                png.save(out, format='jpeg')
                png = out.getvalue()
                url = 'data:image/jpeg;base64,' + base64.b64encode(png).decode('utf-8')
            except:
                url = None

            images_mol.append(url)

        return images_mol

    @calltracker
    def __open_plot(self, p):
        show(p)

    def get_target(self):
        return self.__target

    def get_data(self):
        return self.__df_descriptors.copy()

    def get_test_data(self):
        return self.__test_df_descriptors.copy()
