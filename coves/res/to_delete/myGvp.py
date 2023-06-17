#res model

import torch, random, scipy, math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from atom3d.datasets import LMDBDataset
#import atom3d.datasets.ppi.neighbors as nb
from torch.utils.data import IterableDataset
from . import GVP, GVPConvLayer, LayerNorm
import torch_cluster, torch_geometric, torch_scatter
from .data import _normalize, _rbf

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)
_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)

def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
            (edge_s, edge_v))

    return edge_s, edge_v

class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                            dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, 
                                D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)

class BaseModel(nn.Module):
    '''
    A base 5-layer GVP-GNN for all ATOM3D tasks, using GVPs with 
    vector gating as described in the manuscript. Takes in atomic-level
    structure graphs of type `torch_geometric.data.Batch`
    and returns a single scalar.
    
    This class should not be used directly. Instead, please use the
    task-specific models which extend BaseModel. (Some of these classes
    may be aliases of BaseModel.)
    
    :param num_rbf: number of radial bases to use in the edge embedding
    '''
    def __init__(self, num_rbf=16):
        
        super().__init__()
        activations = (F.relu, None)
        
        self.embed = nn.Embedding(_NUM_ATOM_TYPES, _NUM_ATOM_TYPES) # _NUM_ATOM_TYPES --> size of dictionary of embedding, 
                                                                    # _NUM_ATOM_TYPES --> size of vector of each embedding
        
        self.W_e = nn.Sequential(
            LayerNorm((num_rbf, 1)),
            GVP((num_rbf, 1), _DEFAULT_E_DIM, 
                activations=(None, None), vector_gate=True)
        )
        
        self.W_v = nn.Sequential(
            LayerNorm((_NUM_ATOM_TYPES, 0)),
            GVP((_NUM_ATOM_TYPES, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )
        
        # define 5 gvp convlayers.
        self.layers = nn.ModuleList(
                GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, 
                             activations=activations, vector_gate=True) 
            for _ in range(5))
        
        ns, _ = _DEFAULT_V_DIM
        self.W_out = nn.Sequential(
            LayerNorm(_DEFAULT_V_DIM),
            GVP(_DEFAULT_V_DIM, (ns, 0), 
                activations=activations, vector_gate=True)
        )
        
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2*ns, 1)
        )
    
    def forward(self, batch, scatter_mean=True, dense=True):
        '''
        Forward pass which can be adjusted based on task formulation.
        
        :param batch: `torch_geometric.data.Batch` with data attributes
                      as returned from a BaseTransform
        :param scatter_mean: if `True`, returns mean of final node embeddings
                             (for each graph), else, returns embeddings seperately
        :param dense: if `True`, applies final dense layer to reduce embedding
                      to a single scalar; else, returns the embedding
        '''
        h_V = self.embed(batch.atoms)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        batch_id = batch.batch
        
        for layer in self.layers: # self.layers is the the moduleList of 5 gvpConvLayers
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        if scatter_mean: out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense: out = self.dense(out).squeeze(-1) # squeeze() removes dimensions of size 1, what does -1 mean?
        return out


class RESModel(BaseModel):
    '''
    GVP-GNN for the RES task.
    
    Extends BaseModel to output a 20-dim vector instead of a single
    scalar for each graph, which can be used as logits in 20-way
    classification.
    
    As noted in the manuscript, RESModel uses the final alpha
    carbon embeddings instead of the graph mean embedding.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # makes the child class inherit all proporties of parent class
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential( # this overwrites the self.dense defined in the basemodel to output 20 dimensions instead.
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2*ns, 20)
        )
    def forward(self, batch):
        out = super().forward(batch, scatter_mean=False) # call the base model transformations.
        return out[batch.ca_idx+batch.ptr[:-1]]