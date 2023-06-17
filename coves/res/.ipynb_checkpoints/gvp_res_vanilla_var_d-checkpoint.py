# saved as /n/groups/marks/users/david/res/model_save//RES_1638990570.3313198_32.pt

import gvp
from atom3d.datasets import LMDBDataset
import torch_geometric
from functools import partial
import gvp.atom3d
import torch.nn as nn
import tqdm, torch, time, os
import numpy as np
from atom3d.util import metrics
import sklearn.metrics as sk_metrics
from collections import defaultdict
import scipy.stats as stats
print = partial(print, flush=True)

models_dir = 'models'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())
print(device)




##### get datasets
import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster

import sys

from atom3d.datasets import LMDBDataset
#import atom3d.datasets.ppi.neighbors as nb
from torch.utils.data import IterableDataset
import atom3d



d_cutoff = float(sys.argv[1])
print('using edgecutoff distance of ', d_cutoff)


######################################
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


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

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

class myRESDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D RES dataset.
    
    On each iteration, returns a `torch_geometric.data.Data`
    graph with the attribute `label` encoding the masked residue
    identity, `ca_idx` for the node index of the alpha carbon, 
    and all structural attributes as described in BaseTransform.
    
    Excludes hydrogen atoms.
    
    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    '''
    def __init__(self, lmdb_dataset, d_cutoff, split_path):
        self.dataset = LMDBDataset(lmdb_dataset)
        self.idx = list(map(int, open(split_path).read().split()))
        self.transform = BaseTransform(edge_cutoff = d_cutoff)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.idx))), 
                      shuffle=True)
        else:  
            per_worker = int(math.ceil(len(self.idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(list(range(len(self.idx)))[iter_start:iter_end],
                      shuffle=True)
        return gen
    
    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: random.shuffle(indices)
        with torch.no_grad(): # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
            my_c=0
            for idx in indices:
                data = self.dataset[self.idx[idx]]
                atoms = data['atoms']
                for sub in data['labels'].itertuples():
                    _, num, aa = sub.subunit.split('_')
                    num, aa = int(num), _amino_acids(aa)
                    if aa == 20: continue
                    my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                    #if my_c <3:
                    #    print(set(my_atoms.element))
                    #    my_c+=1
                    ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                    if len(ca_idx) != 1: continue
                        
                    with torch.no_grad():
                        graph = self.transform(my_atoms)
                        graph.label = aa
                        graph.ca_idx = int(ca_idx)
                        yield graph

def get_datasets(task, d_cutoff):
    data_path = {
        'RES' : '/n/groups/marks/users/david/res/atom3d_data/raw/RES/data/',
    }[task]

    if task == 'RES':
        split_path = '/n/groups/marks/users/david/res/atom3d_data/split-by-cath-topology/indices/'
        dataset = partial(myRESDataset, data_path, d_cutoff)        
        trainset = dataset(split_path=split_path+'train_indices.txt')
        valset = dataset(split_path=split_path+'val_indices.txt')
        testset = dataset(split_path=split_path+'test_indices.txt')


    return trainset, valset, testset


datasets = get_datasets('RES', d_cutoff)


batch_size = 8 # control memory of model as long as you can fit a batch size of 1, you can do gradient accumultion to simulate batch size.
num_workers = 4
dataloader = partial(torch_geometric.data.DataLoader, 
                    num_workers=num_workers, batch_size=batch_size)

trainset, valset, testset = map(dataloader, datasets)   # apply dataloader to all the datasets

####### get the model
def get_model(task):
    return {
        'RES' : gvp.atom3d.RESModel,
    }[task]()

model = get_model('RES').to(device)



lr = 1e-4
epochs = 50
train_time = 120
val_time = 20
model_save_path = '/n/groups/marks/users/david/res/model_save/'
smp_idx = None


def get_label(batch, task, smp_idx=None):
    if type(batch) in [list, tuple]: batch = batch[0]
    if task == 'SMP':
        assert smp_idx is not None
        return batch.label[smp_idx::20]
    return batch.label

def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = batch[0].to(device), batch[1].to(device)
    else:
        batch = batch.to(device)
    return model(batch)

def loop(dataset, model, optimizer=None, max_time=None):
    start = time.time()
    
    loss_fn = nn.CrossEntropyLoss()
    #t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0
    
    for batch in dataset:
        if max_time and (time.time() - start) > 60*max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            out = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
            
        label = get_label(batch, 'RES', smp_idx)
        loss_value = loss_fn(out, label)
        total_loss += float(loss_value)
        total_count += 1
        
        if optimizer:
            try:
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
            
        #t.set_description(f"{total_loss/total_count:.8f}")
        
    return total_loss / total_count

def train(model, trainset, valset):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_path, best_val = None, np.inf

    for epoch in range(epochs):
        print('epoch ', epoch)
        model.train()
        loss = loop(trainset, model, optimizer=optimizer, max_time=train_time)
        path = f"{model_save_path}/RES_{model_id}_{epoch}.pt"
        torch.save(model.state_dict(), path)
        print(f'\nEPOCH {epoch} TRAIN loss: {loss:.8f}')
        model.eval()
        with torch.no_grad():
            loss = loop(valset, model, max_time=val_time)
        print(f'\nEPOCH {epoch} VAL loss: {loss:.8f}')
        if loss < best_val:
            best_path, best_val = path, loss
        print(f'BEST {best_path} VAL loss: {best_val:.8f}')


train(model, trainset, valset)