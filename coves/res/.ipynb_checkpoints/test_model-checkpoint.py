# test.py


#test_path ='/n/groups/marks/users/david/res/model_save/model2/RES_1639761348.356219_25.pt'
test_path = '/n/groups/marks/users/david/res/model_save/model2/RES_1646945484.3030427_8.pt'
model_id = test_path.split('/')[-1][:-3]
f_path_out = '/n/groups/marks/users/david/res/out/{}.txt'.format(model_id)


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


lba_split = 30
def get_datasets(task, lba_split=lba_split):
    data_path = {
        'RES' : '/n/groups/marks/users/david/res/atom3d_data/raw/RES/data/',
    }[task]

    if task == 'RES':
        split_path = '/n/groups/marks/users/david/res/atom3d_data/split-by-cath-topology/indices/'
        dataset = partial(gvp.atom3d.RESDataset, data_path)        
        trainset = dataset(split_path=split_path+'train_indices.txt')
        valset = dataset(split_path=split_path+'val_indices.txt')
        testset = dataset(split_path=split_path+'test_indices.txt')


    return trainset, valset, testset

datasets = get_datasets('RES')

batch_size = 8 # control memory of model as long as you can fit a batch size of 1, you can do gradient accumultion to simulate batch size.
num_workers = 4
dataloader = partial(torch_geometric.data.DataLoader, 
                    num_workers=num_workers, batch_size=batch_size)

trainset, valset, testset = map(dataloader, datasets)   

def get_model(task):
    return {
        'RES' : gvp.atom3d.RESModel,
    }[task]()

model = get_model('RES').to(device)


def get_metrics():
    return {'accuracy': metrics.accuracy}



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


def test(model, testset, model_path):
    #model.load_state_dict(torch.load(args.test))
    model.load_state_dict(torch.load(model_path))

    model.eval()
    t = tqdm.tqdm(testset)
    metrics = get_metrics()
    targets, predicts, ids = [], [], []
    with torch.no_grad():
        for batch in t:
            pred = forward(model, batch, device)
            label = get_label(batch, 'RES', None)
            pred = pred.argmax(dim=-1)

            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))

    for name, func in metrics.items():

        value = func(targets, predicts)
        print(f"{name}: {value}")
        with open(f_path_out, 'w') as fout:
            fout.write(f"{model_id}_{name}: {value}")

test(model, testset, test_path)