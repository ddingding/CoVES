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

####### get the model
def get_model(task):
    return {
        'RES' : gvp.atom3d.RESModel,
    }[task]()

model = get_model('RES').to(device)

######################## load the best


#load_path = '/n/groups/marks/users/david/res/model_save/RES_1638990570.3313198_32.pt'
#load_path = '/n/groups/marks/users/david/res/model_save/model2/RES_1639327734.154226_32.pt'
#load_path = '/n/groups/marks/users/david/res/model_save/model2/RES_1639761348.356219_25.pt'
#load_path = '/n/groups/marks/users/david/res/model_save/model2/RES_1640009835.334532_30.pt'
load_path = '/n/groups/marks/users/david/res/model_save/model2/RES_1646945484.3030427_8.pt'
print('starting with model path {}'.format(load_path))

model.load_state_dict(torch.load(load_path))

############ train
lr = 1e-4
epochs = 50
train_time = 120
val_time = 20
model_save_path = '/n/groups/marks/users/david/res/model_save/model2'
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
    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0
    
    for batch in t:
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
            
        t.set_description(f"{total_loss/total_count:.8f}")
        
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