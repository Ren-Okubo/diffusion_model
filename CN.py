import torch, copy, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, Subset
from torch_geometric.loader import DataLoader
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, f1_score
from itertools import product
import argparse
import wandb
import random
from schedulefree import RAdamScheduleFree


def calculation_macro_f1score_for_binary(true_data:np.array,prediction_data:np.array):
    TP, FP, FN, TN = 0,0,0,0
    for i in range(len(true_data)):
        if true_data[i] == 0:
            if prediction_data[i] == 0:
                TP +=1
            else:
                FN +=1
        else:
            if prediction_data[i] == 0:
                FP +=1
            else:
                TN +=1
    if TP == 0:
        f1score_0 = 0
    else:
        f1score_0 = (2*TP)/(2*TP+FN+FP)
    TP, FP, FN, TN = 0,0,0,0
    for i in range(len(true_data)):
        if true_data[i] == 1:
            if prediction_data[i] == 1:
                TP +=1
            else:
                FN +=1
        else:
            if prediction_data[i] == 1:
                FP +=1
            else:
                TN +=1
    if TP == 0:
        f1score_1 = 0
    else:
        f1score_1 = (2*TP)/(2*TP+FN+FP)
    return (f1score_0 + f1score_1)/2

def calculation_macro_f1score_for_multi_label(true_data:np.array,prediction_data:np.array):
    label_list = list(np.unique(true_data))
    score_list = []
    for label in label_list:
        TP, FP, FN, TN = 0,0,0,0
        for i in range(len(true_data)):
            if true_data[i] == label:
                if prediction_data[i] == label:
                    TP +=1
                else:
                    FN +=1
            else:
                if prediction_data[i] == label:
                    FP +=1
                else:
                    TN +=1
        if TP == 0:
            f1score = 0
        else:
            f1score = (2*TP)/(2*TP+FN+FP)
        score_list.append(f1score)
    average_score = sum(score_list)/len(score_list)
    return average_score

class MLP(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(MLP,self).__init__()
        layers = []
        #入力層
        layers.append(nn.Linear(input_size,hidden_size[0]))
        layers.append(torch.nn.ReLU())
        #隠れ層
        for i in range(1,len(hidden_size)):
            layers.append(nn.Linear(hidden_size[i-1],hidden_size[i]))
            layers.append(torch.nn.ReLU())
        #出力層
        layers.append(nn.Linear(hidden_size[-1],1))

        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)
    
def train(model,train_loader,optimizer,criterion):
    model.train()
    optimizer.train()
    running_loss = 0
    for data in train_loader:
        inputs = data.spectrum[0,:].to('cuda')
        labels = torch.tensor(data.x.shape[0],dtype=torch.float32).to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def eval(model,eval_loader,optimizer,criterion):
    model.eval()
    optimizer.eval()
    running_loss = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs = data.spectrum[0,:].to('cuda')
            labels = torch.tensor(data.x.shape[0],dtype=torch.float32).to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            running_loss += loss.item()
    return running_loss

def test(model,test_loader):
    model.eval()
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for data in test_loader:
            inputs = data.spectrum[0,:].to('cuda')
            labels = torch.tensor(data.x.shape[0],dtype=torch.float32).to('cuda')
            outputs = model(inputs)
            prediction_list.append(int(outputs.item()))
            target_list.append(int(labels.item()))
    return prediction_list, target_list

class EarlyStopping():
    def __init__(self, patience=0):
        self._step= 0
        self._loss=float('inf')
        self._patience=patience

    def validate(self,loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self._patience:
                return True
        else:
            self._step = 0
            self._loss = loss
       
        return False

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--project_name', type=str, required=True)
    argparse.add_argument('--run_name', default=None)
    argparse.add_argument('--dataset_path', type=str, required=True)
    args = argparse.parse_args()

    params = {'batch_size': 32,
              'input_size': 200,
              'hidden_size': [100,100,50,25],
              'epoch': 1000,
              'patience' : 50,
              'lr' : 0.0001,
              'seed' : 0,
              'device' : 'cuda',
              'dataset_path' : args.dataset_path,
              'weight_decay' : 1.0e-12}

    run = wandb.init(project=args.project_name, name=args.run_name, config=params)
    config = wandb.config
    device = torch.device(config.device)
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = torch.load(config.dataset_path)
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    eval_size = int(dataset_size*0.1)
    test_size = dataset_size - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(dataset,[train_size,eval_size,test_size])
    train_loader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    eval_loader = DataLoader(eval_dataset,batch_size=config.batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    model = MLP(config.input_size,config.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = RAdamScheduleFree(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    early_stopping = EarlyStopping(patience=config.patience)

    for epoch in range(config.epoch):
        train_loss = train(model,train_dataset,optimizer,criterion)
        eval_loss = eval(model,eval_dataset,optimizer,criterion)
        wandb.log({'train_loss':train_loss,'eval_loss':eval_loss})
        if early_stopping.validate(eval_loss):
            break
    
    torch.save(model.state_dict(),os.path.join(run.dir,'model.pth'))
    wandb.config.update({'model_path':os.path.join(run.dir,'model.pth')})

    prediction_list, target_list = test(model,test_dataset)
    max_num = max(max(prediction_list),max(target_list))
    min_num = min(min(prediction_list),min(target_list))
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,10))
    ax.plot([0,max_num],[0,max_num],'-',color='red')
    ax.plot(target_list,prediction_list,marker='o',linestyle='None')
    ax.set_xlabel('target')
    ax.set_ylabel('prediction')
    ax.set_title('target vs prediction')
    wandb.log({'target vs prediction':wandb.Image(fig)})
    torch.save({'prediction_list':prediction_list,'target_list':target_list},os.path.join(run.dir,'prediction_target.pt'))
    wandb.config.update({'prediction_target_path':os.path.join(run.dir,'prediction_target.pt')})
    run.finish()

        
    


