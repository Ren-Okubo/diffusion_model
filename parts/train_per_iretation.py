import torch, copy, itertools, random, datetime, pdb, yaml, pytz
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import torch.nn.init as init
import split_to_train_and_test
from split_to_train_and_test import SetUpData
from EquivariantGraphNeuralNetwork import EquivariantGNN, EGCL
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from E3diffusion_new import E3DiffusionProcess, remove_mean
from CN2_evaluate import calculate_angle_for_CN2
from DataPreprocessor import SpectrumCompressor
import wandb


def diffuse_as_batch(batch_data:graph,graph_index,params,diffusion_process:E3DiffusionProcess,conditional=True):
    
    time_list = [i for i in range(1,num_diffusion_timestep+1)]
    num_graph = graph_index.max().item()+1
    pos_before_diffusion, h_before_diffusion, pos_after_diffusion,h_after_diffusion, y_for_noise_pos, y_for_noise_h, each_time_list = [],[],[],[],[],[],[]
    
    #バッチ内のグラフごとに処理を行う
    for i in range(num_graph):
        num_atom = len(x_before_dif.shape[0])

        #拡散させる時間をランダムに選ぶ
        time = random.choice(time_list)

        #posの拡散
        pos_before_dif = batch_data.pos[graph_index==i]
        pos_after_dif, noise_pos = diffusion_process.diffuse_zero_to_t(pos_before_dif,time)
        pos_before_diffusion.append(pos_before_dif)
        pos_after_diffusion.append(pos_after_dif)
        y_for_noise_pos.append(noise_pos)

        #hの拡散
        x_before_dif = batch_data.x[graph_index==i]
        if conditional == True:
            spectrum_before_dif = batch_data.spectrum[graph_index==i]
            #使用するスペクトルデータの範囲を限定
            spectrum_before_dif = spectrum_before_dif[:,:spectrum_size]
            #スペクトルデータを圧縮するかどうかのifを追加　todo
            h_before_dif = torch.cat((onehot_scaling_factor*x_before_dif,spectrum_before_dif),dim=1)
        else:
            h_before_dif = onehot_scaling_factor*x_before_dif
        h_after_dif, noise_h = diffusion_process.diffuse_zero_to_t(h_before_dif,time)
        h_before_diffusion.append(h_before_dif)
        h_after_diffusion.append(h_after_dif)
        y_for_noise_h.append(noise_h)
        #グラフごとに拡散させた時間を記録
        each_time_list.append([time for j in range(num_atom)])

    #グラフごとに処理したデータをバッチとしてまとめる
    pos_before_diffusion = torch.cat(pos_before_diffusion,dim=0)
    h_before_diffusion = torch.cat(h_before_diffusion,dim=0)
    pos_after_diffusion = torch.cat(pos_after_diffusion,dim=0)
    h_after_diffusion = torch.cat(h_after_diffusion,dim=0)
    y_for_noise_pos = torch.cat(y_for_noise_pos,dim=0)
    y_for_noise_h = torch.cat(y_for_noise_h,dim=0)
    each_time_list = torch.cat(each_time_list,dim=0)

    #batch_dataに拡散後のデータを格納
    batch_data.pos_at_t = pos_after_diffusion
    batch_data.h_at_t = h_after_diffusion
    batch_data.pos_at_0 = pos_before_diffusion
    batch_data.h_at_0 = h_before_diffusion
    batch_data.y_for_noise_pos = y_for_noise_pos
    batch_data.y_for_noise_h = y_for_noise_h
    batch_data.each_time_list = each_time_list

    return batch_data
    



        

def train_epoch(train_loader,params,diffusion_process,optimizer,nn_dict):
    egnn = model
    egnn.train()
    train_dataset = train_loader

    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    conditional = params['conditional']

    epoch_loss_train = 0
    total_num_train_node = 0   

    for train_data in train_dataset:
        optimizer.zero_grad()
        total_num_train_node += train_data.num_nodes
        graph_index = train_data.batch
        num_diffusion_timestep = params['num_diffusion_timestep']

        #拡散後のデータを取得
        train_data = diffuse_as_batch(train_data,graph_index,num_diffusion_timestep,diffusion_process,conditional=conditional)
        
        #時間のデータをnum_diffusion_timestepで正規化
        time_data = train_data.each_time_list / num_diffusion_timestep

        #equivariant graph neural networkによる予測
        h, x = egnn(train_data.edge_index,torch.cat((train_data.h_at_t,time_data),dim=1),train_data.pos_at_t)
        epsilon_x = x - train_data.pos_at_t
        epsilon_x = remove_mean(epsilon_x,graph_index)
        epsilon_h = h

        #lossの計算
        predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1) #予測したepsilon
        target_epsilon = torch.cat((train_data.y_for_noise_pos,train_data.y_for_noise_h),dim=1) #正解のepsilon
        loss = torch.nn.MSELoss(predicted_epsilon,target_epsilon)
        loss = loss / num_graph #グラフ一つ当たりのloss

        #lossのbackward逆伝播
        loss.backward()

        #optimizerによるパラメータの更新
        optimizer.step()

        #lossの記録
        loss = loss * num_graph
        epoch_loss_train += loss.item()
    
    avg_loss_train = epoch_loss_train / total_num_train_node #各ノードごとのlossの平均

    return avg_loss_train

def eval_epoch(params,model,eval_data,diffusion_process):
    egnn = model
    egnn.eval()
    eval_dataset = eval_data

    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    conditional = params['conditional']

    epoch_loss_val = 0
    total_num_val_node = 0

    with torch.no_grad():
        for val_data in eval_dataset:
            total_num_val_node += val_data.num_nodes
            graph_index = val_data.batch
            num_diffusion_timestep = params['num_diffusion_timestep']

            #拡散後のデータを取得
            val_data = diffuse_as_batch(val_data,graph_index,num_diffusion_timestep,diffusion_process,conditional=conditional)
            
            #時間のデータをnum_diffusion_timestepで正規化
            time_data = val_data.each_time_list / num_diffusion_timestep

            #equivariant graph neural networkによる予測
            h, x = egnn(val_data.edge_index,torch.cat((val_data.h_at_t,time_data),dim=1),val_data.pos_at_t)
            epsilon_x = x - val_data.pos_at_t
            epsilon_x = remove_mean(epsilon_x,graph_index)
            epsilon_h = h

            #lossの計算
            predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1)
            target_epsilon = torch.cat((val_data.y_for_noise_pos,val_data.y_for_noise_h),dim=1)
            loss = torch.nn.MSELoss(predicted_epsilon,target_epsilon)

            #lossの記録
            epoch_loss_val += loss.item()

    avg_loss_val = epoch_loss_val / total_num_val_node #各ノードごとのlossの平均

    return avg_loss_val

def save_model_state(model_save_path,**kwargs):
    torch.save(kwargs,model_save_path)

        
        




        