import torch, copy, itertools, random, datetime, pdb, yaml, pytz, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import torch.nn.init as init
import split_to_train_and_test
from tqdm import tqdm
from split_to_train_and_test import SetUpData
from EquivariantGraphNeuralNetwork import EquivariantGNN, EGCL
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from E3diffusion_new import E3DiffusionProcess, remove_mean
from CN2_evaluate import calculate_angle_for_CN2
from DataPreprocessor import SpectrumCompressor
import wandb

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

def diffuse_as_batch(batch_data,graph_index,params,diffusion_process:E3DiffusionProcess):
    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    spectrum_size = params['spectrum_size']
    onehot_scaling_factor = params['onehot_scaling_factor']

    #各グラフの拡散時間をランダムに選ぶためのリスト
    time_list = [i for i in range(1,num_diffusion_timestep+1)]

    num_graph = graph_index.max().item()+1
    pos_before_diffusion, h_before_diffusion, pos_after_diffusion,h_after_diffusion, y_for_noise_pos, y_for_noise_h, each_time_list = [],[],[],[],[],[],[]
    
    # graph_indexをGPUに移動
    graph_index = graph_index.to(batch_data.pos.device)    
    
    #バッチ内のグラフごとに処理を行う
    for i in range(num_graph):
        num_atom = batch_data.pos[graph_index==i].shape[0]

        #拡散させる時間をランダムに選ぶ
        time = random.choice(time_list)

        #posの拡散
        pos_before_dif = batch_data.pos[graph_index==i]
        pos_after_dif, noise_pos = diffusion_process.diffuse_zero_to_t(pos_before_dif,time)
        pos_before_diffusion.append(pos_before_dif)
        pos_after_diffusion.append(pos_after_dif)
        y_for_noise_pos.append(noise_pos)

        #hの拡散 hは連結によって元素種、spectrum、時間の情報を持つが拡散させるのは元素種のみ
        x_before_dif = batch_data.x[graph_index==i]
        h_after_dif, noise_h = diffusion_process.diffuse_zero_to_t(x_before_dif,time)
        h_before_diffusion.append(x_before_dif)
        h_after_diffusion.append(h_after_dif)
        y_for_noise_h.append(noise_h)
        #グラフごとに拡散させた時間を記録
        each_time_list.append(torch.stack([torch.tensor([time]) for j in range(num_atom)]))

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
    



        

def train_epoch(nn_dict,train_loader,params,diffusion_process,optimizer):
    egnn = nn_dict['egnn']
    egnn.train()
    egnn.to('cuda')
    if params['optimizer'] == 'RAdamScheduleFree':
        optimizer.train()
    criterion = nn.MSELoss(reduction='mean')

    if params['to_compress_spectrum']:
        spectrum_compressor = nn_dict['spectrum_compressor']
        spectrum_compressor.train()

    if params['noise_schedule'] == 'learned':
        diffusion_process.gamma.train()

    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    atom_type_size = params['atom_type_size']
    conditional = params['conditional']

    epoch_loss_train = 0
    total_num_train_node = 0   

    for train_data in train_loader:
        train_data.to('cuda')
        optimizer.zero_grad()
        total_num_train_node += train_data.num_nodes
        graph_index = train_data.batch
        num_graph = graph_index.max().item()+1

        #拡散後のデータを取得
        train_data = diffuse_as_batch(train_data,graph_index,params,diffusion_process)
        
        #時間のデータをnum_diffusion_timestepで正規化
        time_data = train_data.each_time_list / num_diffusion_timestep

        #egnnに渡すhのデータを定義
        h_for_input = train_data.h_at_t.to('cuda')
        if conditional:
            if params['to_compress_spectrum']:
                compressed_spectrum = spectrum_compressor(train_data.spectrum.to('cuda'))
                h_for_input = torch.cat((h_for_input,compressed_spectrum.to('cuda')),dim=1).to('cuda')
            else:
                h_for_input = torch.cat((h_for_input,train_data.spectrum.to('cuda')),dim=1).to('cuda')
        if params['give_exO']:
            h_for_input = torch.cat((h_for_input,train_data.exO.to('cuda')),dim=1)
        h_for_input = torch.cat((h_for_input,time_data.to('cuda')),dim=1)
        
        

        #equivariant graph neural networkによる予測
        h, x = egnn(train_data.edge_index.to('cuda'),h_for_input,train_data.pos_at_t.to('cuda'))
        """
        if conditional:
            if params['to_compress_spectrum']:
                compressed_spectrum = spectrum_compressor(train_data.spectrum.to('cuda'))
                h, x = egnn(train_data.edge_index.to('cuda'),torch.cat((train_data.h_at_t.to('cuda'),compressed_spectrum.to('cuda'),time_data.to('cuda')),dim=1).to('cuda'),train_data.pos_at_t.to('cuda'))
            else:
                h, x = egnn(train_data.edge_index.to('cuda'),torch.cat((train_data.h_at_t.to('cuda'),train_data.spectrum.to('cuda'),time_data.to('cuda')),dim=1).to('cuda'),train_data.pos_at_t.to('cuda'))
        else:
            h, x = egnn(train_data.edge_index.to('cuda'),torch.cat((train_data.h_at_t.to('cuda'),time_data.to('cuda')),dim=1).to('cuda'),train_data.pos_at_t.to('cuda'))
        """
        epsilon_x = x - train_data.pos_at_t
        epsilon_x = remove_mean(epsilon_x,graph_index)
        epsilon_h = h[:,:atom_type_size] #atom_typeの部分のみを取り出す

        #lossの計算
        predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1) #予測したepsilon
        target_epsilon = torch.cat((train_data.y_for_noise_pos,train_data.y_for_noise_h),dim=1) #正解のepsilon
        loss = criterion(predicted_epsilon,target_epsilon)
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

def eval_epoch(nn_dict,eval_loader,params,diffusion_process,optimizer):
    egnn = nn_dict['egnn']
    egnn.eval()
    egnn.to('cuda')
    if params['optimizer'] == 'RAdamScheduleFree':
        optimizer.eval()
    criterion = nn.MSELoss(reduction='mean')

    if params['to_compress_spectrum']:
        spectrum_compressor = nn_dict['spectrum_compressor']
        spectrum_compressor.eval()

    if params['noise_schedule'] == 'learned':
        diffusion_process.gamma.eval()

    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    conditional = params['conditional']
    atom_type_size = params['atom_type_size']

    epoch_loss_val = 0
    total_num_val_node = 0

    with torch.no_grad():
        for val_data in eval_loader:
            val_data.to('cuda')
            total_num_val_node += val_data.num_nodes
            graph_index = val_data.batch
            num_diffusion_timestep = params['num_diffusion_timestep']

            #拡散後のデータを取得
            val_data = diffuse_as_batch(val_data,graph_index,params,diffusion_process)
            
            #時間のデータをnum_diffusion_timestepで正規化
            time_data = val_data.each_time_list / num_diffusion_timestep

            #egnnに渡すのデータを定義
            h_for_input = val_data.h_at_t.to('cuda')
            if conditional:
                if params['to_compress_spectrum']:
                    compressed_spectrum = spectrum_compressor(val_data.spectrum.to('cuda'))
                    h_for_input = torch.cat((h_for_input,compressed_spectrum.to('cuda')),dim=1).to('cuda')
                else:
                    h_for_input = torch.cat((h_for_input,val_data.spectrum.to('cuda')),dim=1).to('cuda')
            if params['give_exO']:
                h_for_input = torch.cat((h_for_input,val_data.exO.to('cuda')),dim=1)
            h_for_input = torch.cat((h_for_input,time_data.to('cuda')),dim=1)
            
            

            #equivariant graph neural networkによる予測
            h, x = egnn(val_data.edge_index.to('cuda'),h_for_input,val_data.pos_at_t.to('cuda'))
            """
            if conditional:
                if params['to_compress_spectrum']:
                    compressed_spectrum = spectrum_compressor(val_data.spectrum.to('cuda'))
                    h, x = egnn(val_data.edge_index,torch.cat((val_data.h_at_t,compressed_spectrum,time_data),dim=1),val_data.pos_at_t)
                else:
                    h, x = egnn(val_data.edge_index,torch.cat((val_data.h_at_t,val_data.spectrum,time_data),dim=1),val_data.pos_at_t)
            else:
                h, x = egnn(val_data.edge_index,torch.cat((val_data.h_at_t,time_data),dim=1),val_data.pos_at_t)
            """
            epsilon_x = x - val_data.pos_at_t
            epsilon_x = remove_mean(epsilon_x,graph_index)
            epsilon_h = h[:,:atom_type_size] #atom_typeの部分のみを取り出す
            

            #lossの計算
            predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1)
            target_epsilon = torch.cat((val_data.y_for_noise_pos,val_data.y_for_noise_h),dim=1)
            loss = criterion(predicted_epsilon,target_epsilon)

            #lossの記録
            epoch_loss_val += loss.item()

    avg_loss_val = epoch_loss_val / total_num_val_node #各ノードごとのlossの平均

    return avg_loss_val

def generate(nn_dict,test_data,params,diffusion_process):
    #モデルの読み込み
    egnn = nn_dict['egnn']
    egnn.eval()
    egnn.to('cuda')
    if params['to_compress_spectrum']:
        spectrum_compressor = nn_dict['spectrum_compressor']
        spectrum_compressor.eval()
        spectrum_compressor.to('cuda')
    if params['noise_schedule'] == 'learned':
        diffusion_process.gamma.eval()
        diffusion_process.gamma.to('cuda')
    
    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    conditional = params['conditional']
    atom_type_size = params['atom_type_size']
    spectrum_size = params['spectrum_size']
    onehot_scaling_factor = params['onehot_scaling_factor']

    
    original_graph_list ,generated_graph_list = [],[]

    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            data = test_data[i].to('cuda')

            #生成するデータサイズを取得（本来は予測する）
            num_of_atoms = data.x.shape[0]

            #一つの条件に対して生成するデータ数と実際に生成したデータ数、nanの数をカウント
            how_many_gen = 5
            num_of_generated_data = 0
            num_of_generated_nan = 0

            while num_of_generated_data != how_many_gen:
                #初期値の設定
                initial_pos = torch.zeros(size=(num_of_atoms,3))
                initial_pos.normal_(mean=0,std=1)
                initial_pos = remove_mean(initial_pos)
                initial_h = torch.zeros(size=(num_of_atoms,atom_type_size))
                initial_h.normal_(mean=0,std=1)

                #グラフデータのedge_indexを生成
                edge_index = []
                for i in range(num_of_atoms):
                    for j in range(num_of_atoms):
                        edge_index.append([i,j])
                edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()

                #初期値をData型に変換
                if conditional:
                    if params['give_exO']:
                        graph = Data(x=initial_h,edge_index=edge_index,pos=initial_pos,spectrum=data.spectrum,exO=data.exO)
                    else:
                        graph = Data(x=initial_h,edge_index=edge_index,pos=initial_pos,spectrum=data.spectrum)
                else:
                    graph = Data(x=initial_h,edge_index=edge_index,pos=initial_pos)

                
                #100stepごとのデータを格納するリスト
                transition_data_per_100step = []

                #逆拡散
                for t in range(num_diffusion_timestep,0,-1):
                    if t%100 == 0:
                        transition_data_per_100step.append(graph)

                    #時間のデータをnum_diffusion_timestepで正規化
                    time_tensor = torch.tensor([[t/num_diffusion_timestep] for d in range(num_of_atoms)],dtype=torch.float32)

                    #特徴量ベクトルの定義
                    graph.h = onehot_scaling_factor*graph.x
                    if conditional:
                        if params['to_compress_spectrum']:
                            compressed_spectrum = spectrum_compressor(graph.spectrum)
                            graph.h = torch.cat((graph.h,compressed_spectrum),dim=1)
                        else:
                            graph.h = torch.cat((graph.h,graph.spectrum),dim=1)
                    if params['give_exO']:
                        graph.h = torch.cat((graph.h,graph.exO),dim=1)
                    graph.h = torch.cat((graph.h,time_tensor),dim=1)
                    """ 
                    if conditional:
                        if params['to_compress_spectrum']:
                            compressed_spectrum = spectrum_compressor(graph.spectrum)
                            graph.h = torch.cat((onehot_scaling_factor*graph.x,compressed_spectrum,time_tensor),dim=1)
                        else:
                            graph.h = torch.cat((onehot_scaling_factor*graph.x,graph.spectrum,time_tensor),dim=1)
                    else:
                        graph.h = torch.cat((onehot_scaling_factor*graph.x,time_tensor),dim=1)
                    """
                    #equivariant graph neural networkによる予測
                    new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos)
                    
                    #epsilonの計算
                    epsilon_x = new_x - graph.pos
                    epsilon_x = remove_mean(epsilon_x)
                    epsilon_h = new_h[:,:atom_type_size] #atom_typeの部分のみを取り出す

                    #逆拡散
                    graph.pos = diffusion_process.reverse_diffuse_one_step(graph.pos,epsilon_x,t,mode='pos')
                    graph.x = diffusion_process.reverse_diffuse_one_step(graph.h[:,:2],epsilon_h,t,mode='h')

                    #nanが出力されていないかの確認
                    if not torch.isfinite(graph.x).all():
                        num_of_generated_nan += 1
                        #seed_value += 1
                        if num_of_generated_nan == 10:
                            print('too much nan was generated')
                            exit()
                        break
                    if not torch.isfinite(graph.pos).all():
                        num_of_generated_nan += 1
                        #seed_value += 1
                        if num_of_generated_nan == 10:
                            print('too much nan was generated')
                            exit()
                        break
                
                time_tensor = torch.tensor([[0] for d in range(num_of_atoms)],dtype=torch.float32)
                graph.h = onehot_scaling_factor*graph.x
                if conditional:
                    if params['to_compress_spectrum']:
                        compressed_spectrum = spectrum_compressor(graph.spectrum)
                        graph.h = torch.cat((graph.h,compressed_spectrum),dim=1)
                    else:
                        graph.h = torch.cat((graph.h,graph.spectrum),dim=1)
                if params['give_exO']:
                    graph.h = torch.cat((graph.h,graph.exO),dim=1)
                graph.h = torch.cat((graph.h,time_tensor),dim=1)
                """
                if conditional:
                    if params['to_compress_spectrum']:
                        compressed_spectrum = spectrum_compressor(graph.spectrum)
                        graph.h = torch.cat((onehot_scaling_factor*graph.x,compressed_spectrum,time_tensor),dim=1)
                    else:
                        graph.h = torch.cat((onehot_scaling_factor*graph.x,graph.spectrum,time_tensor),dim=1)
                else:
                    graph.h = torch.cat((onehot_scaling_factor*graph.x,time_tensor),dim=1)
                """
                new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos)
                epsilon_x = remove_mean(new_x - graph.pos)
                graph.h = graph.h[:,:atom_type_size]
                epsilon_h = new_h[:,:atom_type_size]
                alpha_0 = diffusion_process.alpha(0)
                sigma_0 = diffusion_process.sigma(0)
                mu_x = graph.pos / alpha_0 - sigma_0 * epsilon_x / alpha_0
                noise_x = torch.zeros_like(mu_x)
                noise_x.normal_(mean=0,std=1)
                noise_x = remove_mean(noise_x)
                graph.pos = mu_x + sigma_0 * noise_x / alpha_0
                mu_h = graph.h / alpha_0 - sigma_0 * epsilon_h / alpha_0
                noise_h = torch.zeros_like(mu_h)
                noise_h.normal_(mean=0,std=1)
                graph.h = mu_h + sigma_0 * noise_h / alpha_0
                h_atoms = nn.functional.one_hot(torch.argmax(graph.h,dim=1),num_classes=2)
                graph.x = h_atoms

                #t=0におけるデータを格納
                if torch.isfinite(graph.x).all() and torch.isfinite(graph.pos).all():
                    transition_data_per_100step.append(graph)
                    #座標の値が現実的でないものは除外(50Å以上の座標が出力された場合)
                    if torch.any(graph.pos > 50):
                        continue
                    num_of_generated_data += 1

                    if conditional:
                        original_graph_list.append(data)
                    else:
                        original_graph_list.append(-1)
                    generated_graph_list.append(transition_data_per_100step)
                
    return original_graph_list,generated_graph_list


                    

                        



        
        
