import argparse
import sys
import os
import torch
import yaml
import wandb
import random
import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from EquivariantGraphNeuralNetwork import EquivariantGNN
from diffusion_x_h import E3DiffusionProcess
from split_to_train_and_test import SetUpData
from DataPreprocessor import SpectrumCompressor
from schedulefree import RAdamScheduleFree

sys.path.append('parts/')
from train_per_iretation import diffuse_as_batch, train_epoch, eval_epoch, generate, EarlyStopping
from loss_calculation import kabsch_torch
from def_for_main import load_model_state, evaluate_by_rmsd, noise_schedule_for_GammaNetwork, evaluate_by_rmsd_and_atom_type_eval, define_optimizer

if __name__ == '__main__':
    project_name = str(input('Enter the name of the project: '))
    run_id = str(input('Enter the run id: '))
    api = wandb.Api()
    run = api.run(f'{project_name}/{run_id}')
    prms = run.config
    #パラメータの設定
    #全体のパラメータ
    conditional=prms['conditional']
    seed = prms['seed']
    num_epochs = prms['num_epochs']
    batch_size = prms['batch_size']
    #optimizerのパラメータ
    lr = prms['lr']
    weight_decay = prms['weight_decay']
    optim_type = prms['optimizer']
    #early stoppingのパラメータ
    patience = prms['patience']
    #diffusion_processのパラメータ
    num_diffusion_timestep = prms['num_diffusion_timestep']
    noise_schedule = prms['noise_schedule']
    noise_precision = prms['noise_precision']
    power = prms['noise_schedule_power']
    #spectrum_compressorのパラメータ
    to_compress_spectrum = prms['to_compress_spectrum']
    compressor_hidden_dim = prms['compressor_hidden_dim']
    compressed_spectrum_size = prms['compressed_spectrum_size']
    #egnnのパラメータ
    L = prms['L']
    atom_type_size = prms['atom_type_size']
    spectrum_size = prms['spectrum_size']
    d_size = prms['d_size']
    t_size = prms['t_size']
    exO_size = prms['exO_size']
    if conditional:
        if prms['to_compress_spectrum']:
            h_size = atom_type_size + compressed_spectrum_size + t_size
        else:
            h_size = atom_type_size + spectrum_size + t_size
    else:
        h_size = atom_type_size + t_size
    if prms['give_exO']:
        h_size = h_size + exO_size
    x_size = prms['x_size']
    m_size = prms['m_size']   
    m_input_size = h_size + h_size + d_size
    m_hidden_size = prms['m_hidden_size']
    m_output_size = m_size
    h_input_size = h_size + m_size
    h_hidden_size = prms['h_hidden_size']
    h_output_size = h_size 
    x_input_size = h_size + h_size + d_size
    x_hidden_size = prms['x_hidden_size']
    x_output_size = 1
    onehot_scaling_factor = prms['onehot_scaling_factor']
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)
    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(spectrum_size,compressor_hidden_dim,compressed_spectrum_size)
    #使用するモデルをまとめた辞書nn_dictを定義
    if to_compress_spectrum:
        nn_dict = {'egnn':egnn,'spectrum_compressor':spectrum_compressor}
    else:
        nn_dict = {'egnn':egnn,'spectrum_compressor':None}
    model_save_path = prms['model_save_path']
    load_model_state(nn_dict,model_save_path,prms)
    project = 'amorphous'
    run_name = str(input(f'Enter the run name ({run_id}): '))
    run = wandb.init(project=project,name=run_name,config=prms)
    dataset_path = str(input('Enter the path of the dataset: '))
    dataset = torch.load(dataset_path)
    wandb.config.update({'amorphous_dataset_path': dataset_path})
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    original_graph_list, generated_graph_list = generate(nn_dict,dataset,prms,diffusion_process)
    generated_graph_save_path = os.path.join(wandb.run.dir,'generated_graph.pt')
    original_graph_save_path = os.path.join(wandb.run.dir,'original_graph.pt')
    torch.save(generated_graph_list,generated_graph_save_path)
    torch.save(original_graph_list,original_graph_save_path)
    wandb.config.update({'original_graph_save_path': original_graph_save_path},allow_val_change=True)
    print('The original graph has been saved.')
    wandb.config.update({'generated_graph_save_path': generated_graph_save_path},allow_val_change=True)
    print('The generated graph has been saved.')
    original_graph_list = torch.load(original_graph_save_path)
    generated_graph_list = torch.load(generated_graph_save_path)
    Si_tensor = torch.tensor([0,1],dtype=torch.long)
    O_tensor = torch.tensor([1,0],dtype=torch.long)
    density_O_original = []
    density_O_generated = []
    for i in range(len(original_graph_list)):
        num_O = 0
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        for j in range(original_graph.x.shape[0]):
            if torch.equal(original_graph.x[j],O_tensor):
                num_O += 1
            elif torch.equal(original_graph.x[j],Si_tensor):
                pass
            else:
                print('Error')
        density_O_original.append(num_O/original_graph.x.shape[0])
        num_O = 0
        for j in range(generated_graph.x.shape[0]):
            if torch.equal(generated_graph.x[j],O_tensor):
                num_O += 1
            elif torch.equal(generated_graph.x[j],Si_tensor):
                pass
            else:
                print('Error')
        density_O_generated.append(num_O/generated_graph.x.shape[0])
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1],linestyle='-',color='red')
    ax.plot(density_O_original,density_O_generated,linestyle='None',marker='o')
    ax.set_xlabel('density of O in original')
    ax.set_ylabel('density of O in generated')
    ax.set_title('density of O in original and generated')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.text(0.05, 0.95, f'accuracy: {sum([1 for i in range(len(density_O_original)) if abs(density_O_original[i]-density_O_generated[i])==0])/len(density_O_original)}', 
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    wandb.log({'atom_type_eval':wandb.Image(fig)})
    plt.close()

    wandb.finish()