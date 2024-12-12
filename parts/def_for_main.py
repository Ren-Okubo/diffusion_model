import torch
import matplotlib.pyplot as plt
from diffusion_x_h import E3DiffusionProcess, remove_mean
import os
from loss_calculation import kabsch_torch
from schedulefree import RAdamScheduleFree

def noise_schedule_for_GammaNetwork(model_state_path,params,target:str):
    assert target in ['gamma','alpha','sigma','SNR'], 'target must be one of gamma, alpha, sigma, or SNR'

    #パラメータの設定
    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_schedule = params['noise_schedule']
    noise_precision = params['noise_precision']
    power = params['noise_schedule_power']

    #diffusion_processの定義
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)

    #モデルの読み込み
    state_dicts = torch.load(model_state_path,weights_only=True)
    if 'GammaNetwork' in state_dicts:
        diffusion_process.gamma.load_state_dict(state_dicts['GammaNetwork'])
    
    #targetの計算
    fig, ax = plt.subplots()
    t = diffusion_process.t
    ax.set_xlabel('t')
    if target == 'gamma':
        target = diffusion_process.gamma_schedule()
        ax.set_ylabel('gamma')
        ax.set_title('gamma schedule')
    elif target == 'alpha':
        if noise_schedule == 'predefined':
            target = diffusion_process.alpha_schedule
        elif noise_schedule == 'learned':
            target = torch.sqrt(torch.sigmoid(-diffusion_process.gamma(t)))
        ax.set_ylabel('alpha')
        ax.set_title('alpha schedule')
    elif target == 'sigma':
        if noise_schedule == 'predefined':
            target = diffusion_process.sigma_schedule
        elif noise_schedule == 'learned':
            target = torch.sqrt(torch.sigmoid(diffusion_process.gamma(t)))
        ax.set_ylabel('sigma')
        ax.set_title('sigma schedule')
    elif target == 'SNR':
        if noise_schedule == 'predefined':
            target = diffusion_process.alpha_schedule ** 2 / diffusion_process.sigma_schedule ** 2
        elif noise_schedule == 'learned':
            target = torch.exp(-diffusion_process.gamma(t))
        ax.set_ylabel('SNR')
        ax.set_title('SNR')

    #targetのプロット
    target = target.to('cpu')
    t = t.to('cpu')
    ax.plot(t.detach().numpy(),target.detach().numpy())
    
    return fig


def load_model_state(nn_dict,model_save_path,params):
    state_dicts = torch.load(model_save_path,weights_only=True)
    nn_dict['egnn'].load_state_dict(state_dicts['egnn'])
    if params['to_compress_spectrum']:
        nn_dict['spectrum_compressor'].load_state_dict(state_dicts['spectrum_compressor'])
    if params['noise_schedule'] == 'learned':
        nn_dict['gamma'].load_state_dict(state_dicts['gamma'])
    return nn_dict


def evaluate_by_rmsd(original_graph_list,generated_graph_list):
    id_list = []
    rmsd_value_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        _,_,rmsd_value = kabsch_torch(original_graph.pos,generated_graph.pos)
        rmsd_value_list.append(rmsd_value)
        id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_rmsd_original_generated_list = list(zip(id_list,rmsd_value_list,original_coords_list,generated_coords_list)) #rmsdの値でソート
    sorted_id_rmsd_original_generated_list = sorted(id_rmsd_original_generated_list,key=lambda x:x[1])
    return sorted_id_rmsd_original_generated_list

def evaluate_by_rmsd_and_atom_type_score(original_graph_list,generated_graph_list):
    id_list = []
    rmsd_value_list = []
    atom_type_score_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        _,_,rmsd_value = kabsch_torch(original_graph.pos,generated_graph.pos)
        rmsd_value_list.append(rmsd_value)
        atom_type_score = 0
        for i in range(original_graph.x.shape[0]):
            if torch.equal(original_graph.x[i],generated_graph.x[i]):
                atom_type_score += 1
        atom_type_score = atom_type_score / original_graph.x.shape[0]
        atom_type_score_list.append(atom_type_score)
        id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_rmsd_original_generated_list = list(zip(id_list,rmsd_value_list,atom_type_score_list,original_coords_list,generated_coords_list)) #rmsdの値でソート
    sorted_id_rmsd_atomscore_original_generated_list = sorted(id_rmsd_original_generated_list,key=lambda x:x[1])
    return sorted_id_rmsd_atomscore_original_generated_list

def define_optimizer(params,nn_dict,diffusion_process,optim_type:str):
    assert optim_type in ['Adam','AdamW','RAdamScheduleFree']
    lr = params['lr']
    weight_decay = params['weight_decay']
    if params['to_compress_spectrum']:
        if params['noise_schedule'] == 'learned':
            param_list_for_optim = list(nn_dict['egnn'].parameters())+list(nn_dict['spectrum_compressor'].parameters())+list(diffusion_process.parameters())
        else:
            param_list_for_optim = list(nn_dict['egnn'].parameters())+list(nn_dict['spectrum_compressor'].parameters())
    else:
        if params['noise_schedule'] == 'learned':
            param_list_for_optim = list(nn_dict['egnn'].parameters())+list(diffusion_process.parameters())
        else:
            param_list_for_optim = list(nn_dict['egnn'].parameters())
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(param_list_for_optim,lr=lr,weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(param_list_for_optim,lr=lr,weight_decay=weight_decay,amsgrad=True)
    elif optim_type == 'RAdamScheduleFree':
        optimizer = RAdamScheduleFree(param_list_for_optim,lr=lr)
    return optimizer