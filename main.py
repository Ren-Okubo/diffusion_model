import argparse
import sys
import os
import wandb
import yaml
import random
import torch
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
from make_xyz_from_wandb_run import write_xyz
from schedulefree import RAdamScheduleFree
import torch_geometric.datasets as datasets
from torch.utils.data import random_split

sys.path.append('/mnt/homenfsxx/rokubo/data/diffusion_model/parts/')
from train_per_iretation import diffuse_as_batch, train_epoch, eval_epoch, generate, EarlyStopping
from loss_calculation import kabsch_torch
from def_for_main import load_model_state, evaluate_by_rmsd, noise_schedule_for_GammaNetwork, evaluate_by_rmsd_and_atom_type_eval, define_optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',type=str,default='diffusion_first_nearest_loss_per_atom')
    parser.add_argument('--run_name',type=str,default=None)
    parser.add_argument('--dataset_path',type=str,default='/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset.pt')
    parser.add_argument('--mode',type=str,default='train_and_generate') #train_and_generate, train_only, generate_only, evaluate_only
    parser.add_argument('--record_schedule',type=bool,default=False)
    parser.add_argument('--create_xyz_file',type=bool,default=False)
    parser.add_argument('--note',type=str,default=None)
    parser.add_argument('--test_by_provided_data',type=str,default=None) #"QM9" or None
    args = parser.parse_args()

    #parameterの読み込み
    with open ('parameters.yaml','r') as file:
        prms = yaml.safe_load(file)
    jst=pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    prms['now'] = now.strftime("%Y%m%d%H%M")

    #wandbの設定
    assert args.mode in ['train_and_generate','train_only','generate_only','evaluate_only']
    if args.mode == 'train_and_generate' or args.mode == 'train_only':
        run = wandb.init(project=args.project_name,config=prms,name=args.run_name)
    elif args.mode == 'generate_only' or args.mode == 'evaluate_only':
        run_id = input('run_id:')
        run = wandb.init(project=args.project_name,id=run_id,name=args.run_name,resume='must')
        prms = run.config

    #メモがあればnoteに記録
    if args.note:
        wandb.run.notes = args.note

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
    if args.test_by_provided_data is None:
        atom_type_size = prms['atom_type_size']
    elif args.test_by_provided_data == 'QM9':
        atom_type_size = 5
        prms['atom_type_size'] = 5
        wandb.config.update({'atom_type_size':5},allow_val_change=True)
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

    #default_tensor_typeの設定
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    #seedの設定
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #modelの定義
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)
    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(spectrum_size,compressor_hidden_dim,compressed_spectrum_size)
    early_stopping = EarlyStopping(patience=patience)
    setupdata = SetUpData(seed=seed,conditional=conditional)    

    #使用するモデルをまとめた辞書nn_dictを定義
    if to_compress_spectrum:
        nn_dict = {'egnn':egnn,'spectrum_compressor':spectrum_compressor}
    else:
        nn_dict = {'egnn':egnn,'spectrum_compressor':None}
    
    #datasetの読み込み
    if args.test_by_provided_data is None:
        dataset_path = args.dataset_path
        dataset = torch.load(dataset_path)
        wandb.config.update({'dataset_path':dataset_path})
        for data in dataset: #spectrumサイズの設定（-1~19eVがデフォルト）
            spectrum = data.spectrum
            resized_spectrum = spectrum[:,:spectrum_size]
            data.spectrum = resized_spectrum
    elif args.test_by_provided_data == 'QM9': #QM9のデータセットを使う場合
        dataset = datasets.QM9('/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/QM9/')
        subset, _ = random_split(dataset,[10000,len(dataset)-10000],generator=torch.Generator(device='cuda'))
        dataset = list(subset)
        #dataset = list(dataset)
        for data in dataset:
            data.x = data.x[:,:5] #qm9のデータからatom_typeのみを取り出す
        wandb.config.update({'dataset_path':'QM9'})


    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 乱数生成器を作成し、CUDAデバイスに設定
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)  # 任意のシード値を設定

    #datasetの分割
    train_data, eval_data, test_data = setupdata.split(dataset)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,generator=generator)
    eval_loader = DataLoader(eval_data,batch_size=batch_size,shuffle=True,generator=generator)




    #optimizerの設定

    optimizer = define_optimizer(prms,nn_dict,diffusion_process,optim_type=optim_type)


    if "train" in args.mode:
        #train,eval
        for epoch in range(num_epochs):
            avg_loss_train = train_epoch(nn_dict,train_loader,prms,diffusion_process,optimizer)
            avg_loss_eval = eval_epoch(nn_dict,eval_loader,prms,diffusion_process,optimizer)

            #各epochにおけるlossを記録
            print(f'epoch:{epoch}    train_loss:{avg_loss_train}     eval_loss:{avg_loss_eval}')
            wandb.log({'train_loss':avg_loss_train,'eval_loss':avg_loss_eval})

            #early stopping
            if early_stopping.validate(avg_loss_eval):
                break
        
        #モデルの保存
        model_states = {'egnn':egnn.state_dict()}
        if prms['to_compress_spectrum']:
            model_states['spectrum_compressor'] = spectrum_compressor.state_dict()
        if prms['noise_schedule'] == 'learned':
            model_states['gamma'] = diffusion_process.gamma.state_dict()
        
        model_save_path = os.path.join(wandb.run.dir,'model.pth')
        torch.save(model_states,model_save_path)
        wandb.config.update({'model_save_path':model_save_path})
        print(f'model saved at {model_save_path}')

        #trainのみの場合
        if args.mode == 'only_train':
            wandb.finish()
            sys.exit()

    #generateのみの場合モデルの状態を読み込む
    if args.mode == 'generate_only':
        model_save_path = wandb.config['model_save_path']
        load_model_state(nn_dict,model_save_path,prms)

    if "generate" in args.mode:
        #generate
        original_graph_list, generated_graph_list = generate(nn_dict,test_data,prms,diffusion_process)

        #生成したグラフを保存
        generated_graph_save_path = os.path.join(wandb.run.dir,'generated_graph.pt')
        torch.save(generated_graph_list,generated_graph_save_path)
        wandb.config.update({'generated_graph_save_path':generated_graph_save_path})
        print(f'generated graph saved at {generated_graph_save_path}')
        if conditional:
            original_graph_save_path = os.path.join(wandb.run.dir,'original_graph.pt')
            torch.save(original_graph_list,original_graph_save_path)
            wandb.config.update({'original_graph_save_path':original_graph_save_path})
            print(f'original graph saved at {original_graph_save_path}')
    
    #evaluateのみの場合
    if args.mode == 'evaluate_only':
        original_graph_list = torch.load(wandb.config['original_graph_save_path'])
        generated_graph_list = torch.load(wandb.config['generated_graph_save_path'])


    #生成したデータの評価
    if conditional:
        
        #生成したグラフのrmsdを計算し、ソート
        sorted_id_rmsd_original_generated_list = evaluate_by_rmsd(original_graph_list,generated_graph_list)
        
        #rmsdの値を保存 [(id,rmsd,original_graph,generated_graph),...]
        rmsd_save_path = os.path.join(wandb.run.dir,'rmsd.pt')
        torch.save(sorted_id_rmsd_original_generated_list,rmsd_save_path)
        wandb.config.update({'rmsd_save_path':rmsd_save_path})
        print(f'rmsd saved at {rmsd_save_path}')

        #ソートしたrmsdの描画
        sorted_id_list, sorted_rmsd_list,sorted_original_graph_list,sorted_generated_graph_list = zip(*sorted_id_rmsd_original_generated_list)
        sorted_rmsd_list = torch.tensor(sorted_rmsd_list).cpu().numpy()
        fig, ax = plt.subplots()
        ax.plot(sorted_rmsd_list,marker='o',linestyle='None')
        ax.set_xlabel('sorted_index')
        ax.set_ylabel('rmsd')
        ax.set_yscale('log')
        ax.set_title('rmsd')
        wandb.log({'rmsd':wandb.Image(fig)})
        plt.close()
        
        #生成したグラフのrmsd,atom_type_scoreを計算し、ソート
        sorted_id_rmsd_atomeval_original_generated_list = evaluate_by_rmsd_and_atom_type_eval(original_graph_list,generated_graph_list)

        soretd_id_list, sorted_rmsd_list, sorted_atom_type_eval_list, sorted_original_graph_list, sorted_generated_graph_list = zip(*sorted_id_rmsd_atomeval_original_generated_list)
        sorted_rmsd_list = torch.tensor(sorted_rmsd_list).cpu().numpy()
        sorted_atom_type_eval_list = torch.tensor(sorted_atom_type_eval_list).cpu().numpy()
        density_of_O_for_original = [i[0] for i in sorted_atom_type_eval_list]
        density_of_O_for_generated = [i[1] for i in sorted_atom_type_eval_list]
        fig, ax = plt.subplots()
        ax.plot(density_of_O_for_original,density_of_O_for_generated,'o')
        ax.plot([0,1],[0,1],'-',color='red')
        ax.set_xlabel('density of O for original')
        ax.set_ylabel('density of O for generated')
        ax.set_title('atom_type_eval')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        wandb.log({'atom_type_eval':wandb.Image(fig)})
        plt.close()

        
        
        #xyzファイルの作成
        if args.create_xyz_file:
            first_min_rmsd_data = sorted_id_rmsd_original_generated_list[0]
            second_min_rmsd_data = sorted_id_rmsd_original_generated_list[1]
            third_min_rmsd_data = sorted_id_rmsd_original_generated_list[2]
            mid_rmsd_data = sorted_id_rmsd_original_generated_list[int(len(sorted_id_rmsd_original_generated_list)/2)]
            max_rmad_data = sorted_id_rmsd_original_generated_list[-1]
            write_xyz(os.path.join(run.dir,'first_min_rmsd.xyz'),first_min_rmsd_data[2],first_min_rmsd_data[3],comment='first_min_rmsd ' + str(first_min_rmsd_data[0]) + ' rmsd: ' + str(first_min_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'second_min_rmsd.xyz'),second_min_rmsd_data[2],second_min_rmsd_data[3],comment='second_min_rmsd ' + str(second_min_rmsd_data[0]) + ' rmsd: ' + str(second_min_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'third_min_rmsd.xyz'),third_min_rmsd_data[2],third_min_rmsd_data[3],comment='third_min_rmsd ' + str(third_min_rmsd_data[0]) + ' rmsd: ' + str(third_min_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'mid_rmsd.xyz'),mid_rmsd_data[2],mid_rmsd_data[3],comment='mid_rmsd ' + str(mid_rmsd_data[0]) + ' rmsd: ' + str(mid_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'max_rmsd.xyz'),max_rmad_data[2],max_rmad_data[3],comment='max_rmsd ' + str(max_rmad_data[0]) + ' rmsd: ' + str(max_rmad_data[1].item()))
            wandb.config.update({'rmsd_xyz_path':run.dir})
            print('xyz file created')

    #noise_scheduleの記録
    if args.record_schedule:
        fig_alpha = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'alpha')
        fig_sigma = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'sigma')
        fig_SNR = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'SNR')
        if prms['noise_schedule'] == 'learned':
            fig_gamma = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'gamma')
            wandb.log({'alpha':wandb.Image(fig_alpha),'sigma':wandb.Image(fig_sigma),'gamma':wandb.Image(fig_gamma),'SNR':wandb.Image(fig_SNR)})
            plt.close(fig_gamma)
        elif prms['noise_schedule'] == 'predefined':
            wandb.log({'alpha':wandb.Image(fig_alpha),'sigma':wandb.Image(fig_sigma),'SNR':wandb.Image(fig_SNR)})
        print('noise_schedule saved')
        plt.close(fig_alpha)
        plt.close(fig_sigma)
        plt.close(fig_SNR)

    #wandbの終了
    wandb.finish()





    
