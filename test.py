import torch, copy, itertools, random, datetime, pdb, sys, yaml, os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, f1_score
import split_to_train_and_test
from split_to_train_and_test import SetUpData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch.optim.lr_scheduler import StepLR
from E3diffusion_new import E3DiffusionProcess, remove_mean
from EquivariantGraphNeuralNetwork import EGCL, EquivariantGNN
from CN2_evaluate import calculate_angle_for_CN2
from PIL import Image, ImageFilter
from DataPreprocessor import SpectrumCompressor

def write_xyz_for_prediction_only_si(save_name,generated_coords:torch.tensor,original_coords:torch.tensor=None,mode='individual'):
    if mode == 'individual':
        file_name = '/home/rokubo/data/diffusion_model/test_vesta/individual/' + str(save_name) + '.xyz'
    elif mode == 'seed_dependence':
        file_name = '/home/rokubo/data/diffusion_model/test_vesta/seed_dependence/' + str(save_name) + '.xyz'
    N = generated_coords.shape[0]
    if original_coords is not None:
        with open(file_name,'w') as f:
            f.write(str(N*2)+'\n')
            f.write('\n')
            for i in range(N):
                if i == 0:
                    f.write('F '+str(original_coords[i][0].item())+' '+str(original_coords[i][1].item())+' '+str(original_coords[i][2].item())+'\n')
                else:
                    f.write('Al '+str(original_coords[i][0].item())+' '+str(original_coords[i][1].item())+' '+str(original_coords[i][2].item())+'\n')
            for i in range(N):
                if i == 0:
                    f.write('O '+str(generated_coords[i][0].item())+' '+str(generated_coords[i][1].item())+' '+str(generated_coords[i][2].item())+'\n')
                else:
                    f.write('Si '+str(generated_coords[i][0].item())+' '+str(generated_coords[i][1].item())+' '+str(generated_coords[i][2].item())+'\n')
    else:
        with open(file_name,'w') as f:
            f.write(str(N)+'\n')
            f.write('\n')
            for i in range(N):
                if i == 0:
                    f.write('O '+str(generated_coords[i][0].item())+' '+str(generated_coords[i][1].item())+' '+str(generated_coords[i][2].item())+'\n')
                else:
                    f.write('Si '+str(generated_coords[i][0].item())+' '+str(generated_coords[i][1].item())+' '+str(generated_coords[i][2].item())+'\n')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('device:',device)
    with open('parameters.yaml','r') as file:
        params = yaml.safe_load(file)
    
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    num_diffusion_timestep = params['num_diffusion_timestep']
    num_epochs = params['num_epochs']

    batch_size = params['batch_size']

    conditional = params['conditional']

    atom_type_size = params['atom_type_size']
    spectrum_size = params['spectrum_size']
    t_size = params['t_size']
    x_size = params['x_size']
    m_size = params['m_size']
    d_size = params['d_size']
    if conditional:
        h_size = spectrum_size + atom_type_size + t_size
    else:
        h_size = atom_type_size + t_size
    
    onehot_scaling_factor = params['onehot_scaling_factor']

    L= params['L'] #Lの値を大きくしすぎるとnanが出る（15のときに）
    lr = params['lr']
    weight_decay = params['weight_decay']

    max_grad_norm = params['max_grad_norm']

    m_input_size = h_size + h_size + d_size
    m_output_size = m_size
    m_hidden_size = params['m_hidden_size']

    h_input_size = h_size + m_size
    h_output_size = h_size
    h_hidden_size = params['h_hidden_size']


    x_input_size = h_size + h_size + d_size
    x_output_size = 1
    x_hidden_size = params['x_hidden_size']


    to_compress_spectrum = params['to_compress_spectrum']
    compressed_spectrum_size = params['compressed_spectrum_size']
    compressor_hidden_dim = params['compressor_hidden_dim']

    noise_precision = params['noise_precision']
    power = params['noise_schedule_power']
    noise_schedule = params['noise_schedule']


    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(spectrum_size,compressor_hidden_dim,compressed_spectrum_size).to(device)
        h_size = compressed_spectrum_size + atom_type_size + t_size
        m_input_size = h_size + h_size + d_size
        h_input_size = h_size + m_size
        h_output_size = h_size
        x_input_size = h_size + h_size + d_size


    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size).to(device)

    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)
    


    criterion = nn.MSELoss()

    model_path = 'egnn_202411281517'

    state_dicts = torch.load('/mnt/homenfsxx/rokubo/data/diffusion_model/model_state/model_to_predict_epsilon/'+model_path+'.pth',weights_only=True)
    egnn.load_state_dict(state_dicts['egnn'])
    egnn.eval()
    if to_compress_spectrum:
        spectrum_compressor.load_state_dict(state_dicts['spectrum_compressor'])
        spectrum_compressor.eval()
    if noise_schedule == 'learned':
        diffusion_process.gamma.load_state_dict(state_dicts['GammaNetwork'])
        diffusion_process.gamma.eval()
    setupdata = SetUpData(seed,conditional)
    
    data = np.load("/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/dataset.npy",allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)
    dataset = setupdata.resize_spectrum(dataset,resize=spectrum_size)

    dataset_only_CN2 = []
    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            dataset_only_CN2.append(dataset[i])
    dataset = dataset_only_CN2
    
    #dataset = torch.load('/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/filtered_dataset_only_Si.pt',weights_only=True)

    train_data, val_data, test_data = setupdata.split(dataset)


    """
    for i in range(len(test_data)):
        if test_data[i].id == 'mp-1244968_81':
            print(i)
    pdb.set_trace()
    """

    #theta_list = [] #theta is the angle of original
    #phi_list = [] #phi is the angle of generated

    original_coords_list = []
    generated_coords_list = []

    #test_data_subset = torch.utils.data.Subset(test_data, list(range(5)))


    if conditional:
        seed_value = 0
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            if data.spectrum.shape[0] != 3:
                continue
                
            num_of_generated_coords = 0
            num_of_generated_nun = 0

            how_many_gen = 5
            while num_of_generated_coords != how_many_gen:
                
                torch.manual_seed(seed_value)
                np.random.seed(seed_value)
                random.seed(seed_value)
                #data = train_data[913]
                #data = test_data[9]  #
                num_atom = data.spectrum.shape[0] #


                edge_index = []
                for i in range(num_atom):
                    for j in range(num_atom):
                        if i != j:
                            edge_index.append([i, j])
                initial_coords = torch.zeros(size=(num_atom,3)).to(device)
                initial_coords.normal_()
                initial_coords = initial_coords - torch.mean(initial_coords,dim=0,keepdim=True)
                atom_type = [[1,0]]
                for i in range(num_atom-1):
                    atom_type.append([0,1])
                x = torch.tensor(atom_type,dtype=torch.float32).to(device)
                graph = Data(x=x,edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(device),pos=initial_coords,spectrum=data.spectrum.to(device))#

                graph.node = graph.pos


                transition_of_coords_per_100steps = []
                egnn.eval()
                with torch.no_grad():
                    for time in list(range(num_diffusion_timestep,0,-1)):

                        if time%100 == 0:
                            transition_of_coords_per_100steps.append(graph.pos)
                        
                        time_tensor = torch.tensor([[time/num_diffusion_timestep] for d in range(num_atom)],dtype=torch.float32).to(device)
                        if to_compress_spectrum:
                            compressed_spectrum = spectrum_compressor(graph.spectrum)
                            graph.h = torch.cat((onehot_scaling_factor*graph.x,compressed_spectrum,time_tensor),dim=1)
                        else:
                            graph.h = torch.cat((onehot_scaling_factor*graph.x,graph.spectrum,time_tensor),dim=1)
                        

                        new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos)
                        epsilon = remove_mean(new_x - graph.pos)
                        mu = diffusion_process.calculate_mu(graph.pos,epsilon,time)
                        graph.pos = diffusion_process.reverse_diffuse_one_step(mu,time)

                        if not torch.isfinite(graph.pos).all():
                            #raise ValueError('nan')
                            print('nan')
                            seed_value += 1
                            num_of_generated_nun += 1
                            if num_of_generated_nun == 10:
                                print('too much nan were generated')
                                exit()
                            break
                if torch.isfinite(graph.pos).all():
                    transition_of_coords_per_100steps.append(graph.pos)
                    transition_of_coords_per_100steps = torch.stack(transition_of_coords_per_100steps)
                    transition_of_coords_per_100steps = [coords.cpu().numpy() for coords in transition_of_coords_per_100steps]
                    transition_of_coords_per_100steps = np.array(transition_of_coords_per_100steps)
                    num_of_generated_coords += 1
                    seed_value += 1
                    """
                    if num_of_generated_coords == 2:
                        print(data.pos.numpy())
                        print(transition_of_coords_per_100steps)
                        print(data.pos.numpy().shape)
                        print(transition_of_coords_per_100steps.shape)
                        print(type(transition_of_coords_per_100steps))
                        for i in transition_of_coords_per_100steps:
                            if type(i) != np.ndarray:
                                print('not ndarray')
                                print(i)
                        pdb.set_trace()
                    """   
                    original_coords_list.append(data.pos)
                    generated_coords_list.append(transition_of_coords_per_100steps)
                    print('successfully generated',num_of_generated_coords)
                
        # データの形状を確認
        #for i, (original, generated) in enumerate(zip(original_coords_list, generated_coords_list)):
            #print(f"Original {i}: {original.shape}, Generated {i}: {generated.shape}")

        # データを適切な形状に変換
        original_coords_array = np.array([coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords for coords in original_coords_list])
        generated_coords_array = np.array([coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords for coords in generated_coords_list])
        original_coords_list = original_coords_array
        generated_coords_list = generated_coords_array

        np.savez('conditional_gen_by_dataset_only_CN2_including_180_'+ model_path + '.npz',original_coords_list=original_coords_list,generated_coords_list=generated_coords_list)

    else:
        seed_value = 0
        num_of_generated_coords = 0
        how_many_gen = 1000

        while num_of_generated_coords != how_many_gen:
            
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            random.seed(seed_value)
            #data = train_data[913]
            #data = test_data[9]  #
            num_atom = data.spectrum.shape[0] #


            edge_index = []
            for i in range(num_atom):
                for j in range(num_atom):
                    if i != j:
                        edge_index.append([i, j])
            initial_coords = torch.zeros(size=(num_atom,3)).to(device)
            initial_coords.normal_()
            initial_coords = initial_coords - torch.mean(initial_coords,dim=0,keepdim=True)
            atom_type = [[1,0]]
            for i in range(num_atom-1):
                atom_type.append([0,1])
            x = torch.tensor(atom_type,dtype=torch.float32)
            graph = Data(x=x,edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(device),pos=initial_coords)
            graph.node = graph.pos


            transition_of_coords_per_100steps = []
            egnn.eval()
            with torch.no_grad():
                for time in list(range(num_diffusion_timestep,0,-1)):
                    if time%100 == 0:
                        transition_of_coords_per_100steps.append(graph.pos)
                    
                    time_tensor = torch.tensor([[time/num_diffusion_timestep] for d in range(num_atom)],dtype=torch.float32).to(device)
                    graph.h = torch.cat((onehot_scaling_factor*graph.x,time_tensor),dim=1)
                    new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos)
                    epsilon = remove_mean(new_x - graph.pos)
                    mu = diffusion_process.calculate_mu(graph.pos,epsilon,time)
                    graph.pos = diffusion_process.reverse_diffuse_one_step(mu,time)


                    if not torch.isfinite(graph.pos).all():
                        #raise ValueError('nan')
                        print('nan')
                        seed_value += 1
                        break
            if torch.isfinite(graph.pos).all():
                transition_of_coords_per_100steps.append(graph.pos)
                transition_of_coords_per_100steps = torch.stack(transition_of_coords_per_100steps)
                transition_of_coords_per_100steps = [coords.cpu().numpy() for coords in transition_of_coords_per_100steps]
                transition_of_coords_per_100steps = np.array(transition_of_coords_per_100steps)
                num_of_generated_coords += 1
                seed_value += 1
                
                original_coords_list.append(-1)
                generated_coords_list.append(transition_of_coords_per_100steps)
                print('successfully generated',num_of_generated_coords)
        np.savez('abinitio_gen_by_dataset_only_CN2_including_180_'+ model_path + '.npz',original_coords_list=original_coords_list,generated_coords_list=generated_coords_list)
"""
    for data in test_data:
        if data.spectrum.shape[0] != 3:
            continue

        if conditional is not True:
            data.id = 'abinitio'
        

        seed_value = 0
        num_of_generated_coords = 0
        if conditional:
            how_many_gen = 10
        else:
            how_many_gen = 1000
        while num_of_generated_coords != how_many_gen:
            
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            random.seed(seed_value)
            #data = train_data[913]
            #data = test_data[9]  #
            num_atom = data.spectrum.shape[0] #


            edge_index = []
            for i in range(num_atom):
                for j in range(num_atom):
                    if i != j:
                        edge_index.append([i, j])
            initial_coords = torch.zeros(size=(num_atom,3))
            initial_coords.normal_()
            initial_coords = initial_coords - torch.mean(initial_coords,dim=0,keepdim=True)
            atom_type = [[1,0]]
            for i in range(num_atom-1):
                atom_type.append([0,1])
            x = torch.tensor(atom_type,dtype=torch.float32)
            if conditional:
                graph = Data(x=x,edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous(),pos=initial_coords,spectrum=data.spectrum)#
            else:
                graph = Data(x=x,edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous(),pos=initial_coords)
            graph.node = graph.pos


            record_time = list(range(10,num_diffusion_timestep+10,10))
            #record_time = [None]
            record_mode = 'individual'
            egnn.eval()
            with torch.no_grad():
                for time in list(range(num_diffusion_timestep,0,-1)):
                    
                    if time%100 == 0:
                        print('coords at time',time,':',graph.pos)
                    
                
                    time_tensor = torch.tensor([[time/num_diffusion_timestep] for d in range(num_atom)],dtype=torch.float32)
                    if conditional:
                        if to_compress_spectrum:
                            compressed_spectrum = spectrum_compressor(graph.spectrum)
                            graph.h = torch.cat((onehot_scaling_factor*graph.x,compressed_spectrum,time_tensor),dim=1)
                        else:
                            graph.h = torch.cat((onehot_scaling_factor*graph.x,graph.spectrum,time_tensor),dim=1)
                    else:
                        graph.h = torch.cat((onehot_scaling_factor*graph.x,time_tensor),dim=1)
                    if params['diffusion_process'] == 'GeoDiff':
                        new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos,graph.node)
                        epsilon = diffusion_process.equivaliant_epsilon_torch(new_x,graph.node,time)
                        mu = diffusion_process.calculate_mu(graph.node,epsilon,time)
                        graph.pos = diffusion_process.calculate_onestep_before(mu,time)
                        graph.node = graph.pos
                    elif params['diffusion_process'] == 'E3':
                        new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos)
                        
                        if time in record_time:
                            if record_mode == 'individual':
                                os.makedirs('/home/rokubo/data/diffusion_model/test_vesta/individual/'+str(data.id)+'_'+model_path,exist_ok=True)
                                save_name = str(data.id)+'_'+model_path + '/' + str(data.id) + '_' + str(time)
                            elif record_mode == 'seed_dependence':
                                os.makedirs('/home/rokubo/data/diffusion_model/test_vesta/seed_dependence/'+str(data.id)+'_'+model_path,exist_ok=True)
                                save_name = str(data.id)+'_'+model_path + '/' + 'seed' + str(seed_value) + '_' + str(data.id) + '_' + str(time)
                            if conditional:
                                write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=data.pos,mode=record_mode)
                            else:
                                write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=None,mode=record_mode)
                        
                        #print('time:',time)
                        #print('new_x:',new_x)
                        #print('graph.pos:',graph.pos)
                        epsilon = remove_mean(new_x - graph.pos)
                        #print('epsilon',epsilon)
                        mu = diffusion_process.calculate_mu(graph.pos,epsilon,time)
                        graph.pos = diffusion_process.reverse_diffuse_one_step(mu,time)

                    
                    if time%100 == 0:
                        print('graph.pos',graph.pos)
                        print('epsilon:',epsilon)
                    
                    if not torch.isfinite(graph.pos).all():
                        #raise ValueError('nan')
                        print('nan')
                        seed_value += 1
                        break
            if torch.isfinite(graph.pos).all():
                num_of_generated_coords += 1
                seed_value += 1
                
                if conditional:
                    original_coords_list.append(data.pos)
                else:
                    original_coords_list.append(-1)
                generated_coords_list.append(graph.pos)
                print('successfully generated',num_of_generated_coords)
                
                if record_mode == 'individual':
                    save_name = str(data.id)+'_'+model_path + '/' + str(data.id) + '_0'
                elif record_mode == 'seed_dependence':
                    save_name = str(data.id)+'_'+model_path + '/' + 'seed' + str(seed_value) + '_' + str(data.id) + '_0'
                if conditional:
                    write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=data.pos,mode=record_mode)
                else:
                    write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=None,mode=record_mode)
                
            
            print('graph.id:',data.id)
            print('coords at time 0:',graph.pos)
            print(data.pos)
            
            if record_mode == 'individual':
                save_name = str(data.id)+'_'+model_path + '/' + str(data.id) + '_0'
            elif record_mode == 'seed_dependence':
                save_name = str(data.id)+'_'+model_path + '/' + 'seed' + str(seed_value) + '_' + str(data.id) + '_0'
            if conditional:
                write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=data.pos,mode=record_mode)
            else:
                write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=None,mode=record_mode)        
            
    if conditional:
        for_name = 'conditional'
    else:
        for_name = 'abinitio'
    np.savez(for_name+'_gen_by_dataset_only_CN2_including_180_'+ model_path + '.npz',original_coords_list=original_coords_list,generated_coords_list=generated_coords_list)
"""

