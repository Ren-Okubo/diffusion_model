import torch, copy, itertools, random, datetime, pdb, sys, yaml, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, f1_score
from diffusion_process import DiffusionProcess
import split_to_train_and_test
from split_to_train_and_test import SetUpData
from GeoDiffEGNN import GCL, EGNN, EquivariantEpsilon
from CN import MLP
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch.optim.lr_scheduler import StepLR
from E3diffusion_new import E3DiffusionProcess, remove_mean
from EquivariantGraphNeuralNetwork import EGCL, EquivariantGNN
from angle_evaluate import calculate_angle_for_CN2

def write_xyz_for_prediction_only_si(save_name,generated_coords:torch.tensor,original_coords:torch.tensor=None,mode='individual'):
    if mode == 'individual':
        file_name = '/home/rokubo/data/diffusion_model/test_vesta/individual/' + str(save_name) + '.xyz'
    elif mode == 'seed_dependence':
        file_name = '/home/rokubo/data/diffusion_model/test_vesta/seed_dependence/' + str(save_name) + '.xyz'
    N = original_coords.shape[0]
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
    with open('parameters.yaml','r') as file:
        params = yaml.safe_load(file)
    
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_precision = params['noise_precision']
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

    if params['mlp_x_input'] == 'E3':
        x_input_size = h_size + h_size + d_size
    else:
        x_input_size = m_size
    x_output_size = 1
    x_hidden_size = params['x_hidden_size']

    epsilon_prediction = params['epsilon_prediction']
    

    if epsilon_prediction == 'GeoDiff':
        egnn = EGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    elif epsilon_prediction == 'E3':
        egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    
    if params['diffusion_process'] == 'GeoDiff':
        diffusion_process = DiffusionProcess(initial_beta,final_beta,num_diffusion_timestep)
        equivariant_epsilon = EquivariantEpsilon(initial_beta,final_beta,num_diffusion_timestep)
    elif params['diffusion_process'] == 'E3':
        diffusion_process = E3DiffusionProcess(s=noise_precision,num_diffusion_timestep=num_diffusion_timestep)
    


    criterion = nn.MSELoss()

    model_path = 'egnn_202410021731'

    checkpoint = torch.load('/home/rokubo/data/diffusion_model/model_state/model_to_predict_epsilon/'+model_path+'.pth')

    egnn.load_state_dict(checkpoint)

    setupdata = SetUpData(seed,conditional)

    data = np.load("/home/rokubo/data/diffusion_model/dataset/dataset.npy",allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)

    dataset_only_not_pi = []
    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            if calculate_angle_for_CN2(dataset[i].pos) < 179:
                dataset_only_not_pi.append(dataset[i])
    dataset = dataset_only_not_pi


    train_data, val_data, test_data = setupdata.split(dataset)

    

    theta_list = [] #theta is the angle of original
    phi_list = [] #phi is the angle of generated

    for data in test_data:
        if data.spectrum.shape[0] != 3:
            continue
        

        seed_value = 0
        num_of_generated_coords = 0
        while num_of_generated_coords != 10:
            
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


            #record_time = list(range(100,num_diffusion_timestep+100,100))
            #record_time = [None]
            #record_mode = 'individual'
            egnn.eval()
            with torch.no_grad():
                for time in list(range(num_diffusion_timestep,0,-1)):
                    """
                    if time%100 == 0:
                        print('coords at time',time,':',graph.pos)
                    """
                
                    time_tensor = torch.tensor([[time/num_diffusion_timestep] for d in range(num_atom)],dtype=torch.float32)
                    if conditional:
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
                        """
                        if time in record_time:
                            if record_mode == 'individual':
                                os.makedirs('/home/rokubo/data/diffusion_model/test_vesta/individual/'+str(data.id)+'_'+model_path,exist_ok=True)
                                save_name = str(data.id)+'_'+model_path + '/' + str(data.id) + '_' + str(time)
                            elif record_mode == 'seed_dependence':
                                os.makedirs('/home/rokubo/data/diffusion_model/test_vesta/seed_dependence/'+str(data.id)+'_'+model_path,exist_ok=True)
                                save_name = str(data.id)+'_'+model_path + '/' + 'seed' + str(seed_value) + '_' + str(data.id) + '_' + str(time)
                            write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=data.pos,mode=record_mode)
                        """    
                        #print('time:',time)
                        #print('new_x:',new_x)
                        #print('graph.pos:',graph.pos)
                        epsilon = remove_mean(new_x - graph.pos)
                        #print('epsilon',epsilon)
                        mu = diffusion_process.calculate_mu(graph.pos,epsilon,time)
                        graph.pos = diffusion_process.reverse_diffuse_one_step(mu,time)

                    """
                    if time%100 == 0:
                        print('graph.pos',graph.pos)
                        print('epsilon:',epsilon)
                    """
                    if not torch.isfinite(graph.pos).all():
                        #raise ValueError('nan')
                        print('nan')
                        seed_value += 1
                        break
            if torch.isfinite(graph.pos).all():
                num_of_generated_coords += 1
                seed_value += 1
                theta_list.append(calculate_angle_for_CN2(data.pos))
                phi_list.append(calculate_angle_for_CN2(graph.pos))
            
            """
            print('graph.id:',data.id)
            print('coords at time 0:',graph.pos)
            print(data.pos)
            
            if record_mode == 'individual':
                save_name = str(data.id)+'_'+model_path + '/' + str(data.id) + '_0'
            elif record_mode == 'seed_dependence':
                save_name = str(data.id)+'_'+model_path + '/' + 'seed' + str(seed_value) + '_' + str(data.id) + '_0'
            write_xyz_for_prediction_only_si(save_name,generated_coords=graph.pos,original_coords=data.pos,mode=record_mode)
            """

    np.savez('angle_comparison_between_original_and_generated_except_180.npz',theta_list=theta_list,phi_list=phi_list)
    plt.plot(theta_list,phi_list,'o')
    plt.plot([0,180],[0,180])
    plt.xlim(70,180)
    plt.ylim(70,180)
    plt.xlabel('theta')
    plt.ylabel('phi')
    plt.savefig('angle_comparison_between_original_and_generated_except_180.png')
    plt.close()

