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
from torch.optim.lr_scheduler import StepLR
from schedulefree import RAdamScheduleFree
#from E3diffusion import E3DiffusionProcess, remove_mean
from E3diffusion_new import E3DiffusionProcess, remove_mean
from CN2_evaluate import calculate_angle_for_CN2
from DataPreprocessor import SpectrumCompressor
import wandb
import argparse

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
        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f"device: {device}")

    torch.autograd.set_detect_anomaly(True)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_each_epoch',type=bool,default=False)
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_name', type=str, default=None)
    #argparser.add_argument('--dataset', type=str, required=True) #dataset, dataset_only_CN2_Si, filtered_dataset, filtered_dataset_only_CN2_Si, spectrum_to_only_exO_dataset_except_CN0
    argparser.add_argument('--dataset_path', type=str, required=True)
    args = argparser.parse_args()

    with open('parameters.yaml','r') as file:
        params = yaml.safe_load(file)

    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    
    params['now'] = now.strftime("%Y%m%d%H%M")

    if params['noise_schedule'] == 'learned':
        del params['noise_precision']
        del params['noise_schedule_power']

    wandb.init(project=args.project_name,config=params,name=args.run_name)
    
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_schedule = params['noise_schedule']
    if noise_schedule == 'predefined':
        noise_precision = params['noise_precision']
        power = params['noise_schedule_power']
    elif noise_schedule == 'learned':
        noise_precision = None
        power = None

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

    
    early_stopping = EarlyStopping(patience=params['patience'])
    message_passing = MessagePassing(aggr='sum',flow='target_to_source')
    setupdata = SetUpData(seed=seed,conditional=conditional)
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)

    """
    data = np.load("/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/dataset.npy",allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)
    dataset = setupdata.resize_spectrum(dataset=dataset,resize=spectrum_size)

    dataset_only_CN2 = []
    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            #if calculate_angle_for_CN2(dataset[i].pos) < 179:
                #dataset_only_CN2.append(dataset[i])
            dataset_only_CN2.append(dataset[i])
    dataset = dataset_only_CN2
    """

    #dataset = torch.load('/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset_only_CN2_Si.pt')
    """
    dataset_name = args.dataset
    dataset = torch.load(f'/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/{dataset_name}.pt')
    wandb.config.update({'dataset':dataset_name})
    """
    dataset = torch.load(args.dataset_path)
    wandb.config.update({'dataset_path':args.dataset_path})

    train_data, val_data, test_data = setupdata.split(dataset)

    train_dataset = []
    for data in train_data:
        if data.pos.shape[0] == 1:
            continue
        else:
            train_dataset.append(data)
    train_data = train_dataset
    eval_dataset = []
    for data in val_data:
        if data.pos.shape[0] == 1:
            continue
        else:
            eval_dataset.append(data)
    val_data = eval_dataset


    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,generator=torch.Generator(device='cuda'))
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True,generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True,generator=torch.Generator(device='cuda'))

    time_list = [i for i in range(1,num_diffusion_timestep+1)]


    #train
    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(spectrum_size,compressor_hidden_dim,compressed_spectrum_size).to(device)
        h_size = compressed_spectrum_size + atom_type_size + t_size
        m_input_size = h_size + h_size + d_size
        h_input_size = h_size + m_size
        h_output_size = h_size
        x_input_size = h_size + h_size + d_size


    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size).to(device)
    
    if to_compress_spectrum:
        if noise_schedule == 'learned':
            param_list_for_optim = list(egnn.parameters())+list(spectrum_compressor.parameters())+list(diffusion_process.gamma.parameters())
        else:
            param_list_for_optim = list(egnn.parameters())+list(spectrum_compressor.parameters())
    else:
        if noise_schedule == 'learned':
            param_list_for_optim = list(egnn.parameters())+list(diffusion_process.gamma.parameters())
        else:
            param_list_for_optim = list(egnn.parameters())

    #optimizer = optim.Adam(param_list_for_optim,lr=lr,weight_decay=weight_decay)
    optimizer = RAdamScheduleFree(param_list_for_optim,lr=lr,weight_decay=weight_decay)

    """
    if to_compress_spectrum:
        if noise_schedule == 'learned':
            optimizer = optim.Adam(list(egnn.parameters())+list(spectrum_compressor.parameters())+list(diffusion_process.gamma.parameters()),lr=lr,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(list(egnn.parameters())+list(spectrum_compressor.parameters()),lr=lr,weight_decay=weight_decay)
    else:
        if noise_schedule == 'learned':
            optimizer = optim.Adam(list(egnn.parameters())+list(diffusion_process.gamma.parameters()),lr=lr,weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(egnn.parameters(),lr=lr,weight_decay=weight_decay)
    """

    

    criterion = nn.MSELoss(reduction='sum')

    epoch_list, loss_list_train, loss_list_val = [], [], []

    for epoch in range(num_epochs):
        egnn.train()
        optimizer.train()
        if to_compress_spectrum:
            spectrum_compressor.train()
        if noise_schedule == 'learned':
            diffusion_process.gamma.train()
        epoch_loss_val = 0
        epoch_loss_train = 0
        total_num_train_node = 0
        total_num_val_node = 0
        
        for train_graph in train_loader:
            train_graph = train_graph.to(device)
            optimizer.zero_grad()
            num_graph = train_graph.batch.max().item()+1
            total_num_train_node += train_graph.num_nodes
            diffused_pos = []
            h_list = []
            y = []
            attr_time_list = []
            for i in range(num_graph):
                graph_index = i
                pos_to_diffuse = train_graph.pos[train_graph.batch == graph_index]
                x_per_graph = train_graph.x[train_graph.batch == graph_index]
                time = random.choice(time_list)
                attr_time_list += [time for j in range(x_per_graph.shape[0])]
                time_tensor = torch.tensor([[time/num_diffusion_timestep] for j in range(x_per_graph.shape[0])],dtype=torch.float32).to(device)
                if conditional:
                    spectrum_per_graph = train_graph.spectrum[train_graph.batch == graph_index].to(device)
                    if to_compress_spectrum:
                        spectrum_per_graph = spectrum_compressor(spectrum_per_graph)
                    h_per_graph = torch.cat((onehot_scaling_factor*x_per_graph,spectrum_per_graph,time_tensor),dim=1)
                else:
                    h_per_graph = torch.cat((onehot_scaling_factor*x_per_graph,time_tensor),dim=1) 
                h_list.append(h_per_graph)

                pos_after_diffusion, noise = diffusion_process.diffuse_zero_to_t(pos_to_diffuse,time)
                diffused_pos.append(pos_after_diffusion)
                y.append(noise)

            diffused_pos = torch.cat(diffused_pos,dim=0).to(device)
            h = torch.cat(h_list,dim=0).to(device)
            y = torch.cat(y,dim=0).to(device)
            train_graph.diffused_coords = diffused_pos
            train_graph.h = h
            train_graph.y = y
            train_graph.pos = diffused_pos
            train_graph.time = torch.tensor(attr_time_list,dtype=torch.long).to(device)
            
            h, x = egnn(train_graph.edge_index,train_graph.h,train_graph.diffused_coords)
            epsilon = x - train_graph.diffused_coords
            epsilon = remove_mean(epsilon,batch_index=train_graph.batch)

            #print('epsilon : ',epsilon)
            loss = criterion(epsilon,train_graph.y)
            loss = loss / num_graph
            loss.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(model_x.parameters(), max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(model_h.parameters(), max_grad_norm)
            optimizer.step()
            loss = loss * num_graph
            epoch_loss_train += loss.item()
        

        egnn.eval()
        optimizer.eval()
        if to_compress_spectrum:
            spectrum_compressor.eval()
        if noise_schedule == 'learned':
            diffusion_process.gamma.eval()
        
        with torch.no_grad():
            for val_graph in val_loader:
                val_graph = val_graph.to(device)
                optimizer.zero_grad()
                num_graph = val_graph.batch.max().item()+1
                total_num_val_node += val_graph.num_nodes
                diffused_pos = []
                h_list = []
                y = []
                attr_time_list = []
                for i in range(num_graph):
                    graph_index = i
                    pos_to_diffuse = val_graph.pos[val_graph.batch == graph_index]
                    x_per_graph = val_graph.x[val_graph.batch == graph_index]
                    time = random.choice(time_list)
                    attr_time_list += [time for j in range(x_per_graph.shape[0])]
                    time_tensor = torch.tensor([[time/num_diffusion_timestep] for j in range(x_per_graph.shape[0])],dtype=torch.float32).to(device)
                    if conditional:
                        spectrum_per_graph = val_graph.spectrum[val_graph.batch == graph_index].to(device)
                        if to_compress_spectrum:
                            spectrum_per_graph = spectrum_compressor(spectrum_per_graph)
                        h_per_graph = torch.cat((onehot_scaling_factor*x_per_graph,spectrum_per_graph,time_tensor),dim=1)
                    else:
                        h_per_graph = torch.cat((onehot_scaling_factor*x_per_graph,time_tensor),dim=1) 
                    h_list.append(h_per_graph)

                    pos_after_diffusion, noise = diffusion_process.diffuse_zero_to_t(pos_to_diffuse,time)
                    diffused_pos.append(pos_after_diffusion)
                    y.append(noise)

                diffused_pos = torch.cat(diffused_pos,dim=0).to(device)
                h = torch.cat(h_list,dim=0).to(device)
                y = torch.cat(y,dim=0).to(device)
                val_graph.diffused_coords = diffused_pos
                val_graph.h = h
                val_graph.y = y
                val_graph.pos = diffused_pos
                val_graph.time = torch.tensor(attr_time_list,dtype=torch.long).to(device)
                


                h, x = egnn(val_graph.edge_index,val_graph.h,val_graph.diffused_coords)
                epsilon = x - val_graph.diffused_coords
                epsilon = remove_mean(epsilon,batch_index=val_graph.batch)

                loss = criterion(epsilon,val_graph.y)
                epoch_loss_val += loss.item()
        avg_loss_val = epoch_loss_val / total_num_val_node
        avg_loss_train = epoch_loss_train / total_num_train_node


        epoch_list.append(epoch)
        loss_list_val.append(avg_loss_val)
        loss_list_train.append(avg_loss_train)
        if args.output_each_epoch:
            print("epoch : ",epoch,"    loss_train : ",avg_loss_train,"    loss_val : ",avg_loss_val)
        wandb.log({"loss_train":avg_loss_train,"loss_val":avg_loss_val})
        if early_stopping.validate(avg_loss_train):
            break
    
    model_states = {}
    model_state1 = egnn.state_dict()
    model_states.update({'egnn': model_state1})
    if to_compress_spectrum:
        model_state2 = spectrum_compressor.state_dict()
        model_states.update({'spectrum_compressor':model_state2})
    if noise_schedule == 'learned':
        model_state3 = diffusion_process.gamma.state_dict()
        model_states.update({'GammaNetwork':model_state3})
    
    torch.save(model_states,"./model_state/model_to_predict_epsilon/egnn_"+now.strftime("%Y%m%d%H%M")+".pth")
    
    
