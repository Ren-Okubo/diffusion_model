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
#from E3diffusion import E3DiffusionProcess, remove_mean
from E3diffusion_new import E3DiffusionProcess, remove_mean
from CN2_evaluate import calculate_angle_for_CN2
from DataPreprocessor import SpectrumCompressor
import wandb
from kabsch_algorithm import kabsch_torch

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
    print(f"device: {device}")
    with open('parameters.yaml','r') as file:
        params = yaml.safe_load(file)

    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    
    params['now'] = now.strftime("%Y%m%d%H%M")

    wandb.init(project='coords_loss,noise_loss',config=params,name='conditional dataset only CN2 except 180')
    
    seed = params['seed']
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_precision = params['noise_precision']
    power = params['noise_schedule_power']
    num_epochs = params['num_epochs']
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep)
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
    """
    data = np.load("/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/dataset.npy",allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)
    dataset = setupdata.resize_spectrum(dataset=dataset,resize=spectrum_size)

    dataset_only_CN2 = []
    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            if calculate_angle_for_CN2(dataset[i].pos) < 179:
                dataset_only_CN2.append(dataset[i])
    dataset = dataset_only_CN2
    """
    dataset = torch.load('/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/filtered_dataset_only_Si.pt')

    train_data, val_data, test_data = setupdata.split(dataset)

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

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
    
    optimizer = optim.Adam(egnn.parameters(),lr=lr,weight_decay=weight_decay)
    

    criterion = nn.MSELoss(reduction='sum')

    epoch_list, loss_list_train, loss_list_val = [], [], []

    for epoch in range(num_epochs):
        egnn.train()
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
            loss_noise = criterion(epsilon,train_graph.y)
            loss_noise = loss_noise / num_graph
            batch_loss_coords = 0
            for i in range(num_graph):
                graph_index = i
                edge_index = []
                num_atom = train_graph.x[train_graph.batch == graph_index].shape[0]
                pos_to_gen = torch.zeros_like(train_graph.pos[train_graph.batch == graph_index]).to(device)
                pos_to_gen.normal_(mean=0,std=1)
                pos_to_gen = remove_mean(pos_to_gen)
                for i in range(num_atom):
                    for j in range(num_atom):
                        if i != j:
                            edge_index.append([i, j])
                graph_data = Data(x=train_graph.x[train_graph.batch == graph_index],edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(device),pos=pos_to_gen)
                for time in list(range(num_diffusion_timestep,0,-1)):
                    graph_data.time = torch.tensor([[time/num_diffusion_timestep] for j in range(graph_data.x.shape[0])],dtype=torch.float32).to(device)
                    if conditional:
                        graph_data.spectrum = train_graph.spectrum[train_graph.batch == graph_index].to(device)
                        if to_compress_spectrum:
                            graph_data.spectrum = spectrum_compressor(graph_data.spectrum)
                        graph_data.h = torch.cat((onehot_scaling_factor*graph_data.x,graph_data.spectrum,graph_data.time),dim=1)
                    else:
                        graph_data.h = torch.cat((onehot_scaling_factor*graph_data.x,graph_data.time),dim=1)
                    h, x = egnn(graph_data.edge_index,graph_data.h,graph_data.pos)
                    epsilon = x - pos_to_gen
                    epsilon = remove_mean(epsilon)
                    mu = diffusion_process.calculate_mu(pos_to_gen,epsilon,time)
                    pos_to_gen = diffusion_process.reverse_diffuse_one_step(mu,time)
                    graph_data.pos = pos_to_gen
                    assert torch.isfinite(graph_data.pos).all(), 'nan or inf in graph_data.pos'
                _, _, loss_coords = kabsch_torch(pos_to_gen,train_graph.pos[train_graph.batch == graph_index])
                batch_loss_coords += loss_coords
            loss_coords = batch_loss_coords / num_graph
            loss = loss_noise + loss_coords
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model_x.parameters(), max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(model_h.parameters(), max_grad_norm)
            optimizer.step()
            loss_noise = loss_noise * num_graph
            epoch_loss_train += loss_noise.item()



        

        egnn.eval()
        
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

                loss_noise = criterion(epsilon,val_graph.y)
                loss_coords = 0
                for i in range(num_graph):
                    graph_index = i
                    edge_index = []
                    num_atom = val_graph.x[val_graph.batch == graph_index].shape[0]
                    pos_to_gen = torch.zeros_like(val_graph.pos[val_graph.batch == graph_index]).to(device)
                    pos_to_gen.normal_(mean=0,std=1)
                    pos_to_gen = remove_mean(pos_to_gen)
                    for i in range(num_atom):
                        for j in range(num_atom):
                            if i != j:
                                edge_index.append([i, j])
                    graph_data = Data(x=val_graph.x[val_graph.batch == graph_index],edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(device),pos=pos_to_gen)
                    for time in list(range(num_diffusion_timestep,0,-1)):
                        graph_data.time = torch.tensor([[time/num_diffusion_timestep] for j in range(graph_data.x.shape[0])],dtype=torch.long).to(device)
                        if conditional:
                            graph_data.spectrum = val_graph.spectrum[val_graph.batch == graph_index].to(device)
                            if to_compress_spectrum:
                                graph_data.spectrum = spectrum_compressor(graph_data.spectrum)
                            graph_data.h = torch.cat((onehot_scaling_factor*graph_data.x,graph_data.spectrum,graph_data.time),dim=1)
                        else:
                            graph_data.h = torch.cat((onehot_scaling_factor*graph_data.x,graph_data.time),dim=1)
                        h, x = egnn(graph_data.edge_index,graph_data.h,graph_data.pos)
                        epsilon = x - pos_to_gen
                        epsilon = remove_mean(epsilon)
                        mu = diffusion_process.calculate_mu(pos_to_gen,epsilon,time)
                        pos_to_gen = diffusion_process.reverse_diffuse_one_step(mu,time)
                        graph_data.pos = pos_to_gen
                        if not torch.isfinite(graph_data.pos).all():
                            wandb.run.notes = 'nan or inf in graph_data.pos'
                            assert torch.isfinite(graph_data.pos).all(), 'nan or inf in graph_data.pos'
                    _, _, rmsd = kabsch_torch(pos_to_gen,val_graph.pos[val_graph.batch == graph_index])
                    loss_coords += rmsd
                loss = loss_noise + loss_coords
                epoch_loss_val += loss.item()
        avg_loss_val = epoch_loss_val / total_num_val_node
        avg_loss_train = epoch_loss_train / total_num_train_node


        epoch_list.append(epoch)
        loss_list_val.append(avg_loss_val)
        loss_list_train.append(avg_loss_train)
        print("epoch : ",epoch,"    loss_train : ",avg_loss_train,"    loss_val : ",avg_loss_val)
        wandb.log({"loss_train":avg_loss_train,"loss_val":avg_loss_val})
        if early_stopping.validate(avg_loss_train):
            break
    
    model_state1 = egnn.state_dict()
    if to_compress_spectrum:
        model_state2 = spectrum_compressor.state_dict()
        model_states = {'egnn':model_state1,'spectrum_compressor':model_state2}
    else:
        model_states = {'egnn':model_state1}
    torch.save(model_states,"./model_state/model_to_predict_epsilon/egnn_"+now.strftime("%Y%m%d%H%M")+".pth")
    
    