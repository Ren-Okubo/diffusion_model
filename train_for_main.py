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




def train_for_main(parameters_yaml_file:str, wandb_project_name:str, wandb_run_name:str, dataset_path:str, only_train:bool):

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

    with open(parameters_yaml_file,'r') as file:
        params = yaml.safe_load(file)

    jst = pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    
    params['now'] = now.strftime("%Y%m%d%H%M")

    run = wandb.init(project=wandb_project_name,config=params,name=wandb_run_name)
    
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_precision = params['noise_precision']
    num_epochs = params['num_epochs']
    if params['diffusion_process'] == 'GeoDiff':
        diffusion_process = DiffusionProcess(initial_beta,final_beta,num_diffusion_timestep,schedule_func=schedule_func)
        equivariant_epsilon = EquivariantEpsilon(initial_beta,final_beta,num_diffusion_timestep)
    elif params['diffusion_process'] == 'E3':
        diffusion_process = E3DiffusionProcess(s=noise_precision,num_diffusion_timestep=num_diffusion_timestep)
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

    compressed_spectrum_dim = params['compressed_spectrum_dim']
    compressor_hidden_dim = params['compressor_hidden_dim']

    
    early_stopping = EarlyStopping(patience=params['patience'])
    message_passing = MessagePassing(aggr='sum',flow='target_to_source')
    setupdata = SetUpData(seed=seed,conditional=conditional)

    data = np.load(dataset_path,allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)
    dataset = setupdata.resize_spectrum(dataset=dataset,resize=spectrum_size)

    dataset_only_CN2 = []
    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            if calculate_angle_for_CN2(dataset[i].pos) < 179:
                dataset_only_CN2.append(dataset[i])
    dataset = dataset_only_CN2
    

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

    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    optimizer = optim.Adam(egnn.parameters(),lr=lr,weight_decay=weight_decay)
    

    criterion = nn.MSELoss(reduction='sum')

    epoch_list, loss_list_train, loss_list_val = [], [], []

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    for epoch in range(num_epochs):
        egnn.train()
        epoch_loss_val = 0
        epoch_loss_train = 0
        total_num_train_node = 0
        total_num_val_node = 0
        
        for train_graph in train_loader:
            optimizer.zero_grad()
            num_graph = train_graph.batch.max().item()+1
            total_num_train_node += train_graph.num_nodes
            pos_before_diffusion, h_before_diffusion = [], []
            diffused_pos, diffused_h = [], []
            y_for_noise_pos,y_for_noise_h = [], []            
            attr_time_list = []
            each_time_list = []
            for i in range(num_graph):
                graph_index = i
                #バッチ内のグラフごとに処理
                pos_to_diffuse = train_graph.pos[train_graph.batch == graph_index]
                x_per_graph = train_graph.x[train_graph.batch == graph_index]
                #conditionalの場合は特徴量ベクトルにスペクトルベクトルを連結
                if conditional:
                    spectrum_per_graph = train_graph.spectrum[train_graph.batch == graph_index]
                    h_to_diffuse = torch.cat((onehot_scaling_factor*x_per_graph,spectrum_per_graph),dim=1)
                else:
                    h_to_diffuse = onehot_scaling_factor*x_per_graph
                
                time = random.choice(time_list)
                #time_tensor = torch.tensor([[time/num_diffusion_timestep] for j in range(x_per_graph.shape[0])],dtype=torch.float32)
                
                pos_after_diffusion, noise_pos = diffusion_process.diffuse_zero_to_t(pos_to_diffuse,time)
                h_after_diffusion , noise_h = diffusion_process.diffusion_zero_to_t(h_to_diffuse,time,mode='h')
                

                diffused_pos.append(pos_after_diffusion)
                diffused_h.append(h_after_diffusion)
                pos_before_diffusion.append(pos_to_diffuse)
                h_before_diffusion.append(h_to_diffuse)
                each_time_list += [time for j in range(x_per_graph.shape[0])]
                #each_time_list.append(time_tensor)                
                y_for_noise_pos.append(noise_pos)
                y_for_noise_h.append(noise_h)


            diffused_pos = torch.cat(diffused_pos,dim=0)
            diffuse_h = torch.cat(diffused_h,dim=0)
            each_time_in_batch = torch.cat(each_time_list,dim=0)
            y_for_pos = torch.cat(y_for_noise_pos,dim=0)
            y_for_h = torch.cat(y_for_noise_h,dim=0)

            train_graph.diffused_pos = diffused_pos
            train_graph.diffused_h = diffused_h

            train_graph.pos = pos_before_diffusion
            train_graph.h = h_before_diffusion
            train_graph.y_for_pos = y_for_pos
            train_graph.y_for_h = y_for_h
            train_graph.time = torch.tensor(each_time_list,dtype=torch.long)
            
            h, x = egnn(train_graph.edge_index,torch.cat((train_graph.diffused_h,(train_graph.time/num_diffusion_timestep)),dim=1),train_graph.diffused_pos)
            epsilon_x = x - train_graph.diffused_pos
            epsilon_x = remove_mean(epsilon_x,batch_index=train_graph.batch)
            epsilon_h = h

            predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1)
            target_epsilon = torch.cat((train_graph.y_for_pos,train_graph.y_for_h),dim=1)


            #print('epsilon : ',epsilon)
            loss = criterion(predicted_epsilon,target_epsilon)
            loss = loss / num_graph
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model_x.parameters(), max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(model_h.parameters(), max_grad_norm)
            optimizer.step()
            loss = loss * num_graph
            epoch_loss_train += loss.item()
        

        egnn.eval()
        
        with torch.no_grad():
            for val_graph in val_loader:
                optimizer.zero_grad()
                num_graph = val_graph.batch.max().item()+1
                total_num_val_node += val_graph.num_nodes
                pos_before_diffusion, h_before_diffusion = [], []
                diffused_pos, diffused_h = [], []
                y_for_noise_pos,y_for_noise_h = [], []            
                attr_time_list = []
                each_time_list = []
                for i in range(num_graph):
                    graph_index = i
                    #バッチ内のグラフごとに処理
                    pos_to_diffuse = val_graph.pos[val_graph.batch == graph_index]
                    x_per_graph = val_graph.x[val_graph.batch == graph_index]
                    #conditionalの場合は特徴量ベクトルにスペクトルベクトルを連結
                    if conditional:
                        spectrum_per_graph = val_graph.spectrum[val_graph.batch == graph_index]
                        h_to_diffuse = torch.cat((onehot_scaling_factor*x_per_graph,spectrum_per_graph),dim=1)
                    else:
                        h_to_diffuse = onehot_scaling_factor*x_per_graph
                    
                    time = random.choice(time_list)
                    #time_tensor = torch.tensor([[time/num_diffusion_timestep] for j in range(x_per_graph.shape[0])],dtype=torch.float32)
                    
                    pos_after_diffusion, noise_pos = diffusion_process.diffuse_zero_to_t(pos_to_diffuse,time)
                    h_after_diffusion , noise_h = diffusion_process.diffusion_zero_to_t(h_to_diffuse,time,mode='h')
                    

                    diffused_pos.append(pos_after_diffusion)
                    diffused_h.append(h_after_diffusion)
                    pos_before_diffusion.append(pos_to_diffuse)
                    h_before_diffusion.append(h_to_diffuse)
                    each_time_list += [time for j in range(x_per_graph.shape[0])]
                    #each_time_list.append(time_tensor)                
                    y_for_noise_pos.append(noise_pos)
                    y_for_noise_h.append(noise_h)


                diffused_pos = torch.cat(diffused_pos,dim=0)
                diffuse_h = torch.cat(diffused_h,dim=0)
                each_time_in_batch = torch.cat(each_time_list,dim=0)
                y_for_pos = torch.cat(y_for_noise_pos,dim=0)
                y_for_h = torch.cat(y_for_noise_h,dim=0)

                val_graph.diffused_pos = diffused_pos
                val_graph.diffused_h = diffused_h

                val_graph.pos = pos_before_diffusion
                val_graph.h = h_before_diffusion
                val_graph.y_for_pos = y_for_pos
                val_graph.y_for_h = y_for_h
                val_graph.time = torch.tensor(each_time_list,dtype=torch.long)
                
                h, x = egnn(val_graph.edge_index,torch.cat((val_graph.diffused_h,(val_graph.time/num_diffusion_timestep)),dim=1),val_graph.diffused_pos)
                epsilon_x = x - val_graph.diffused_pos
                epsilon_x = remove_mean(epsilon_x,batch_index=val_graph.batch)
                epsilon_h = h

                predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1)
                target_epsilon = torch.cat((val_graph.y_for_pos,val_graph.y_for_h),dim=1)


                #print('epsilon : ',epsilon)
                loss = criterion(predicted_epsilon,target_epsilon)
                loss = loss / num_graph
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model_x.parameters(), max_grad_norm)
                #torch.nn.utils.clip_grad_norm_(model_h.parameters(), max_grad_norm)
                optimizer.step()
                loss = loss * num_graph
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
    

    model_state_name = now.strftime("%Y%m%d%H%M")
    model_state1 = egnn.state_dict()
    if to_compress_spectrum:
        model_state2 = spectrum_compressor.state_dict()
        model_states = {'egnn':model_state1,'spectrum_compressor':model_state2}
    else:
        model_states = {'egnn':model_state1}
    torch.save(model_states,"./model_state/model_to_predict_epsilon/egnn_"+model_state_name+".pth")


    run.name = 'egnn_'+model_state_name
    run_id = wandb.run.id
    run.finish()
    
    if not only_train:
        return run_id, model_state_name