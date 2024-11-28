import argparse
import sys
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from train_per_iretation import diffuse_as_batch, train_epoch, val_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',type=str,default='diffusion_first_nearest')
    parser.add_argument('--dataset_path',type=str,default='/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset.pt')
    args = parser.parse_args()

    #parameterの読み込み
    with open ('/mnt/homenfsxx/rokubo/data/diffusion_model/parameters.yaml','r') as file:
        prms = yaml.safe_load(file)
    jst=pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    prms['now'] = now.strftime("%Y%m%d%H%M")

    #wandbの設定
    run = wandb.init(project=args.project_name,config=prms)

    #パラメータの設定
    #全体のパラメータ
    conditional=prms['conditional']
    seed = prms['seed']
    num_epochs = prms['num_epochs']
    batch_size = prms['batch_size']
    #optimizerのパラメータ
    lr = prms['lr']
    weight_decay = prms['weight_decay']
    #egnnのパラメータ
    L = prms['L']
    atom_type_size = prms['atom_type_size']
    spectrum_size = prms['spectrum_size']
    d_size = prms['d_size']
    t_size = prms['t_size']
    if conditional:
        h_size = atom_type_size + spectrum_size
    else:
        h_size = atom_type_size
    x_size = prms['x_size']
    m_size = prms['m_size']   
    m_input_size = h_size + t_size + h_size + t_size + d_size
    m_hidden_size = prms['m_hidden_size']
    m_output_size = m_size
    h_input_size = h_size + t_size + m_size
    h_hidden_size = prms['h_hidden_size']
    h_output_size = h_size
    x_input_size = h_size + t_size + h_size + t_size + d_size
    x_hidden_size = prms['x_hidden_size']
    x_output_size = 1
    onehot_scaling_factor = prms['onehot_scaling_factor']
    #early stoppingのパラメータ
    patience = prms['patience']
    #diffusion_processのパラメータ
    num_diffusion_timestep = prms['num_diffusion_timestep']
    noise_schedule = prms['noise_schedule']
    noise_precision = prms['noise_precision']
    power = prms['power']
    #spectrum_compressorのパラメータ
    to_compress_spectrum = prms['to_compress_spectrum']
    compresser_hidden_dim = prms['compresser_hidden_dim']
    compressed_spectrum_dim = prms['compressed_spectrum_dim']

    #default_tensor_typeの設定


    #seedの設定
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    #datasetの読み込み
    dataset = torch.load('/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset.pt')

    #spectrumサイズの設定（-1~19eVがデフォルト）

    #DataLoaderの設定
    train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    #modelの定義
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)
    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(original_spectrum_dim,spectrum_hidden_dim,compressed_spectrum_dim)
    early_stopping = EarlyStopping(patience=patience)
    setupdata = SetUpData(seed=seed,conditional=conditional)

    #optimizerの設定
    if to_compress_spectrum:
        if noise_schedule == 'learned':
            optimizer = torch.optim.Adam(list(egnn.parameters())+list(spectrum_compressor.parameters())+list(diffusion_process.parameters()),lr=lr,weight_decay=weight_decay)
            nn_dict = {'egnn':egnn,'spectrum_compressor':spectrum_compressor}
        else:
            optimizer = torch.optim.Adam(list(egnn.parameters())+list(spectrum_compressor.parameters()),lr=lr,weight_decay=weight_decay)
            nn_dict = {'egnn':egnn,'spectrum_compressor':spectrum_compressor}
    else:
        if noise_schedule == 'learned':
            optimizer = torch.optim.Adam(list(egnn.parameters())+list(diffusion_process.parameters()),lr=lr,weight_decay=weight_decay)
            nn_dict = {'egnn':egnn,'specturm_compressor':None}
        else:
            optimizer = torch.optim.Adam(list(egnn.parameters()),lr=lr,weight_decay=weight_decay)
            nn_dict = {'egnn':egnn,'spectrum_compressor':None}
    
    #train

    for epoch in range(num_epochs):
        train_epoch()
    #test
    
