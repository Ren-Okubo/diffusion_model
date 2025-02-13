import numpy as np
import os, itertools, pdb
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure, IStructure
from pymatgen.core import Lattice, SiteCollection
from pymatgen.core.sites import Site
from scipy.interpolate import InterpolatedUnivariateSpline
import torch
import torch.nn as nn 
from torch_geometric.data import Data


# CASTEPの出力ファイルから構造情報を読み取る関数
def read_castep_output_structure(file_path):
    with open(file_path, 'r') as f:
        # ファイルから構造情報を抽出する処理を実装する
        # 例えば、原子の座標やセルの情報を読み取る
        f.readline()
        lattice_length = [float(num) for num in f.readline().rstrip().split()]
        lattice_angle = [float(num) for num in f.readline().rstrip().split()]        
        lattice = Lattice.from_parameters(lattice_length[0],lattice_length[1],lattice_length[2],lattice_angle[0],lattice_angle[1],lattice_angle[2])
        f.readline()
        f.readline()
        f.readline()
        """
        line = f.readline().rstrip()
        while line != "%BLOCK POSITIONS_FRAC":
            line = f.readline().rstrip()
        """
        coords = []
        species = []
        line = f.readline().rstrip()
        i = 0 #iはO:exの引数
        while line != "%ENDBLOCK POSITIONS_FRAC":
            pos = []
            pos = line.split()
            if pos[0] == 'O:ex':
                O_ex = i
                pos[0] = "O"
            species.append(pos[0])
            site = Site(pos[0],[float(pos[1]),float(pos[2]),float(pos[3])])
            del pos[0]
            pos = [float(num) for num in pos]
            coords.append(pos)
            pos =[]
            i += 1
            line = f.readline().rstrip()
        # 抽出した情報を使ってStructureオブジェクトを作成する
        structure = Structure(lattice, species, coords)
    return structure

def find_line_number(file_path, target_text):
    with open(file_path, 'r') as file:
        line_number = 0
        for line in file:
            line_number += 1
            if target_text in line:
                return line_number
        # 特定の文言が見つからない場合
        return None

def calculate_center_of_mass(local_atom_dict): # kwargs = {"atom_type": [[],[],...]vector}
    atom_mass_list, vector_list = [], []
    for key, value in local_atom_dict.items():
        if key == 'O':
            for coords in value:
                vector_list.append(coords)
                atom_mass_list.append(16)
        elif key == 'Si':
            for coords in value:
                vector_list.append(coords)
                atom_mass_list.append(28.0855)
    sum_mass = sum(atom_mass_list)
    x_CoM = sum([atom_mass_list[i]*vector_list[i][0] for i in range(len(atom_mass_list))])/sum_mass
    y_CoM = sum([atom_mass_list[i]*vector_list[i][1] for i in range(len(atom_mass_list))])/sum_mass
    z_CoM = sum([atom_mass_list[i]*vector_list[i][2] for i in range(len(atom_mass_list))])/sum_mass
    return np.array([x_CoM, y_CoM, z_CoM])

def rotation_matrix_from_vector(vector):  #vector = O:ex - CoM
    """
    Calculate the rotation matrix that aligns the given vector to the x-axis.
    """
    vector = vector / np.linalg.norm(vector)
    x_axis = np.array([1, 0, 0])

    if np.allclose(vector, x_axis):
        return np.eye(3)

    axis = np.cross(vector, x_axis)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(vector, x_axis))

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def rotate_vectors(vectors, ref_vector):
    """
    Rotate the given vectors so that the ref_vector aligns with the x-axis.
    """
    R = rotation_matrix_from_vector(ref_vector)
    rotated_vectors = np.dot(vectors, R.T)
    return rotated_vectors

def base_convert(local_atoms:dict) -> dict:
    CoM_vector = calculate_center_of_mass(**local_atoms)
    ref_vector = local_atoms['O'] - CoM_vector
    for key, value in local_atoms.items():
        local_atoms[key] = rotate_vectors(value, ref_vector)
    return local_atoms

def padding_and_flatten(coords:dict) -> list:
    coords_list = []
    for key, value in coords.items():
        coords_list.append(value)
    for t in range(5-len(coords_list)):
        coords_list.append([0,0,0])
    coords_list = np.array(coords_list).flatten()
    return coords_list

def ex_O_vector(path) -> list:
    # coreloss.cell内のO:exの行数をカウントして励起酸素原子のindexを特定
    with open(path,'r') as f:
        for line in f:
            content = line.split(" ")
            if content[0] == "O:ex":
                ex_coords = np.array([float(content[1]),float(content[2]),float(content[3].replace("\n",""))]) #励起酸素の座標を取得
                break
    structure = read_castep_output_structure(path)
    istructure = IStructure.from_sites(structure.sites)
    for i,site in enumerate(istructure.sites):
        if np.array_equal(ex_coords,site.frac_coords):
            ex_index = i


    vector_list = []


    # 結合を解析するためのインスタンスを作成
    nn = CrystalNN()

    # 引数のサイトインデックスを設定
    site_index = ex_index

    # 近傍原子を取得
    neighbors = nn.get_nn_info(structure, site_index)

    # 近傍原子へのベクトルを計算して出力
    site = structure[site_index]
    for neighbor in neighbors:
        vector_dict = {}
        neighbor_site = neighbor['site']
        neighbor_index = neighbor['site_index']
        # サイトと近傍原子の分数座標を取得
        site_frac_coords = site.frac_coords
        neighbor_frac_coords = neighbor_site.frac_coords
    
        # 分数座標系でのベクトルを計算
        vector_frac = neighbor_frac_coords - site_frac_coords
    
        # 最小画像法を適用してベクトルを正規化
        vector_frac = vector_frac - np.round(vector_frac)  # ベクトルを [-0.5, 0.5] の範囲に収める
        vector_cartesian = structure.lattice.get_cartesian_coords(vector_frac)  # デカルト座標系に変換
        
        vector_dict['index'] = neighbor_index
        vector_dict['site'] = neighbor_site.specie
        vector_dict['vector'] = vector_cartesian
        vector_list.append(vector_dict)
    return vector_list

def normalize_list(input_list):
    # リスト内の最小値と最大値を見つける
    min_val = min(input_list)
    max_val = max(input_list)
    # スケーリングするための範囲を計算
    range_val = max_val - min_val
    # スケーリングされた値を計算して新しいリストに追加
    normalized_list = [(x - min_val) / range_val for x in input_list]
    return normalized_list

def fitted_intensity(coreloss_core_edge_path):
    target_text = "#  O 1    K1      O:ex"
    line_number = find_line_number(coreloss_core_edge_path,target_text)
    data = np.loadtxt(coreloss_core_edge_path,skiprows=line_number).T
    smooth = 0.001
    wavelengths = np.array(data[0])
    wavelength_min = wavelengths.min()
    wavelength_max = wavelengths.max()
    intensities = np.array(normalize_list(data[1]))
    # InterPolatedUnivariateSplineのインスタンス化
    spline = InterpolatedUnivariateSpline(wavelengths, intensities)
    new_wavelengths = np.arange(-1,19,0.1)
    # 近似曲線の強度値を計算
    fitted_intensities = spline(new_wavelengths)
    return fitted_intensities

def fitted_intensity_wo_normalize(coreloss_core_edge_path):
    target_text = "#  O 1    K1      O:ex"
    line_number = find_line_number(coreloss_core_edge_path,target_text)
    data = np.loadtxt(coreloss_core_edge_path,skiprows=line_number).T
    smooth = 0.001
    wavelengths = np.array(data[0])
    wavelength_min = wavelengths.min()
    wavelength_max = wavelengths.max()
    intensities = np.array(data[1])
    # InterPolatedUnivariateSplineのインスタンス化
    spline = InterpolatedUnivariateSpline(wavelengths, intensities)
    new_wavelengths = np.arange(-1,19,0.1)
    # 近似曲線の強度値を計算
    fitted_intensities = spline(new_wavelengths)
    return fitted_intensities

def get_beta_schedule(initial_beta:float, final_beta:float, num_diffusion_timestep:int) -> list:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    beta_schedule = np.linspace(-6,6,num_diffusion_timestep)
    beta_schedule = sigmoid(beta_schedule) * (final_beta - initial_beta) + initial_beta
    return beta_schedule

def noised_data_at_t(coords_onestep_before:dict, initial_beta:float, final_beta:float, num_diffusion_timestep:int, time:int):
    beta_schedule = get_beta_schedule(initial_beta, final_beta, num_diffusion_timestep)
    beta_t = beta_schedule[time-1]
    coords_at_t = {}
    noise_at_t = {}
    for key, value in coords_onestep_before.items():
        noise = np.random.randn(*value.shape)
        coords_for_each_atom = (np.sqrt(1-beta_t)*value + np.sqrt(beta_t)*noise)
        coords_at_t[key] = coords_for_each_atom
        noise_at_t[key] = noise
    return coords_at_t, noise_at_t, time 
    
def adjust_coords(standard_coord, coord_to_adjust):
    """
    周期的境界条件を考慮し、また励起酸素の座標を原点とした座標に変換する
    """
    dif = coord_to_adjust - standard_coord
    adjusted_coord = coord_to_adjust - np.round(dif) - standard_coord
    return adjusted_coord



def local_env_coords(path):
    # coreloss.cell内のO:exの行数をカウントして励起酸素原子のindexを特定
    with open(path,'r') as f:
        for line in f:
            content = line.split(" ")
            if content[0] == "O:ex":
                ex_coords = np.array([float(content[1]),float(content[2]),float(content[3].replace("\n",""))]) #励起酸素の座標を取得
                break
    structure = read_castep_output_structure(path)
    istructure = IStructure.from_sites(structure.sites)
    for i,site in enumerate(istructure.sites):
        if np.array_equal(ex_coords,site.frac_coords):
            ex_index = i
    vector_list = []
    # 結合を解析するためのインスタンスを作成
    nn = CrystalNN()
    # 引数のサイトインデックスを設定
    site_index = ex_index
    # 近傍原子を取得
    neighbors = nn.get_nn_info(structure, site_index)
    # 近傍原子へのベクトルを計算して出力
    local_env = {}
    site = structure[site_index]
    site_coords = site.frac_coords
    exO_cartesian_coord = structure.lattice.get_cartesian_coords(site_coords)
    local_env['O:ex'] = [exO_cartesian_coord - exO_cartesian_coord]
    local_env['Si'] = []
    local_env['O'] = []
    for neighbor in neighbors:
        neighbor_site = neighbor['site']
        neighbor_site_coords = neighbor_site.frac_coords
        neighbor_site_coords = adjust_coords(site_coords, neighbor_site_coords)
        neighbor_site_cartesian_coords = structure.lattice.get_cartesian_coords(neighbor_site_coords)
        local_env[str(neighbor_site.specie)].append(neighbor_site_cartesian_coords)
    return local_env



"""
initial_beta = 1.0*10**(-7)
final_beta = 2*10**(-3)
num_diffusion_timestep = 5000

if __name__ == "__main__":
    root_dir = '/home/rokubo/data/o-si_mp_voptados_8ang_primitive/test/o-si_mp_voptados_8ang_primitive/eels_o/'

    # 指定されたディレクトリ内のすべてのディレクトリを取得
    dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    #ディレクトリ内にcoreloss_core_edge.datファイルがあるかの判定
    dirs = [d for d in dirs if os.path.exists(os.path.join(root_dir,d,"coreloss_core_edge.dat"))]

    spectrum_for_CN_binary, label_for_CN_binary = [], []
    spectrum_for_CN_0134, label_for_CN_0134 = [], []

    coords_input_for_noise_prediction, noise_prediction_target = [], []
    spectrum_for_noise = []
    time_input_for_noise_prediction = []

    dict_for_CN_binary = {0:0,1:0,2:1,3:0,4:0}
    dict_for_CN_0134 = {0:0,1:1,3:2,4:3}

    for d in dirs:
        coreloss_core_edge_path = os.path.join(root_dir,d,"coreloss_core_edge.dat")
        fitted_intensities = fitted_intensity(coreloss_core_edge_path)
        coreloss_cell_path = os.path.join(root_dir,d,"coreloss.cell")
        bonded_atom = ex_O_vector(coreloss_cell_path)
        if d == 'mp-1173536_30':
            bonded_Si = []
        else:
            bonded_Si = [element for element in bonded_atom if str(element['site'])=='Si']
        CN_Si = len(bonded_Si)
        spectrum_for_CN_binary.append(fitted_intensities)
        label_for_CN_binary.append(dict_for_CN_binary[CN_Si])
        if CN_Si != 2:
            spectrum_for_CN_0134.append(fitted_intensities)
            label_for_CN_0134.append(dict_for_CN_0134[CN_Si])
        spectrum_for_noise.append(fitted_intensities)
        local_atoms = {"O":np.array([0,0,0])}
        for atom in bonded_atom:
            local_atoms[str(atom['site'])] = atom['vector']
        local_atoms = base_convert(local_atoms)
        atoms_coords = padding_and_flatten(local_atoms)
        coords_input_for_noise_prediction.append(atoms_coords)
        noise_prediction_target.append([0 for i in range(15)])
        time_input_for_noise_prediction.append(0)
        for t in range(num_diffusion_timestep):
            time = t+1
            local_atoms, noise, time = noised_data_at_t(local_atoms, initial_beta, final_beta, num_diffusion_timestep, time)
            local_atoms = base_convert(local_atoms)
            atoms_coords = padding_and_flatten(local_atoms)
            noise = padding_and_flatten(noise)
            coords_input_for_noise_prediction.append(atoms_coords)
            noise_prediction_target.append(noise)
            time_input_for_noise_prediction.append(time)

    spectrum_for_CN_binary = np.array(spectrum_for_CN_binary)
    label_for_CN_binary = np.array(label_for_CN_binary)
    spectrum_for_CN_0134 = np.array(spectrum_for_CN_0134)
    label_for_CN_0134 = np.array(label_for_CN_0134)
    coords_input_for_noise_prediction = np.array(coords_input_for_noise_prediction)
    noise_prediction_target = np.array(noise_prediction_target)
    time_input_for_noise_prediction = np.array(time_input_for_noise_prediction)
    spectrum_for_noise = np.array(spectrum_for_noise)

    dataset_path = '/home/rokubo/data/diffusion_model/dataset/'
    np.save(os.path.join(dataset_path,"data_for_CN_prediction",'binary','spectrum_for_CN_binary.npy'),spectrum_for_CN_binary)
    np.save(os.path.join(dataset_path,"data_for_CN_prediction",'binary','label_for_CN_binary.npy'),label_for_CN_binary)
    np.save(os.path.join(dataset_path,"data_for_CN_prediction",'0134','spectrum_for_CN_0134.npy'),spectrum_for_CN_0134)
    np.save(os.path.join(dataset_path,"data_for_CN_prediction",'0134','label_for_CN_0134.npy'),label_for_CN_0134)
    np.save(os.path.join(dataset_path,"data_for_noise_prediction",'coords_input_for_noise_prediction.npy'),coords_input_for_noise_prediction)
    np.save(os.path.join(dataset_path,"data_for_noise_prediction",'noise_prediction_target.npy'),noise_prediction_target)
    np.save(os.path.join(dataset_path,"data_for_noise_prediction",'time_input_for_noise_prediction.npy'),time_input_for_noise_prediction)
    np.save(os.path.join(dataset_path,"data_for_noise_prediction",'spectrum_for_noise.npy'),spectrum_for_noise)
        

        #five_coords = vector_to_data_style(**local_atoms) #3*5(座標×サイト数)
        #coords = five_coords[0] + five_coords[1] + five_coords[2] + five_coords[3] + five_coords[4] #15*1に変換
        #coords_for_noise.append(coords)


if __name__ == "__main__":
    root_dir = '/home/rokubo/data/o-si_mp_voptados_8ang_primitive/test/o-si_mp_voptados_8ang_primitive/eels_o/'

    # 指定されたディレクトリ内のすべてのディレクトリを取得
    dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    #ディレクトリ内にcoreloss_core_edge.datファイルがあるかの判定
    dirs = [d for d in dirs if os.path.exists(os.path.join(root_dir,d,"coreloss_core_edge.dat"))]

    dataset = []
    d_list, fitted_intensities_list, graph_list = [], [], []
    for d in dirs:
        coreloss_core_edge_path = os.path.join(root_dir,d,"coreloss_core_edge.dat")
        fitted_intensities = fitted_intensity(coreloss_core_edge_path)
        coreloss_cell_path = os.path.join(root_dir,d,"coreloss.cell")
        local_env = local_env_coords(coreloss_cell_path)
        if d == 'mp-1173536_30':
            local_env.pop('Si',None)
        Si_local_env = {}
        for key,value in local_env.items():
            if key == 'O':
                pass
            else:
                Si_local_env[key] = value
        local_env_list = []
        one_hot_dict = {'O:ex':[1,0],'Si':[0,1],'O':[1,0]}
        for key, values in Si_local_env.items():
            for value in values:
                local_env_list.append([one_hot_dict[key], value])
        dataset.append([d,fitted_intensities,local_env_list])
    for data in dataset:
        d_list.append(data[0])
        fitted_intensities_list.append(data[1])
        graph_list.append(data[2])
    dataset = np.array(dataset,dtype=object)
    d_list = np.array(d_list)
    fitted_intensities_list = np.array(fitted_intensities_list)
    graph_list = np.array(graph_list,dtype=object)

    np.save('/home/rokubo/data/diffusion_model/dataset/dataset.npy',dataset)
    np.savez('/home/rokubo/data/diffusion_model/dataset/dataset.npz',mp=d_list,fitted_intensity=fitted_intensities_list,graph_info=graph_list)

"""

if __name__ == '__main__':
    raw_data_path = '/mnt/homenfsxx/rokubo/o-si_mp_voptados_8ang_primitive/eels_o/'

    dirs = [d for d in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))]
    dirs = [d for d in dirs if os.path.exists(os.path.join(raw_data_path,d,"coreloss_core_edge.dat"))]
    
    dataset = []
    for d in dirs:    
        vector_list = ex_O_vector(os.path.join(raw_data_path,d,"coreloss.cell"))
        atoms_for_graph = []
        pos_for_graph = []
        intensities_for_graph = []
        edge_index = []

        #ex:Oの座標を原点とした
        atoms_for_graph.append(torch.tensor([1,0],dtype=torch.long))
        pos_for_graph.append(torch.tensor([0,0,0],dtype=torch.long))

        for vector in vector_list:
            if str(vector['site']) == 'O':
                atoms_for_graph.append(torch.tensor([1,0],dtype=torch.long))
            elif str(vector['site']) == 'Si':
                atoms_for_graph.append(torch.tensor([0,1],dtype=torch.long))
            pos_for_graph.append(torch.tensor(vector['vector'],dtype=torch.float32))

        fitted_intensities = fitted_intensity(os.path.join(raw_data_path,d,"coreloss_core_edge.dat"))
        num_atoms = len(atoms_for_graph)
        for i in range(num_atoms):
            intensities_for_graph.append(torch.tensor(fitted_intensities,dtype=torch.float32))

        node_num_list = [i for i in range(num_atoms)]
        permutations = list(itertools.permutations(node_num_list,2))
        for permutation in permutations:
            edge_index.append(list(permutation))
        edge_index = torch.tensor(edge_index,dtype=torch.long)
        
        data = Data(x=torch.stack(atoms_for_graph),edge_index=edge_index.t().contiguous(),pos=torch.stack(pos_for_graph),spectrum=torch.stack(intensities_for_graph),id=d)
        dataset.append(data)

    torch.save(dataset,'/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset.pt')
    
    """
    dataset = torch.load('/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset.pt')

    assert len(dataset) == len(dirs), "The number of data is different from the number of directories"
    for data in dataset:
        intensities_for_graph = []
        d = data.id
        num_atom = data.x.size()[0]
        coreloss_core_edge_path = os.path.join(raw_data_path,d,"coreloss_core_edge.dat")
        fitted_intensities = fitted_intensity(coreloss_core_edge_path)
        for i in range(num_atom):
            intensities_for_graph.append(torch.tensor(fitted_intensities,dtype=torch.float32))
        data.spectrum = torch.stack(intensities_for_graph)
    torch.save(dataset,'/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/first_nearest/dataset.pt')
    """

    