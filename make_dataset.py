import torch
from torch_geometric.data import Data
import numpy as np
import os
from data_preparation import fitted_intensity, fitted_intensity_wo_normalize

def degrees_to_radians(degrees):
    return degrees * np.pi / 180

def lattice_constants_to_matrix(a, b, c, alpha, beta, gamma):
    # 角度をラジアンに変換
    alpha = degrees_to_radians(alpha)
    beta = degrees_to_radians(beta)
    gamma = degrees_to_radians(gamma)
    
    # 格子定数行列の計算
    lattice_matrix = np.zeros((3, 3))
    lattice_matrix[0, 0] = a
    lattice_matrix[0, 1] = b * np.cos(gamma)
    lattice_matrix[0, 2] = c * np.cos(beta)
    lattice_matrix[1, 1] = b * np.sin(gamma)
    lattice_matrix[1, 2] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    lattice_matrix[2, 2] = c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)
    
    return lattice_matrix

def fractional_to_cartesian(lattice_matrix, fractional_coords):
    """
    格子定数と相対座標から絶対座標を計算する関数

    Parameters:
    lattice_constants (numpy.ndarray): 3x3の格子定数行列
    fractional_coords (numpy.ndarray): Nx3の相対座標行列

    Returns:
    numpy.ndarray: Nx3の絶対座標行列
    """


    # 相対座標を絶対座標に変換
    cartesian_coords = np.dot(fractional_coords, lattice_matrix)

    return cartesian_coords


def cell_to_atom_list(cell_path):
    with open(cell_path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    a, b, c = [float(x) for x in lines[1].split()]
    alpha, beta, gamma = [float(x) for x in lines[2].split()]
    lattice_matrix = lattice_constants_to_matrix(a, b, c, alpha, beta, gamma)
    first_atom_index = lines.index('%BLOCK POSITIONS_FRAC') + 1
    last_atom_index = lines.index('%ENDBLOCK POSITIONS_FRAC') - 1
    atom_list = []
    for i in range(first_atom_index, last_atom_index + 1):
        lines[i] = lines[i].split()
        fractional_coords = np.array([float(x) for x in lines[i][1:]])
        cartesian_coords = fractional_to_cartesian(lattice_matrix, fractional_coords)
        atom_list.append([lines[i][0], cartesian_coords])

    # 周囲のセルの原子を追加
    neighbor_shifts = [-1, 0, 1]
    extended_atom_list = []
    for atom in atom_list:
        for dx in neighbor_shifts:
            for dy in neighbor_shifts:
                for dz in neighbor_shifts:
                    shift = np.array([dx, dy, dz])
                    shifted_coords = atom[1] + np.dot(shift, lattice_matrix)
                    if (dx, dy, dz) != (0, 0, 0) and atom[0] == 'O:ex':
                        extended_atom_list.append(['O', shifted_coords])
                    else:
                        extended_atom_list.append([atom[0], shifted_coords])

    return extended_atom_list

def atom_list_to_graph(atom_list,cutoff=3.0):
    for i in range(len(atom_list)):
        if atom_list[i][0] == 'Si':
            atom_list[i][0] = np.array([0,1])
        elif atom_list[i][0] == 'O':
            atom_list[i][0] = np.array([1,0])
        elif atom_list[i][0] == 'O:ex':
            atom_list[i][0] = np.array([1,0])
            exO_index = i
        else:
            print('error')
            return
    exO_coords = torch.tensor(atom_list[exO_index][1],dtype=torch.float32)
    atom_list_exO = []
    for node in atom_list:
        node[0] = torch.tensor(node[0],dtype=torch.long)
        node[1] = torch.tensor(node[1],dtype=torch.float32) - exO_coords
        assert node[1].dim() == 1
    cutoffed_atom_list = []
    for node in atom_list:
        if torch.norm(node[1]) < cutoff:
            cutoffed_atom_list.append(node)
    num_of_atoms = len(cutoffed_atom_list)
    for i in range(len(cutoffed_atom_list)):
        if torch.norm(cutoffed_atom_list[i][1]) == 0:
            exO_index = i
    edge_index = []
    for i in range(num_of_atoms):
        for j in range(num_of_atoms):
            edge_index.append([i,j])
    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    sorted_atom_list = []
    sorted_atom_list.append(cutoffed_atom_list[exO_index])
    for i in range(num_of_atoms):
        if i != exO_index:
            sorted_atom_list.append(cutoffed_atom_list[i])
    graph = Data(x=torch.stack([node[0] for node in sorted_atom_list]),pos=torch.stack([node[1] for node in sorted_atom_list]),edge_index=edge_index)
    assert graph.x.dim() == 2
    assert graph.pos.dim() == 2
    return graph

def return_index_within_2ang(atom_list,reference_index):
    reference_pos = atom_list[reference_index][1]
    index_list = []
    for i in range(len(atom_list)):
        if i == reference_index:
            continue
        if torch.norm(atom_list[i][1] - reference_pos) < 2.0:
            index_list.append(i)
    return index_list




if __name__ == '__main__':
    dataset = []
    dirs = os.listdir('/home/rokubo/jbod/data/diffusion_model/dataset/dat_files')
    dirs = [d for d in dirs if os.path.exists(os.path.join('/home/rokubo/jbod/data/diffusion_model/dataset/dat_files',d,'coreloss.cell'))]
    for d in dirs:
        cell_path = os.path.join('/home/rokubo/jbod/data/diffusion_model/dataset/dat_files',d,'coreloss.cell')
        atom_list = cell_to_atom_list(cell_path)
        for i in range(len(atom_list)):
            if atom_list[i][0] == 'Si':
                atom_list[i][0] = np.array([0,1])
            elif atom_list[i][0] == 'O':
                atom_list[i][0] = np.array([1,0])
            elif atom_list[i][0] == 'O:ex':
                atom_list[i][0] = np.array([1,0])
                exO_index = i
            else:
                print('error')
                exit()
        exO_coords = torch.tensor(atom_list[exO_index][1],dtype=torch.float32)
        for node in atom_list:
            node[0] = torch.tensor(node[0],dtype=torch.long)
            node[1] = torch.tensor(node[1],dtype=torch.float32) - exO_coords
            assert node[1].dim() == 1
        filtered_atom_index = []
        firstNN_index_list = return_index_within_2ang(atom_list,exO_index)
        filtered_atom_index += firstNN_index_list
        for index in firstNN_index_list:
            filtered_atom_index += return_index_within_2ang(atom_list,index)
        filtered_atom_index = list(set(filtered_atom_index))
        filtered_atom_index = [i for i in filtered_atom_index if i != exO_index]
        filtered_atom_index = [exO_index] + filtered_atom_index
        filtered_atom_list = [atom_list[i] for i in filtered_atom_index]
        graph = Data(x=torch.stack([node[0] for node in filtered_atom_list]),pos=torch.stack([node[1] for node in filtered_atom_list]))
        spectrum = fitted_intensity(os.path.join('/home/rokubo/jbod/data/diffusion_model/dataset/dat_files',d,'coreloss_core_edge.dat'))
        spectrum_raw = fitted_intensity_wo_normalize(os.path.join('/home/rokubo/jbod/data/diffusion_model/dataset/dat_files',d,'coreloss_core_edge.dat'))
        num_of_atoms = graph.x.shape[0]
        spectrum_tensor = torch.zeros(num_of_atoms,spectrum.shape[0])
        spectrum_tensor_raw = torch.zeros(num_of_atoms,spectrum_raw.shape[0])
        spectrum_tensor[0] = torch.tensor(spectrum)
        spectrum_tensor_raw[0] = torch.tensor(spectrum_raw)
        exO_tensor = torch.zeros(num_of_atoms,1)
        exO_tensor[0] = 1
        edge_index_list = []
        for i in range(num_of_atoms):
            for j in range(num_of_atoms):
                if i != j:
                    edge_index_list.append([i,j])
        edge_index = torch.tensor(edge_index_list,dtype=torch.long).t().contiguous()
        graph.edge_index = edge_index
        graph.spectrum = spectrum_tensor
        graph.spectrum_raw = spectrum_tensor_raw
        graph.exO = exO_tensor
        graph.id = d
        dataset.append(graph)
    torch.save(dataset,'/home/rokubo/jbod/data/diffusion_model/dataset/2NN/dataset.pt')



