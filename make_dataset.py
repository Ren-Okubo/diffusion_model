import torch
from torch_geometric.data import Data
import numpy as np
import os
import argparse
from data_preparation import fitted_intensity, fitted_intensity_wo_normalize
from pymatgen.core.structure import Structure, IStructure
from pymatgen.core import Lattice, SiteCollection
from pymatgen.core.sites import Site
from tqdm import tqdm

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
                pos[0] = "C"
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

def return_index_within_2ang(distance_matrix,reference_index):
    index_list = []
    for i in range(distance_matrix.shape[0]):
        if i == reference_index:
            continue
        if distance_matrix[reference_index,i] < 2.0:
            index_list.append(i)
    return index_list


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--range', type=str, required=True) # '2NN' or '4NN'
    argparser.add_argument('--save_dir_path', type=str, required=True)
    argparser.add_argument('--cell_dir_path', type=str, required=True)
    args = argparser.parse_args()
    assert args.range in ['2NN','3NN','4NN']
    if args.range == '2NN':
        cell_resource_path = args.cell_dir_path
        dataset = []
        dirs = os.listdir(cell_resource_path)
        dirs = [d for d in dirs if os.path.exists(os.path.join(cell_resource_path,d,'coreloss.cell'))]
        for i in tqdm(range(len(dirs))):
            d = dirs[i]
            cell_path = os.path.join(cell_resource_path,d,'coreloss.cell')
            structure = read_castep_output_structure(cell_path)
            num_sites = structure.num_sites
            abc = structure.lattice.abc
            angles = structure.lattice.angles
            lattice = Lattice.from_parameters(abc[0]*3,abc[1]*3,abc[2]*3,angles[0],angles[1],angles[2])
            extended_coords = []
            extended_species = []
            for i in range(num_sites):
                extended = [-1,0,1]
                for x in extended:
                    for y in extended:
                        for z in extended:
                            extended_coords.append(structure.sites[i].coords + structure.lattice.get_cartesian_coords(np.array([x,y,z]))+structure.lattice.get_cartesian_coords(np.array([1,1,1])))
                            if (x,y,z) != (0,0,0) and structure.sites[i].species_string == 'C':
                                extended_species.append('O')
                            else:
                                extended_species.append(structure.sites[i].species_string)
            structure = Structure(lattice,extended_species,extended_coords,coords_are_cartesian=True)
            istructure = IStructure.from_sites(structure)
            num_sites = istructure.num_sites
            for i in range(num_sites):
                if istructure.sites[i].species_string == 'C':
                    exO_index = i
                    break
            distance_matrix = istructure.distance_matrix
            filtered_atom_index = []
            firstNN_index_list = return_index_within_2ang(distance_matrix,exO_index)
            filtered_atom_index += firstNN_index_list
            for index in firstNN_index_list:
                filtered_atom_index += return_index_within_2ang(distance_matrix,index)
            filtered_atom_index = list(set(filtered_atom_index))
            filtered_atom_index = [i for i in filtered_atom_index if i != exO_index]
            filtered_atom_index = [exO_index] + filtered_atom_index
            filtered_position_list = []
            filtered_species_list = []
            for index in filtered_atom_index:
                filtered_position_list.append(torch.from_numpy(istructure.sites[index].coords).to(torch.float32)-torch.from_numpy(istructure.sites[filtered_atom_index[0]].coords).to(torch.float32))
                if istructure.sites[index].species_string == 'Si':
                    filtered_species_list.append(torch.tensor([0,1],dtype=torch.long))
                elif istructure.sites[index].species_string == 'O':
                    filtered_species_list.append(torch.tensor([1,0],dtype=torch.long))
                elif istructure.sites[index].species_string == 'C':
                    filtered_species_list.append(torch.tensor([1,0],dtype=torch.long))
                else:
                    print('error')
                    exit()
            graph = Data(x=torch.stack(filtered_species_list),pos=torch.stack(filtered_position_list))
            spectrum = fitted_intensity(os.path.join(cell_resource_path,d,'coreloss_core_edge.dat'))
            spectrum_raw = fitted_intensity_wo_normalize(os.path.join(cell_resource_path,d,'coreloss_core_edge.dat'))
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
        torch.save(dataset,os.path.join(args.save_dir_path,'dataset.pt'))
    elif args.range == '3NN':
        cell_resource_path = args.cell_dir_path
        dataset = []
        dirs = os.listdir(cell_resource_path)
        dirs = [d for d in dirs if os.path.exists(os.path.join(cell_resource_path,d,'coreloss.cell'))]
        for i in tqdm(range(len(dirs))):
            d = dirs[i]
            cell_path = os.path.join(cell_resource_path,d,'coreloss.cell')
            structure = read_castep_output_structure(cell_path)
            num_sites = structure.num_sites
            abc = structure.lattice.abc
            angles = structure.lattice.angles
            lattice = Lattice.from_parameters(abc[0]*3,abc[1]*3,abc[2]*3,angles[0],angles[1],angles[2])
            extended_coords = []
            extended_species = []
            for i in range(num_sites):
                extended = [-1,0,1]
                for x in extended:
                    for y in extended:
                        for z in extended:
                            extended_coords.append(structure.sites[i].coords + structure.lattice.get_cartesian_coords(np.array([x,y,z]))+structure.lattice.get_cartesian_coords(np.array([1,1,1])))
                            if (x,y,z) != (0,0,0) and structure.sites[i].species_string == 'C':
                                extended_species.append('O')
                            else:
                                extended_species.append(structure.sites[i].species_string)
            structure = Structure(lattice,extended_species,extended_coords,coords_are_cartesian=True)
            istructure = IStructure.from_sites(structure)
            num_sites = istructure.num_sites
            for i in range(num_sites):
                if istructure.sites[i].species_string == 'C':
                    exO_index = i
                    break
            distance_matrix = istructure.distance_matrix
            filtered_atom_index = []
            firstNN_index_list = return_index_within_2ang(distance_matrix,exO_index)
            filtered_atom_index += firstNN_index_list
            for index in firstNN_index_list:
                second_NN_index_list = return_index_within_2ang(distance_matrix,index)
                filtered_atom_index += second_NN_index_list
                for second_index in second_NN_index_list:
                    third_NN_index_list = return_index_within_2ang(distance_matrix,second_index)
                    filtered_atom_index += third_NN_index_list
            filtered_atom_index = list(set(filtered_atom_index))
            filtered_atom_index = [i for i in filtered_atom_index if i != exO_index]
            filtered_atom_index = [exO_index] + filtered_atom_index
            filtered_position_list = []
            filtered_species_list = []
            for index in filtered_atom_index:
                filtered_position_list.append(torch.from_numpy(istructure.sites[index].coords).to(torch.float32)-torch.from_numpy(istructure.sites[filtered_atom_index[0]].coords).to(torch.float32))
                if istructure.sites[index].species_string == 'Si':
                    filtered_species_list.append(torch.tensor([0,1],dtype=torch.long))
                elif istructure.sites[index].species_string == 'O':
                    filtered_species_list.append(torch.tensor([1,0],dtype=torch.long))
                elif istructure.sites[index].species_string == 'C':
                    filtered_species_list.append(torch.tensor([1,0],dtype=torch.long))
                else:
                    print('error')
                    exit()
            graph = Data(x=torch.stack(filtered_species_list),pos=torch.stack(filtered_position_list))
            spectrum = fitted_intensity(os.path.join(cell_resource_path,d,'coreloss_core_edge.dat'))
            spectrum_raw = fitted_intensity_wo_normalize(os.path.join(cell_resource_path,d,'coreloss_core_edge.dat'))
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
        torch.save(dataset,os.path.join(args.save_dir_path,'dataset.pt'))
    elif args.range == '4NN':
        cell_resource_path = args.cell_dir_path
        dataset = []
        dirs = os.listdir(cell_resource_path)
        dirs = [d for d in dirs if os.path.exists(os.path.join(cell_resource_path,d,'coreloss.cell'))]
        for i in tqdm(range(len(dirs))):
            d = dirs[i]
            cell_path = os.path.join(cell_resource_path,d,'coreloss.cell')
            structure = read_castep_output_structure(cell_path)
            num_sites = structure.num_sites
            abc = structure.lattice.abc
            angles = structure.lattice.angles
            lattice = Lattice.from_parameters(abc[0]*3,abc[1]*3,abc[2]*3,angles[0],angles[1],angles[2])
            extended_coords = []
            extended_species = []
            for i in range(num_sites):
                extended = [-1,0,1]
                for x in extended:
                    for y in extended:
                        for z in extended:
                            extended_coords.append(structure.sites[i].coords + structure.lattice.get_cartesian_coords(np.array([x,y,z]))+structure.lattice.get_cartesian_coords(np.array([1,1,1])))
                            if (x,y,z) != (0,0,0) and structure.sites[i].species_string == 'C':
                                extended_species.append('O')
                            else:
                                extended_species.append(structure.sites[i].species_string)
            structure = Structure(lattice,extended_species,extended_coords,coords_are_cartesian=True)
            istructure = IStructure.from_sites(structure)
            num_sites = istructure.num_sites
            for i in range(num_sites):
                if istructure.sites[i].species_string == 'C':
                    exO_index = i
                    break
            distance_matrix = istructure.distance_matrix
            filtered_atom_index = []
            firstNN_index_list = return_index_within_2ang(distance_matrix,exO_index)
            filtered_atom_index += firstNN_index_list
            for index in firstNN_index_list:
                second_NN_index_list = return_index_within_2ang(distance_matrix,index)
                filtered_atom_index += second_NN_index_list
                for second_index in second_NN_index_list:
                    third_NN_index_list = return_index_within_2ang(distance_matrix,second_index)
                    filtered_atom_index += third_NN_index_list
                    for third_index in third_NN_index_list:
                        fourth_NN_index_list = return_index_within_2ang(distance_matrix,third_index)
                        filtered_atom_index += fourth_NN_index_list
            filtered_atom_index = list(set(filtered_atom_index))
            filtered_atom_index = [i for i in filtered_atom_index if i != exO_index]
            filtered_atom_index = [exO_index] + filtered_atom_index
            filtered_position_list = []
            filtered_species_list = []
            for index in filtered_atom_index:
                filtered_position_list.append(torch.from_numpy(istructure.sites[index].coords).to(torch.float32)-torch.from_numpy(istructure.sites[filtered_atom_index[0]].coords).to(torch.float32))
                if istructure.sites[index].species_string == 'Si':
                    filtered_species_list.append(torch.tensor([0,1],dtype=torch.long))
                elif istructure.sites[index].species_string == 'O':
                    filtered_species_list.append(torch.tensor([1,0],dtype=torch.long))
                elif istructure.sites[index].species_string == 'C':
                    filtered_species_list.append(torch.tensor([1,0],dtype=torch.long))
                else:
                    print('error')
                    exit()
            graph = Data(x=torch.stack(filtered_species_list),pos=torch.stack(filtered_position_list))
            spectrum = fitted_intensity(os.path.join(cell_resource_path,d,'coreloss_core_edge.dat'))
            spectrum_raw = fitted_intensity_wo_normalize(os.path.join(cell_resource_path,d,'coreloss_core_edge.dat'))
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
        torch.save(dataset,os.path.join(args.save_dir_path,'dataset.pt'))