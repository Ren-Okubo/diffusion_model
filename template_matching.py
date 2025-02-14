import os, torch, random
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from ase.build import molecule


def cos_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def graph_to_ase_molecule(graph):
    pos = graph.pos.cpu().numpy()
    atoms = []
    for i in range(graph.x.shape[0]):
        if torch.equal(graph.x[i].cpu(),torch.tensor([0,1]).cpu()):
            atoms.append('Si')
        elif torch.equal(graph.x[i].cpu(),torch.tensor([1,0]).cpu()):
            atoms.append('O')
        else:
            print('error')
            exit()
    ase_molecule = Atoms(symbols=atoms, positions=pos)
    return ase_molecule


if __name__ == '__main__':
    seed = 0
    #seedの設定
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reference_dataset = torch.load(str(input('reference_dataset_path:')))
    reference_dataset = [data for data in reference_dataset if data.pos.shape[0] > 1]

    target_dataset  = torch.load(str(input('target_dataset_path:')))
    target_dataset = [data for data in target_dataset if data.pos.shape[0] > 1]


    soap = SOAP(species=["O", "Si"], r_cut=6, n_max=15, l_max=10,sigma=0.1)
    best3_for_each_graph = {}
    for data in target_dataset:
        id = data.id
        target_spectrum = data.spectrum[0].numpy()
        target_soap = soap.create(graph_to_ase_molecule(data))
        mse_list = []
        id_list = []
        for d in reference_dataset:
            if d.id == id:
                continue
            id_list.append(d.id)
            reference_spectrum = d.spectrum[0].numpy()
            mse = np.mean((target_spectrum - reference_spectrum)**2)
            mse_list.append(mse)
        mse_id_list = list(zip(mse_list,id_list))
        sorted_mse_id_list = sorted(mse_id_list,key=lambda x:x[0])
        sorted_mse_list, sorted_id_list = zip(*sorted_mse_id_list)
        best3_list = []
        for i in range(3):
            for r_data in reference_dataset:
                if r_data.id == sorted_id_list[i]:
                    reference_soap = soap.create(graph_to_ase_molecule(r_data))
                    similarity = cos_similarity(target_soap[0],reference_soap[0])
                    best3_list.append({r_data.id:similarity})
                    break
        best3_for_each_graph[id] = best3_list
    save_path = str(input('enter the save directory:'))
    torch.save(best3_for_each_graph,os.path.join(save_path,'template_matching_result.pt'))
