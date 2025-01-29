import os
import numpy as np
import torch
import itertools
import argparse
import wandb

def write_xyz_for_pos_generation(save_path,original_pos,generated_pos,comment=None):
    num_atom = original_pos.shape[0]
    
    with open(save_path,'w') as f:
        f.write(str(num_atom*2)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            f.write(f'F {original_pos[i][0].item()} {original_pos[i][1].item()} {original_pos[i][2].item()}\n')
        for i in range(num_atom):
            f.write(f'O {generated_pos[i][0].item()} {generated_pos[i][1].item()} {generated_pos[i][2].item()}\n')

def write_xyz_for_aligned_pos(save_path,position,comment=None):
    num_atom = position.shape[0]
    
    with open(save_path,'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            f.write(f'O {position[i][0].item()} {position[i][1].item()} {position[i][2].item()}\n')

def write_xyz(save_path,graph,comment=None):
    num_atom = graph.pos.shape[0]
    Si_tensor = torch.tensor([0,1]).to('cuda')
    O_tensor = torch.tensor([1,0]).to('cuda')
    with open(save_path,'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            if torch.equal(graph.x[i],Si_tensor):
                atom_type = 'Si'
            elif torch.equal(graph.x[i],O_tensor):
                atom_type = 'O'
            else:
                print('atom type error')
                exit()
            f.write(f'{atom_type} {graph.pos[i][0].item()} {graph.pos[i][1].item()} {graph.pos[i][2].item()}\n')
    

def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    original_coords = q
    generated_coords = np.dot(p, R.T)

    return R, t, rmsd, original_coords, generated_coords

def return_index_within_2ang(position):
    index_list = []
    for i in range(1,position.shape[0]):
        if torch.norm(position[i]-position[0]) < 2.0:
            index_list.append(i)
    return index_list


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--project_name', type=str, required=True)
    args.add_argument('--run_id', type=str, required=True)
    args = args.parse_args()

    api = wandb.Api()
    run = api.run(f'{args.project_name}/{args.run_id}')
    config = run.config
    generated_graph_save_path = config['generated_graph_save_path']
    original_graph_save_path = config['original_graph_save_path']

    os.makedirs(os.path.join('/home/rokubo/jbod/data/diffusion_model/result_generated',args.run_id),exist_ok=True)
    
    original_graph_list = torch.load(original_graph_save_path)
    generated_graph_list = torch.load(generated_graph_save_path)

    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]

        id = f'{original_graph.id}_{i}'
        #id = f'{args.run_id}_{i}'
        id_path = os.path.join('/home/rokubo/jbod/data/diffusion_model/result_generated',args.run_id,id)
        if os.path.exists(id_path):
            #continue
            pass
        os.makedirs(id_path,exist_ok=True)
        

        write_xyz_for_pos_generation(os.path.join(id_path,'original_generated.xyz'),original_graph.pos,generated_graph.pos,comment=id)

        generated_index_list = return_index_within_2ang(generated_graph.pos)
        original_index_list = return_index_within_2ang(original_graph.pos)
        if len(original_index_list) == 0 or len(generated_index_list) == 0:
            continue
        if len(original_index_list) != len(generated_index_list):
            continue
        original_pos_within_2ang = original_graph.pos[[0]+original_index_list]
        permuted_index_list = list(itertools.permutations(generated_index_list))
        min_rmsd = 1e+10
        min_permuted_index_list = None
        for perm in permuted_index_list:
            permuted_pos = generated_graph.pos[[0]+list(perm)]
            R, t, rmsd, original_coords, generated_coords = kabsch_numpy(permuted_pos.cpu().numpy(),original_pos_within_2ang.cpu().numpy())
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                min_permuted_index_list = perm
                min_R = R
                min_t = t

        P = generated_graph.pos.cpu().numpy()
        Q = original_graph.pos.cpu().numpy()
        p = P - np.mean(P, axis=0)
        q = Q - np.mean(Q, axis=0)
        rmsd_for_full_position = np.sqrt(np.sum(np.square(np.dot(p, min_R.T) - q)) / P.shape[0])
        aligned_original_pos = q
        aligned_generated_pos = np.dot(p, min_R.T)
        """
        rmsd_for_full_position = np.sqrt(np.sum(np.square(np.dot((P+min_t), min_R.T) - Q)) / P.shape[0])
        aligned_generated_pos = np.dot((P+min_t), min_R.T)
        aligned_original_pos = Q
        """
        """
        minus = aligned_generated_pos[0]
        for i in range(aligned_generated_pos.shape[0]):
            aligned_generated_pos[i] -= minus
        minus = aligned_original_pos[0]
        for i in range(aligned_original_pos.shape[0]):
            aligned_original_pos[i] -= minus
        """

        aligned_generated_pos = torch.from_numpy(aligned_generated_pos)
        aligned_original_pos = torch.from_numpy(aligned_original_pos)
        write_xyz_for_pos_generation(os.path.join(id_path,'aligned_generated.xyz'),aligned_original_pos,aligned_generated_pos,comment=str(rmsd_for_full_position))

