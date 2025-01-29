from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch, os
import wandb
import matplotlib.pyplot as plt
from PIL import Image


def make_xyz(original_graph,generated_graph,save_dir_path,comment=None):
    original_graph = original_graph.to('cuda')
    generated_graph = generated_graph.to('cuda')
    num_atom = original_graph.pos.shape[0]
    Si_tensor = torch.tensor([0,1]).to('cuda')
    O_tensor = torch.tensor([1,0]).to('cuda')
    id = original_graph.id
    os.makedirs(os.path.join(save_dir_path,id),exist_ok=True)
    save_path = os.path.join(save_dir_path,id)
    with open(os.path.join(save_path,'original.xyz'),'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            if torch.equal(original_graph.x[i],Si_tensor):
                atom_type = 'Si'
            elif torch.equal(original_graph.x[i],O_tensor):
                atom_type = 'O'
            else:
                print('error')
                exit()
            f.write(f'{atom_type} {original_graph.pos[i][0].item()} {original_graph.pos[i][1].item()} {original_graph.pos[i][2].item()}\n')
    with open(os.path.join(save_path,'generated.xyz'),'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            if torch.equal(generated_graph.x[i],Si_tensor):
                atom_type = 'Si'
            elif torch.equal(generated_graph.x[i],O_tensor):
                atom_type = 'O'
            else:
                print('error')
                exit()
            f.write(f'{atom_type} {generated_graph.pos[i][0].item()} {generated_graph.pos[i][1].item()} {generated_graph.pos[i][2].item()}\n')
    

# 原子間の結合を距離に基づいて推定する
def guess_bonds(atoms, threshold=1.2):
    """
    原子間距離を基に結合を推定する。
    :param atoms: ASE Atomsオブジェクト
    :param threshold: 結合距離のスケールファクター（原子半径の和に掛ける）
    :return: RDKitのMolオブジェクト
    """
    # ASEから元素と座標を取得
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # RDKit Molオブジェクトを初期化
    mol = Chem.RWMol()

    # RDKitの原子を追加
    atom_indices = []
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)  # 暗黙の水素を無効化
        atom_indices.append(mol.AddAtom(atom))

    # 結合距離の閾値を決定
    covalent_radii = Chem.GetPeriodicTable().GetRcovalent  # 共価半径
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i >= j:  # 対称行列の下側のみを計算
                continue
            distance = np.linalg.norm(pos1 - pos2)
            # 原子半径の和にスケールファクターを掛けて結合を判定
            if distance < threshold * (covalent_radii(symbols[i]) + covalent_radii(symbols[j])):
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    # 部分的なサニタイズを実行（暗黙の水素を無視）
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)

    return mol

# フィンガープリントを生成
def generate_fingerprint(mol):
    # MorganGeneratorの作成
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # フィンガープリントの生成
    fp = morgan_generator.GetFingerprint(mol)
    return fp

# 類似性計算
def calculate_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def eval_by_xyz(xyz_file1,xyz_file2):
    # XYZファイルをASEで読み込み
    atoms1 = read(xyz_file1)
    atoms2 = read(xyz_file2)

    # 分子を生成
    mol1 = guess_bonds(atoms1)
    mol2 = guess_bonds(atoms2)

    # フィンガープリントを生成
    fp1 = AllChem.GetAtomPairFingerprint(mol1)
    fp2 = AllChem.GetAtomPairFingerprint(mol2)

    # 類似性を計算
    similarity = calculate_similarity(fp1, fp2)
    return similarity

if __name__ == '__main__':
    project_name = str(input('project_name : '))
    run_id = str(input('run_id : '))
    run = wandb.init(project=project_name, id=run_id, resume='must')
    config = run.config
    original_graph_list = torch.load(config['original_graph_save_path'])
    generated_graph_list = torch.load(config['generated_graph_save_path'])

    os.makedirs(os.path.join(run.dir,'xyz'),exist_ok=True)
    save_dir_path = os.path.join(run.dir,'xyz')
    similarity_list = []
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        make_xyz(original_graph,generated_graph,save_dir_path,comment=original_graph.id)
        similarity = eval_by_xyz(os.path.join(save_dir_path,original_graph.id,'original.xyz'),os.path.join(save_dir_path,original_graph.id,'generated.xyz'))
        similarity_list.append(similarity)
    sorted_similarity_list = sorted(similarity_list)
    fig, ax = plt.subplots()
    ax.plot(range(len(sorted_similarity_list)),sorted_similarity_list,marker='o',linestyle='None')
    ax.set_xlabel('index')
    ax.set_ylabel('similarity')
    ax.set_title('similarity between original and generated')
    plt.savefig(os.path.join(save_dir_path,'similarity.png'))
    image = Image.open(os.path.join(save_dir_path,'similarity.png'))
    wandb.log({'similarity by fingerprint':wandb.Image(image)})
    wandb.config.update({"xyz_save_dir": save_dir_path},allow_val_change=True)
    wandb.finish()
