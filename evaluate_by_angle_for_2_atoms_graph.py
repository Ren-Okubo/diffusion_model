import torch, os, pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def calculate_angle_for_CN2(coords_tensor):
    v1 = coords_tensor[1] - coords_tensor[0]
    v2 = coords_tensor[2] - coords_tensor[0]
    cos = torch.dot(v1,v2) / (torch.norm(v1) * torch.norm(v2))
    return np.degrees(torch.acos(cos).item())

def calculate_bond_length_for_CN2(coords_tensor):
    v1 = coords_tensor[1] - coords_tensor[0]
    v2 = coords_tensor[2] - coords_tensor[0]
    return torch.norm(v1).item(),torch.norm(v2).item()


