import os, torch
from torch_geometric.data import Data
import numpy as np
import argparse
import wandb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from CN2_evaluate import calculate_angle_for_CN2, calculate_bond_length_for_CN2, r2score

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_id', type=str, required=True)
    args = argparser.parse_args()

    run = wandb.init(project=args.project_name, id=args.run_id, resume='must')
    params = run.config

    original_graph_list = torch.load(params['original_graph_save_path'])
    generated_graph_list = torch.load(params['generated_graph_save_path'])
    considered_original_pos_list, considered_generated_pos_list = [], []
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        num_atom = original_graph.pos.shape[0]
        original_index_list, generated_index_list = [], []
        for i in range(1,num_atom):
            if torch.norm(original_graph.pos[i]-original_graph.pos[0]) < 2.0:
                original_index_list.append(i)
            if torch.norm(generated_graph.pos[i]-generated_graph.pos[0]) < 2.0:
                generated_index_list.append(i)
        if len(original_index_list) != 2 or len(generated_index_list) != 2:
            continue
        Si_tensor = torch.tensor([0,1],dtype=torch.long)
        if not torch.equal(original_graph.x[original_index_list[0]].to('cuda'),Si_tensor.to('cuda')) or not torch.equal(original_graph.x[original_index_list[1]].to('cuda'),Si_tensor.to('cuda')):
            continue
        if not torch.equal(generated_graph.x[generated_index_list[0]].to('cuda'),Si_tensor.to('cuda')) or not torch.equal(generated_graph.x[generated_index_list[1]].to('cuda'),Si_tensor.to('cuda')):
            continue
        considered_original_pos_list.append(original_graph.pos[[0]+original_index_list])
        considered_generated_pos_list.append(generated_graph.pos[[0]+generated_index_list])
    theta_angle, phi_angle = [], []
    theta_length, phi_length = [], []
    for i in range(len(considered_original_pos_list)):
        theta_angle.append(calculate_angle_for_CN2(considered_original_pos_list[i]))
        phi_angle.append(calculate_angle_for_CN2(considered_generated_pos_list[i]))
        norm1, norm2 = calculate_bond_length_for_CN2(considered_original_pos_list[i])
        theta_length.append((norm1+norm2)/2)
        norm1, norm2 = calculate_bond_length_for_CN2(considered_generated_pos_list[i])
        phi_length.append((norm1+norm2)/2)
    
    r2score_angle = r2score(theta_angle,phi_angle)
    r2score_length = r2score(theta_length,phi_length)


    gs = GridSpec(2,2,height_ratios=[1,4],width_ratios=[4,1])
    fig = plt.figure(figsize=(10,10))

    ax_scatter = fig.add_subplot(gs[1,0])
    ax_scatter.plot([0,180],[0,180],color='red',zorder=3,alpha=0.7)
    ax_scatter.plot(theta_angle,phi_angle,'o',markersize=1.5)
    ax_scatter.set_xlabel('target angle')
    ax_scatter.set_ylabel('generated angle')
    ax_scatter.set_xlim(60,180)
    ax_scatter.set_ylim(60,180)

    ax_text = fig.add_subplot(gs[0,1])
    ax_text.text(0.5,0.5,'R2 score of angle:\n{:.2f}'.format(r2score_angle),fontsize=12,ha='center', va='center')
    ax_text.axis('off')

    ax_hist_theta = fig.add_subplot(gs[0,0],sharex=ax_scatter)
    ax_hist_phi = fig.add_subplot(gs[1,1],sharey=ax_scatter)
    ax_hist_theta.hist(theta_angle,bins=50,orientation='vertical')
    ax_hist_phi.hist(phi_angle,bins=50,orientation='horizontal')
    ax_hist_theta.get_xaxis().set_visible(False)
    ax_hist_phi.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)

    plt.savefig(os.path.join(run.dir,'angle_Si-exO-Si.png'))
    image = Image.open(os.path.join(run.dir,'angle_Si-exO-Si.png'))
    wandb.log({'angle_Si-exO-Si':wandb.Image(image)})
    plt.close()

    gs = GridSpec(2,2,height_ratios=[1,4],width_ratios=[4,1])
    fig = plt.figure(figsize=(10,10))

    ax_scatter = fig.add_subplot(gs[1,0])
    ax_scatter.plot([1.3,2.2],[1.3,2.2],color='red',zorder=3,alpha=0.7)
    ax_scatter.plot(theta_length,phi_length,'o',markersize=1.5)
    ax_scatter.set_xlabel('target bond length')
    ax_scatter.set_ylabel('generated bond length')
    ax_scatter.set_xlim(1.3,2.2)
    ax_scatter.set_ylim(1.3,2.2)
    
    ax_text = fig.add_subplot(gs[0,1])
    ax_text.text(0.5,0.5,'R2 score of bond length:\n{:.2f}'.format(r2score_length),fontsize=12,ha='center', va='center')
    ax_text.axis('off')

    ax_hist_theta = fig.add_subplot(gs[0,0],sharex=ax_scatter)
    ax_hist_phi = fig.add_subplot(gs[1,1],sharey=ax_scatter)
    ax_hist_theta.hist(theta_length,bins=50,orientation='vertical')
    ax_hist_phi.hist(phi_length,bins=50,orientation='horizontal')
    ax_hist_theta.get_xaxis().set_visible(False)
    ax_hist_phi.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)

    plt.savefig(os.path.join(run.dir,'bond_length_Si-exO-Si.png'))
    image = Image.open(os.path.join(run.dir,'bond_length_Si-exO-Si.png'))
    wandb.log({'bond_length_Si-exO-Si':wandb.Image(image)})
    plt.close()
    
    run.finish()
