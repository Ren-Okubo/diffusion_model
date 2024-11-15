import torch
import torch.nn as nn

def remove_mean(x:torch.tensor,batch_index=None):
    if batch_index is None:
        mean = torch.mean(x,dim=0,keepdim=True)
        x = x - mean
    else:
        num_graph = batch_index.max().item()+1
        for i in range(num_graph):
            mean = torch.mean(x[batch_index==i],dim=0,keepdim=True)
            x[batch_index==i] = x[batch_index==i] - mean
    return x

class E3DiffusionProcess():
    def __init__(self,s,num_diffusion_timestep:int):
        self.noise_precision = s
        self.num_diffusion_timestep = num_diffusion_timestep
        self.t = torch.linspace(0,num_diffusion_timestep,num_diffusion_timestep+1) #０からnum_diffusion_timestepまでの5001個の整数
        self.alpha_schedule = self.polynomial_schedule(num_diffusion_timestep,s=s,power=2.)
        self.sigma_schedule = torch.sqrt(1-self.alpha_schedule**2)
        


    def diffuse_zero_to_t(self,pos:torch.tensor,h:torch.tensort:int):
        noise_x = torch.zeros_like(pos).to(pos.device)
        noise_h = torch.zeros_like(h).to(h.device)
        noise_x.normal_(mean=0,std=1)
        noise_h.normal_(mean=0,std=1)
        noise_x = remove_mean(noise_x)
        noise_z = torch.cat((noise_x,noise_h),dim=1)
        z_after_diffuse = self.alpha_schedule[t] * torch.cat((pos,h),dim=1) + self.sigma_schedule[t] * noise_z
        pos_after_diffuse = z_after_diffuse[:,:pos.size(1)]
        h_after_diffuse = z_after_diffuse[:,pos.size(1):]
        return pos_after_diffuse, h_after_diffuse, noise_x, noise_h
    
    def calculate_mu(self,pos:torch.tensor,h:torch.tensor,epsilon:torch.tensor,t:int):
        z = torch.cat((pos,h),dim=1)
        alpha_t = self.alpha_schedule[t]
        alpha_s = self.alpha_schedule[t-1]
        squared_sigma_t = 1 - alpha_t**2
        sigma_t = torch.sqrt(squared_sigma_t)
        squared_sigma_s = 1 - alpha_s**2
        sigma_s = torch.sqrt(squared_sigma_s)
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s

        mu = z / alpha_ts - squared_sigma_ts * epsilon / alpha_ts / sigma_t
        return mu
    
    def reverse_diffuse_one_step(self,pos:torch.tensor,h:torch.tensor,epsilon:torch.tensor,t:int):
        mu = self.calculate_mu(pos,h,epsilon,t)
        alpha_t = self.alpha_schedule[t]
        alpha_s = self.alpha_schedule[t-1]
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
        noise_x = torch.zeros_like(pos).to(pos.device)
        noise_h = torch.zeros_like(h).to(h.device)
        noise_x.normal_(mean=0,std=1)
        noise_x = remove_mean(noise_x)
        noise = torch.cat((noise_x,noise_h),dim=1)
        z = mu + std * noise
        x = z[:,:pos.size(1)]
        h = z[:,pos.size(1):]
        return x, h

    def clip_noise_schedule(self,alphas2,clip_value=0.001):
        alphas2 = torch.cat([torch.ones(1),alphas2],dim=0)
        alphas_step = (alphas2[1:] / alphas2[:-1])
        alphas_step = torch.clamp(alphas_step,min=clip_value,max=1.)
        alphas2 = torch.cumprod(alphas_step,dim=0)
        return alphas2
    
    def polynomial_schedule(self,timesteps:int,s=1e-4,power=3.):
        steps = timesteps
        x = torch.linspace(0,steps,steps+1)
        alphas2 = torch.pow(1 - torch.pow(x / steps,power),2)
        alphas2 = self.clip_noise_schedule(alphas2,clip_value=0.001)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        return alphas2

