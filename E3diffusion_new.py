import torch
import torch.nn as nn

def remove_mean(x:torch.tensor):
    mean = torch.mean(x,dim=0,keepdim=True)
    x = x - mean
    return x

class E3DiffusionProcess():
    def __init__(self,s,num_diffusion_timestep:int):
        self.noise_precision = s
        self.num_diffusion_timestep = num_diffusion_timestep
        self.t = torch.linspace(0,num_diffusion_timestep,num_diffusion_timestep+1) #０からnum_diffusion_timestepまでの5001個の整数
        self.alpha_schedule = (1-2*s) * (1-(self.t / num_diffusion_timestep)**2) + s
        self.sigma_schedule = torch.sqrt(1-self.alpha_schedule**2)

    def diffuse_zero_to_t(self,pos:torch.tensor,t:int):
        noise = torch.zeros_like(pos)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos_after_diffuse = self.alpha_schedule[t] * pos + self.sigma_schedule[t] * noise
        return pos_after_diffuse , noise
    
    def calculate_mu(self,pos:torch.tensor,epsilon:torch.tensor,t:int):
        alpha_t = self.alpha_schedule[t]
        alpha_s = self.alpha_schedule[t-1]
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        x_hat = (pos - self.sigma_schedule[t] * epsilon) / alpha_t
        mu = alpha_ts * squared_sigma_s * pos / squared_sigma_t + alpha_s * squared_sigma_ts * x_hat / squared_sigma_t
        return mu
    
    def reverse_diffuse_one_step(self,mu,t:int):
        alpha_t = self.alpha_schedule[t]
        alpha_s = self.alpha_schedule[t-1]
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
        noise = torch.zeros_like(mu)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos = mu + std * noise
        return pos
    

    