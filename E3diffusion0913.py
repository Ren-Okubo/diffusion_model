import torch
import torch.nn as nn

def remove_mean(x:torch.tensor):
    mean = torch.mean(x,dim=1,keepdim=True)
    x = x - mean
    return x

class E3DiffusionProcess():
    def __init__(self,initial_beta,final_beta,num_diffusion_timestep:int,schedule_function='sigmoid'):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.schedule_function = schedule_function
        self.num_diffusion_timestep = num_diffusion_timestep
        if schedule_function == 'sigmoid':
            self.beta_schedule = torch.sigmoid(torch.linspace(-6,6,num_diffusion_timestep))
            self.beta_schedule = self.beta_schedule * (final_beta - initial_beta) + initial_beta
        self.alpha_schedule = torch.ones(self.beta_schedule.shape) - self.beta_schedule
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule,dim=0)

    def diffuse_zero_to_t(self,pos:torch.tensor,t:int):
        noise = torch.zeros_like(pos)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos_after_diffuse = self.alpha_bar_schedule[t-1] * pos + self.beta_schedule[t-1] * noise
        return pos_after_diffuse , noise
    
    def calculate_mu(self,pos:torch.tensor,epsilon:torch.tensor,t:int):
        alpha_t = torch.sqrt(self.alpha_bar_schedule[t-1])
        alpha_s = torch.sqrt(self.alpha_bar_schedule[t-2])
        squared_sigma_t = 1 - self.alpha_bar_schedule[t-1]
        squared_sigma_s = 1 - self.alpha_bar_schedule[t-2]
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        x_hat = (pos - torch.sqrt(1 - self.alpha_bar_schedule[t-1]) * epsilon) / torch.sqrt(self.alpha_schedule[t-1])
        mu = alpha_ts * squared_sigma_s * pos / squared_sigma_t + alpha_s * squared_sigma_ts * x_hat / squared_sigma_t

        #alpha_bar = self.alpha_bar_schedule[t-1]
        #alpha_t = self.alpha_schedule[t-1]
        #beta_t = self.beta_schedule[t-1]
        #mu = (pos - beta_t * epsilon / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_t)
        return mu
    
    def reverse_diffuse_one_step(self,mu,t:int):
        #std = (1 - self.alpha_bar_schedule[time-2]) * self.beta_schedule[time-1] / (1 - self.alpha_bar_schedule[time-1])
        alpha_t = torch.sqrt(self.alpha_bar_schedule[t-1])
        alpha_s = torch.sqrt(self.alpha_bar_schedule[t-2])
        squared_sigma_t = 1 - self.alpha_bar_schedule[t-1]
        squared_sigma_s = 1 - self.alpha_bar_schedule[t-2]
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
        noise = torch.zeros_like(mu)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos = mu + std * noise
        return pos
    

    