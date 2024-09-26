import torch
import torch.nn as nn

def remove_mean(x:torch.tensor):
    mean = torch.mean(x,dim=0,keepdim=True)
    x = x - mean
    return x

class E3DiffusionProcess():
    def __init__(self,initial_beta,final_beta,num_diffusion_timestep:int,schedule_function='sigmoid'):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.schedule_function = schedule_function
        self.num_diffusion_timestep = num_diffusion_timestep
        if schedule_function == 'sigmoid':
            self.beta_schedule = torch.sigmoid(torch.linspace(-6,6,num_diffusion_timestep+1))
            self.beta_schedule = self.beta_schedule * (final_beta - initial_beta) + initial_beta
        elif schedule_function == 'linear':
            self.beta_schedule = torch.linspace(initial_beta,final_beta,num_diffusion_timestep+1)
        self.alpha_schedule = torch.ones(self.beta_schedule.shape) - self.beta_schedule
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule,dim=0)

    def diffuse_zero_to_t(self,pos:torch.tensor,t:int):
        noise = torch.zeros_like(pos)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos_after_diffuse = self.alpha_bar_schedule[t] * pos + self.beta_schedule[t] * noise
        return pos_after_diffuse , noise
    
    def calculate_mu(self,pos:torch.tensor,epsilon:torch.tensor,t:int):
        alpha_t = torch.sqrt(self.alpha_bar_schedule[t])
        #print(min(self.alpha_bar_schedule))
        #print(min(self.alpha_bar_schedule[1:]/self.alpha_bar_schedule[:-1]))
        alpha_s = torch.sqrt(self.alpha_bar_schedule[t-1])
        squared_sigma_t = 1 - self.alpha_bar_schedule[t]
        squared_sigma_s = 1 - self.alpha_bar_schedule[t-1]
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        #x_hat = (pos - torch.sqrt(1 - self.alpha_bar_schedule[t]) * epsilon) / torch.sqrt(self.alpha_schedule[t])
        x_hat = pos / alpha_t - torch.sqrt(squared_sigma_t) / alpha_t * epsilon
        if not torch.isfinite(x_hat).all():
            print('time',t)
            print('pos',pos)
            print('epsilon',epsilon)
            print('x_hat',x_hat)
            print('alpha_t',alpha_t)
            print('sigma_t',torch.sqrt(squared_sigma_t),torch.sqrt(1-self.alpha_bar_schedule[t]))
            print('alpha_ts',alpha_ts)
        mu = alpha_ts * squared_sigma_s * pos / squared_sigma_t + alpha_s * squared_sigma_ts * x_hat / squared_sigma_t

        #alpha_bar = self.alpha_bar_schedule[t-1]
        #alpha_t = self.alpha_schedule[t-1]
        #beta_t = self.beta_schedule[t-1]
        #mu = (pos - beta_t * epsilon / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_t)
        return mu
    
    def reverse_diffuse_one_step(self,mu,t):
        #std = (1 - self.alpha_bar_schedule[time-2]) * self.beta_schedule[time-1] / (1 - self.alpha_bar_schedule[time-1])
        alpha_t = torch.sqrt(self.alpha_bar_schedule[t])
        alpha_s = torch.sqrt(self.alpha_bar_schedule[t-1])
        squared_sigma_t = 1 - self.alpha_bar_schedule[t]
        squared_sigma_s = 1 - self.alpha_bar_schedule[t-1]
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
        noise = torch.zeros_like(mu)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos = mu + std * noise
        return pos

    def clip_noise_schedule(self,alphas2,clip_value=0.001):
        alphas2 = torch.cat([torch.ones(1),alphas2],dim=0)
        alphas_step = (alphas2[1:] / alphas2[:-1])
        alphas_step = torch.clamp(alphas_step,min=clip_value,max=1.)
        alphas2 = torch.cumprod(alphas_step,dim=0)
        return alphas2
    
    def polynomial_schedule(self,timesteps:int,s=1e-4,power=3.):
        steps = timesteps + 1
        x = torch.linspace(0,steps,steps)
        alphas2 = torch.pow(1 - torch.pow(x / steps,power),2)
        alphas2 = self.clip_noise_schedule(alphas2,clip_value=0.001)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        return alphas2
    
    def diffuse_to_t(self,pos:torch.tensor,t:int,s=1e-4):
        alpha = self.polynomial_schedule(self.num_diffusion_timestep,s=s)
        noise = torch.zeros_like(pos)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos_after_diffuse = alpha[t] * pos + torch.sqrt(1 - alpha[t]**2) * noise
        return pos_after_diffuse , noise
    def mu_calculate(self,pos:torch.tensor,epsilon:torch.tensor,t:int,s=1e-4):
        alpha = self.polynomial_schedule(self.num_diffusion_timestep,s=s)
        alpha_t = alpha[t]
        alpha_s = alpha[t-1]
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - alpha_ts**2 * squared_sigma_s
        x_hat = pos / alpha_t - torch.sqrt(squared_sigma_t) / alpha_t * epsilon
        mu = alpha_ts * squared_sigma_s * pos / squared_sigma_t + alpha_s * squared_sigma_ts * x_hat / squared_sigma_t
        return mu

    def reverse_onestep(self,mu,t,s=1e-4):
        alpha = self.polynomial_schedule(self.num_diffusion_timestep,s=s)
        alpha_t = alpha[t]
        alpha_s = alpha[t-1]
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - alpha_ts**2 * squared_sigma_s
        std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
        noise = torch.zeros_like(mu)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos = mu + std * noise
        return pos


"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from split_to_train_and_test import SetUpData
    setupdata = SetUpData()
    data = np.load("/home/rokubo/data/diffusion_model/dataset/dataset.npy",allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)
    diffusion_process = E3DiffusionProcess(0.0001,0.01,5000)
    for data in dataset:
        if data.id == 'mp-10064_1':
            file_name = '/home/rokubo/data/diffusion_model/test_vesta/10064_t_0.xyz'
            with open(file_name,'w') as f:
                f.write('5\n\n')
                f.write('O '+str(data.pos[0][0].item())+' '+str(data.pos[0][1].item())+' '+str(data.pos[0][2].item())+'\n')
                for i in range(1,5):
                    f.write('Si '+str(data.pos[i][0].item())+' '+str(data.pos[i][1].item())+' '+str(data.pos[i][2].item())+'\n')

            file_name = '/home/rokubo/data/diffusion_model/test_vesta/10064_t_1000.xyz'
            pos_after_diffuse, _ = diffusion_process.diffuse_zero_to_t(data.pos,1000)
            with open(file_name,'w') as f:
                f.write('5\n\n')
                f.write('O '+str(pos_after_diffuse[0][0].item())+' '+str(pos_after_diffuse[0][1].item())+' '+str(pos_after_diffuse[0][2].item())+'\n')
                for i in range(1,5):
                    f.write('Si '+str(pos_after_diffuse[i][0].item())+' '+str(pos_after_diffuse[i][1].item())+' '+str(pos_after_diffuse[i][2].item())+'\n')
            
            file_name = '/home/rokubo/data/diffusion_model/test_vesta/10064_t_2000.xyz'
            pos_after_diffuse, _ = diffusion_process.diffuse_zero_to_t(data.pos,2000)
            with open(file_name,'w') as f:
                f.write('5\n\n')
                f.write('O '+str(pos_after_diffuse[0][0].item())+' '+str(pos_after_diffuse[0][1].item())+' '+str(pos_after_diffuse[0][2].item())+'\n')
                for i in range(1,5):
                    f.write('Si '+str(pos_after_diffuse[i][0].item())+' '+str(pos_after_diffuse[i][1].item())+' '+str(pos_after_diffuse[i][2].item())+'\n')
            
            file_name = '/home/rokubo/data/diffusion_model/test_vesta/10064_t_3000.xyz'
            pos_after_diffuse, _ = diffusion_process.diffuse_zero_to_t(data.pos,3000)
            with open(file_name,'w') as f:
                f.write('5\n\n')
                f.write('O '+str(pos_after_diffuse[0][0].item())+' '+str(pos_after_diffuse[0][1].item())+' '+str(pos_after_diffuse[0][2].item())+'\n')
                for i in range(1,5):
                    f.write('Si '+str(pos_after_diffuse[i][0].item())+' '+str(pos_after_diffuse[i][1].item())+' '+str(pos_after_diffuse[i][2].item())+'\n')
            
            file_name = '/home/rokubo/data/diffusion_model/test_vesta/10064_t_4000.xyz'
            pos_after_diffuse, _ = diffusion_process.diffuse_zero_to_t(data.pos,4000)
            with open(file_name,'w') as f:
                f.write('5\n\n')
                f.write('O '+str(pos_after_diffuse[0][0].item())+' '+str(pos_after_diffuse[0][1].item())+' '+str(pos_after_diffuse[0][2].item())+'\n')
                for i in range(1,5):
                    f.write('Si '+str(pos_after_diffuse[i][0].item())+' '+str(pos_after_diffuse[i][1].item())+' '+str(pos_after_diffuse[i][2].item())+'\n')
            
            file_name = '/home/rokubo/data/diffusion_model/test_vesta/10064_t_5000.xyz'
            pos_after_diffuse, _ = diffusion_process.diffuse_zero_to_t(data.pos,5000)
            with open(file_name,'w') as f:
                f.write('5\n\n')
                f.write('O '+str(pos_after_diffuse[0][0].item())+' '+str(pos_after_diffuse[0][1].item())+' '+str(pos_after_diffuse[0][2].item())+'\n')
                for i in range(1,5):
                    f.write('Si '+str(pos_after_diffuse[i][0].item())+' '+str(pos_after_diffuse[i][1].item())+' '+str(pos_after_diffuse[i][2].item())+'\n')
"""

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diffusion_process1 = E3DiffusionProcess(0.00001,0.002,5000)
    diffusion_process2 = E3DiffusionProcess(0.00001,0.005,5000)
    diffusion_process3 = E3DiffusionProcess(0.00001,0.008,5000)
    x = torch.linspace(0,5000,5001)
    alphas1 = diffusion_process1.alpha_bar_schedule
    alphas2 = diffusion_process2.alpha_bar_schedule
    alphas3 = diffusion_process3.alpha_bar_schedule
    plt.plot(x,alphas1,label='final=0.002')
    plt.plot(x,alphas2,label='final=0.005')
    plt.plot(x,alphas3,label='final=0.008')
    plt.legend()
    plt.xlabel('timestep')
    plt.ylabel('alpha_bar')
    plt.savefig('alpha_bar_schedule_final.png')
    plt.close()

