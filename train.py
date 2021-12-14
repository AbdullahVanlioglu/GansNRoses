import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle

from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm
from generate_noise import generate_spatial_noise
from environment.level_utils import  one_hot_to_ascii_level
from models import init_models, reset_grads, calc_gradient_penalty, save_networks
from draw_concat import draw_concat
from read_maps import *

from stable_baselines3 import DQN
from environment.singleAgentTestEnv import TestGanEnv


def test_score():

    env = TestGanEnv(map_type="test", visualization=False)
    model = DQN.load("./weights/single_map_dqn_1", env = env)

    total_rew = 0

    done = False
    obs = env.reset()

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        total_rew += rewards

    return total_rew


stat_columns = ['errD_fake', 'errD_real', 'errG']

def write_stats(stats,file_name='errors.csv'):
    df_stats = pd.DataFrame([stats], columns=stat_columns)
    df_stats.to_csv(file_name, mode='a', index=False,header=not os.path.isfile(file_name))

class GAN:
    def __init__(self,opt):
        self.D, self.G = init_models(opt)

        self.padsize = int(1 * opt.num_layer)  # As kernel size is always 3 currently, padsize goes up by one per layer

        self.pad_noise = nn.ZeroPad2d(self.padsize)
        self.pad_image = nn.ZeroPad2d(self.padsize)

        # setup optimizer
        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerD, milestones=[1500, 2500], gamma=opt.gamma)
        self.schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerG, milestones=[1500, 2500], gamma=opt.gamma)

    def train(self, real, opt):
        """ Train one scale. D and G are the discriminator and generator, real is the original map and its label.
        opt is a namespace that holds all necessary parameters. """
        real = torch.FloatTensor(real) # 1x2x4x4
        nzx = real.shape[2]  # Noise size x
        nzy = real.shape[3]  # Noise size y

        for step in tqdm(range(opt.niter)):
            noise_ = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device) # 1x2x4x4
            #noise_ = self.pad_noise(noise_) # 1x2x6x6

            ###############################################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###############################################
            for j in range(opt.Dsteps):

                if opt.add_prev:
                    if(j==0 and step==0):
                        prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                        prev = self.pad_image(prev)
                    else:
                        prev = interpolate(prev, real.shape[-2:], mode="bilinear", align_corners=False)
                        prev = self.pad_image(prev)
                else:
                    prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    prev = self.pad_image(prev)

                # train with real and fake
                self.D.zero_grad()
                real = real.to(opt.device)
                output = self.D(real).to(opt.device)
                # errD_real = -torch.clamp(output_r.mean(),min=-5.0,max=5.0)
                errD_real = -output.mean()
                errD_real.backward(retain_graph=True)

                # After creating our correct noise input, we feed it to the generator:
                #noise = noise_ + prev

                fake = self.G(noise_.detach(), prev, temperature=1).to(opt.device)

                # Then run the result through the discriminator
                output = self.D(fake.detach()).to(opt.device)
                errD_fake = output.mean()
                # errD_fake = torch.clamp(output_f.mean(),min=-5.0,max=5.0)

                # Backpropagation
                errD_fake.backward(retain_graph=False)

                # # Gradient Penalty
                gradient_penalty = calc_gradient_penalty(self.D, real, fake, opt.lambda_grad, opt.device)
                gradient_penalty.backward(retain_graph=False)

                self.optimizerD.step()

            ########################################
            # (2) Update G network: maximize D(G(z))
            ########################################

            for j in range(opt.Gsteps):
                self.G.train()
                self.G.zero_grad()
                fake = self.G(noise_.detach(), prev.detach(), temperature=1).to(opt.device)
                output = self.D(fake).to(opt.device)
                
                with open('./library/temp_map.pkl', 'wb') as f:
                    pickle.dump(fake.detach().numpy(), f)

                agent_score = test_score()

                if agent_score >= -16:
                    loss = 1
                else:
                    loss = 0

                errG = -output.mean() + torch.Tensor(loss)
                errG.backward(retain_graph=False)

                self.optimizerG.step()
            
            write_stats([errD_fake.item(), errD_real.item(), errG.item()])

            self.schedulerD.step()
            self.schedulerG.step()

        # self.G = reset_grads(self.G, True)
        # self.D = reset_grads(self.D, True)
        
        with torch.no_grad():
            self.G.eval()
            generated_map = self.G(noise_.detach(), prev.detach(), temperature=1).to(opt.device)
            
        return generated_map
    
    def better_save(self, iteration):
        save_networks(self.G, self.D, iteration)
