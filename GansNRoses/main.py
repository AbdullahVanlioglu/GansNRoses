from numpy.lib.shape_base import expand_dims
import wandb
import sys
import torch
import os
import pandas as pd

from loguru import logger
from train import GAN
from read_maps import *
from torch.optim import Adam

from construct_library import Library
from config import get_arguments, post_config
from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS
from environment.level_image_gen import LevelImageGen as LevelGen
from environment.level_utils import read_level, one_hot_to_ascii_level
from generate_noise import generate_spatial_noise


def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]

colon = ['testc_labeled', 'train_loss', 'trainc_labeled', 'train_lib_size']

def write_tocsv(stats,file_name='performance.csv'):
    df_stats = pd.DataFrame([stats], columns=colon)
    df_stats.to_csv(file_name, mode='a', index=False,header=not os.path.isfile(file_name))

def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """
    #==================================================================================
    # torch.autograd.set_detect_anomaly(True)
    # Logger init
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      + "<level>{level}</level> | "
                      + "<light-black>{file.path}:{line}</light-black> | "
                      + "{message}")

    # Parse arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)

    # Init wandb
    run = wandb.init(project="environment", tags=get_tags(opt),
                     config=opt, dir=opt.out, mode="offline")
    opt.out_ = run.dir
    # Init game specific inputs
    sprite_path = opt.game + '/sprites'
    opt.ImgGen = LevelGen(sprite_path)
    replace_tokens = REPLACE_TOKENS
    #==================================================================================

    for i in range(10):

        if opt.model == "train":
            print("Train Mode")
            gen_lib = Library(100)

            G = GAN(opt)
            
            opt.input_name = "map_zero.txt"
            init_map = read_level(opt, None, replace_tokens)
            idx = 0

            while(True):
                
                generated_map = G.train(np.array(init_map), opt, i)
                coded_fake_map = one_hot_to_ascii_level(generated_map.detach(), opt.token_list)
                ground_locations, prize_locations, trap_locations, matrix_map = fa_regenate(coded_fake_map, opt)

                if idx >= 300:
                    G.better_save(i)
                    break
                    
                idx += 1
            
            G.generate_map(gen_lib, opt)

        elif opt.model =="test":
            _ = read_level(opt, None, replace_tokens)
            print("Test Mode")
            gen_lib = Library(100)
            G = GAN(opt)
            G.generate_map(gen_lib, opt, i)

if __name__ == "__main__":
    main()
