import os

from level_utils import load_level_from_text
from level_image_gen import LevelImageGen

# Renders all level.txt files to png images inside a given folder. Expects ONLY .txt in that folder.

SPRITE_PATH = "./sprites"

if __name__ == '__main__':
    ImgGen = LevelImageGen(SPRITE_PATH)

    directory = "../output/"
    dir_names = os.listdir(directory)
    dir_names.sort()
    if 'README.md' in dir_names:  # Ignore readme for default input folder
        dir_names.remove('README.md')

    if "wandb" in dir_names:
        dir_names.remove("wandb")

    for curr_gen in dir_names:
        directory_gen = directory + curr_gen
        print(directory_gen)
        names = os.listdir(directory_gen)
        names.sort()

        target_dir = '/home/avsp/Masaüstü/output/' + curr_gen 
        os.makedirs(target_dir, exist_ok=True)

        for i in range(min(10000, len(names))):
            lvl = load_level_from_text(os.path.join(directory_gen, names[i]))
            if lvl[-1][-1] == '\n':
                lvl[-1] = lvl[-1][0:-1]
            lvl_img = ImgGen.render(lvl)
            lvl_img.save(os.path.join(target_dir, names[i][0:-4] + '.png'), format='png')