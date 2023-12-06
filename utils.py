import os
import shutil

def save_constants(constants, save_dir):
    with open(os.path.join(save_dir, "constants.txt"), "w") as f:
        for key, value in constants.items():
            f.write("{} = {}\n".format(key, value))

def setup_save_dir(save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)