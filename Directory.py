import os


def create_exp_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
