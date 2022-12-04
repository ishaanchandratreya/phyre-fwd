import sys
import os

BASE_LINK = 'https://dl.fbaipublicfiles.com/phyre-fwd-agents/'
EXPT_PATH = 'expts'
WRITE_PATH = 'saved_models'


def search_for_models():


    final_expt_path = os.path.join(os.getcwd(), EXPT_PATH)
    final_write_path = os.path.join(os.getcwd(), WRITE_PATH)
    expt_folder_type = os.listdir(final_expt_path)
    for each_type in expt_folder_type:
        current_exp_folder = os.path.join(final_expt_path, each_type)
        current_exp_store = os.path.join(final_write_path, each_type)
        if not os.path.exists(current_exp_store):
            os.makedirs(current_exp_store)




