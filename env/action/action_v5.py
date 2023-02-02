import numpy as np
from gym import spaces
import pandas as pd
import os

class action_v5:

    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dic = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
        self.spaces = spaces.Discrete(2)
        self.spaces.seed(cfg.action.seed)

    def space(self):
        return self.spaces

    def process(self, split, temp, action):

        if action == 0:
            split = pd.read_csv(os.path.join('excel_files','phase_split_7am_v2.csv'), header=0)
            plan_temp = pd.read_csv(os.path.join('excel_files', 'plan_temp_7am_v2.csv'), header=0)
        elif action == 1:
            split = pd.read_csv(os.path.join('excel_files','phase_split_7am.csv'), header=0)
            plan_temp = pd.read_csv(os.path.join('excel_files', 'plan_temp_7am.csv'), header=0)


        return split, temp

    def save_csv(self, path, fname, split, temp):
        split.to_csv(f"{path}/{fname}_split.csv")
        temp.to_csv(f"{path}/{fname}_temp.csv")