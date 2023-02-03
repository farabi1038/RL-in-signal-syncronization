from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import sys
import csv
import glob
import time
import hydra
import traci
import numpy as np
import pandas as pd
from gym import spaces
import traci.constants as tc
from utils import evaluation
from functools import partial
from sumolib import checkBinary
from env.state.state_selector import state_selector
from env.action.action_selector import action_selector
from env.reward.reward_selector import reward_selector
from utils.trafficLight import Signal
from utils.functions import (clone_files, create_signals, pack_timing_obs,
                                setup_customs, traci_subscribe, traci_retrieve, 
                                traci_load, traci_start, traci_sumocfg_reset)
from utils.tingting import (ped_control, define_ring_barrier,
                            convert_split_by_ratio, generate_force_off_point,
                            input_conversion, create_phase_object, state_convert)

class foothill_v1:
    '''
    foothill_v1 is formulated such that one env.step() is equals to one full simulation. i.e.: 5 or 15 minute simulation using one timing plan
    '''
    def __init__(self, cfg, logger):
        '''
        Define init variables for environment

        '''
        self.config = cfg
        self.logger = logger
        self.cwd = hydra.utils.get_original_cwd()
        self.seeding = None

        self.state = state_selector(self.config) # --> returns a function
        self.action = action_selector(self.config) # --> returns a spaces object
        self.reward = reward_selector(self.config) # --> returns a function

        self.t = 0

        clone_files()
        setup_customs(self)

        self.all_signals = {"7216":None,"7217":None,"7218":None,"7219":None,"7503":None,"7220":None,"7221":None,"7222":None,"7223":None,"7371":None}
        self.multi_dic = {}

        self.sumoCfg = os.path.join(self.cwd,self.config.env.env.source)
        self.sumoBinary = checkBinary('sumo')

    def seed(self, seed):
        '''
        Seed sumo simulation to ensure reproducibility.

        '''
        self.seeding = seed
        np.random.seed(seed)

    def render(self):
        '''
        Edit sumoBinary for GUI

        '''
        self.sumoBinary = checkBinary('sumo-gui')

    def init(self):
        '''
        Initialize sumo with given sumo xml files provided in the config/env .yaml file.

        '''
        traci_start(self)
        traci_load(self)

        self.detectorlist = traci.inductionloop.getIDList()

        self.dict = {}
        for i in self.detectorlist:
            if i[0:4] not in self.dict:
                self.dict[i[0:4]] = np.zeros(8)

    def reset(self):
        '''
        Reloads simulation without killing the pipeline.

        '''
        self.t = 0
        for i in self.dict.keys():
            self.dict[i] = np.zeros(8)
        self.state.reset()
        if self.config.env.env.randomize:
            traci_sumocfg_reset(self)
        traci_load(self)

        if self.config.train.train.status == True or self.config.test.status == True:
            splitname = self.config.train.train.split
            planname = self.config.train.train.plan
        elif self.config.benchmark.status == True:
            splitname = self.config.benchmark.split
            planname = self.config.benchmark.plan

        self.split = pd.read_csv(os.path.join('excel_files', splitname), header=0)
        self.plan_temp = pd.read_csv(os.path.join('excel_files', planname), header=0)
        create_signals(self)
        traci_subscribe(self, self.state)

        step = 1
        trigger = 0
        now = traci.simulation.getTime()
        trigger = ped_control(now, trigger)

        for signalid in self.all_signals.keys():
            signal_object = self.all_signals[signalid]
            for i in range(signal_object.cycle_length-signal_object.offset):
                occ = [1,1,1,1,1,1,1,1]
                signal_object.update(occ, i)
            state = signal_object.get_state()
            phase_seq = signal_object.get_phase_seq()
            sumo_state = state_convert(self.phase_state_index, signalid, phase_seq, state)
            traci.trafficlight.setRedYellowGreenState("%s"%signalid, sumo_state)
        
        while step < self.config.env.warmup_steps:
            now = traci.simulation.getTime()
            trigger = ped_control(now, trigger)
            loop, edge, lanearea, multi = traci_retrieve(self.state)
            for item in loop:
                lvl_1 = item[0:4]
                for phases in loop[item]:
                    lvl_2 = int(phases[6:7]) - 1
                    self.dict[lvl_1][lvl_2] += list(loop[item][phases].values())[0]
            for signalid in self.all_signals.keys():
                signal_object = self.all_signals[signalid]
                t = (step-1+signal_object.cycle_length-signal_object.offset)%signal_object.cycle_length
                signal_object.update(self.dict[signalid], t)
                state = signal_object.get_state()
                phase_seq = signal_object.get_phase_seq()
                sumo_state = state_convert(self.phase_state_index, signalid, phase_seq, state)
                traci.trafficlight.setRedYellowGreenState("%s"%signalid, sumo_state)
            step += 1

        if self.config.env.warmup:
            total = self.config.env.local_resolution + self.config.env.warmup_steps
        else:
            total = self.config.env.local_resolution

        while step < total:
            now = traci.simulation.getTime()
            trigger = ped_control(now, trigger)
            loop, edge, lanearea, multi= traci_retrieve(self.state)
            for item in loop:
                lvl_1 = item[0:4]
                for phases in loop[item]:
                    lvl_2 = int(phases[6:7]) - 1
                    self.dict[lvl_1][lvl_2] += list(loop[item][phases].values())[0]
            for signalid in self.all_signals.keys():
                signal_object = self.all_signals[signalid]
                t = (step-1+signal_object.cycle_length-signal_object.offset)%signal_object.cycle_length
                signal_object.update(self.dict[signalid], t)
                state = signal_object.get_state()
                phase_seq = signal_object.get_phase_seq()
                sumo_state = state_convert(self.phase_state_index, signalid, phase_seq, state)
                traci.trafficlight.setRedYellowGreenState("%s"%signalid, sumo_state)
            self.state.update(loop, edge, lanearea, multi, self)
            traci.simulationStep()
            step += 1
        self.t += 1

        obs = self.state.retrieve()
        obs = pack_timing_obs(self, obs)

        return obs

    def step_local(self, action):
        for i in self.dict.keys():
            self.dict[i] = np.zeros(8)
        self.state.reset()
        if self.config.env.per_step:
            if self.config.env.randomize:
                traci_sumocfg_reset(self)
        traci_load(self)
        if self.config.train.status == True or self.config.test.status == True:
            self.split, self.plan_temp = self.action.process(self.split, self.plan_temp, action)

        create_signals(self)
        traci_subscribe(self, self.state)

        sped = 0
        step = 1
        trigger = 0
        track = {}
        now = traci.simulation.getTime()
        trigger = ped_control(now, trigger)

        for signalid in self.all_signals.keys():
            signal_object = self.all_signals[signalid]
            for i in range(signal_object.cycle_length-signal_object.offset):
                occ = [1,1,1,1,1,1,1,1]
                signal_object.update(occ, i)
            state = signal_object.get_state()
            phase_seq = signal_object.get_phase_seq()
            sumo_state = state_convert(self.phase_state_index, signalid, phase_seq, state)
            traci.trafficlight.setRedYellowGreenState("%s"%signalid, sumo_state)

        while step < self.config.env.warmup_steps:
            now = traci.simulation.getTime()
            trigger = ped_control(now, trigger)
            loop, edge, lanearea, multi = traci_retrieve(self.state)
            for item in loop:
                lvl_1 = item[0:4]
                for phases in loop[item]:
                    lvl_2 = int(phases[6:7]) - 1
                    self.dict[lvl_1][lvl_2] += list(loop[item][phases].values())[0]
            for signalid in self.all_signals.keys():
                signal_object = self.all_signals[signalid]
                t = (step-1+signal_object.cycle_length-signal_object.offset)%signal_object.cycle_length
                signal_object.update(self.dict[signalid], t)
                state = signal_object.get_state()
                phase_seq = signal_object.get_phase_seq()
                sumo_state = state_convert(self.phase_state_index, signalid, phase_seq, state)
                traci.trafficlight.setRedYellowGreenState("%s"%signalid, sumo_state)
            step += 1

        if self.config.env.warmup:
            total = self.config.env.local_resolution + self.config.env.warmup_steps
        else:
            total = self.config.env.local_resolution

        while step < total:
            now = traci.simulation.getTime()
            trigger = ped_control(now, trigger)
            loop, edge, lanearea, multi = traci_retrieve(self.state)
            for item in loop:
                lvl_1 = item[0:4]
                for phases in loop[item]:
                    lvl_2 = int(phases[6:7]) - 1
                    self.dict[lvl_1][lvl_2] += list(loop[item][phases].values())[0]
            for signalid in self.all_signals.keys():
                signal_object = self.all_signals[signalid]
                t = (step-1+signal_object.cycle_length-signal_object.offset)%signal_object.cycle_length
                signal_object.update(self.dict[signalid], t)
                state = signal_object.get_state()
                phase_seq = signal_object.get_phase_seq()
                sumo_state = state_convert(self.phase_state_index, signalid, phase_seq, state)
                traci.trafficlight.setRedYellowGreenState("%s"%signalid, sumo_state)

                if self.config.test.status == True or self.config.benchmark.status == True:
                    # if test or benchmark mode, enter this if loop
                    if self.config.test.status == True:
                        test_path, used_agent =  os.path.split(self.config.test.load)
                    else:
                        test_path, used_agent =  os.path.split("benchmark/benchmark")
                    save_path = hydra.utils.to_absolute_path(os.path.join("outputs", test_path))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # save timing plan
                    self.action.save_csv(save_path, used_agent, self.split, self.plan_temp)

                    # save data for probabilistic plot
                    green = np.array([0,0,0,0,0,0,0,0])
                    active_phases_index = [p.actual_number-1 for p in signal_object.active_phases]
                    green[active_phases_index] = 1
                    fname = "signal_res_test.csv" if self.config.test.status else "signal_res_benchmark.csv"
                    fpath = os.path.join(save_path, fname)
                    if not os.path.isfile(fpath):
                        with open(fpath, 'a') as f:
                            f.write("signalid,t,phase1,phase2,phase3,phase4,phase5,phase6,phase7,phase8\n")

                    with open(fpath, 'a') as f:
                        f.write(f"{signalid},{t},")
                        f.write(f"{green[0]},{green[1]},{green[2]},{green[3]},{green[4]},{green[5]},{green[6]},{green[7]}\n")

            self.state.update(loop, edge, lanearea, multi, self)
            traci.simulationStep()

            
            depart_list = traci.simulation.getDepartedIDList()
            for pot_veh in depart_list:
                if traci.vehicle.getRouteID(pot_veh) == "1_2":
                    track[pot_veh] = pot_veh
                elif traci.vehicle.getRouteID(pot_veh) == "2_1":
                    track[pot_veh] = pot_veh
                else:
                    pass

            arrive_list = traci.simulation.getArrivedIDList()
            for pot_veh in arrive_list:
                if pot_veh in track:
                    del track[pot_veh]
                else:
                    pass
            for veh in track.keys():
                if traci.vehicle.getSpeed(veh) > 0.0:
                    sped += (1/traci.vehicle.getSpeed(veh)) * 0.05988
                else:
                    sped += (1/0.1) * 0.05988

            step += 1
        self.t += 1

        obs = self.state.retrieve()
        reward = self.reward(self.state, sped, self.config)
        obs = pack_timing_obs(self, obs)

        return obs, reward
