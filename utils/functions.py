import os
import csv
import traci
import hydra
import random
import subprocess
import numpy as np
import pandas as pd
from scipy.special import factorial
import traci.constants as tc
import xml.etree.ElementTree as ET
from shutil import copyfile, copytree, rmtree
from utils.trafficLight import Signal
from utils.tingting import (ped_control, define_ring_barrier,
                            convert_split_by_ratio, generate_force_off_point,
                            input_conversion, create_phase_object, state_convert)

### unused class
class multientry:
    def __init__(self, sensor_id):
        self.id = sensor_id
        self.last = set()
        self.inter = set()
        self.current = set()
        self.volume = 0

    def compute(self, value):
        self.current = set(traci.multientryexit.getLastStepVehicleIDs(self.id))
        diff = self.current ^ self.last
        left = diff & self.inter
        if len(left) == 0:
            self.last = self.inter | diff
            self.volume = 0
        else:
            self.last = self.inter ^ diff
            self.volume = len(left)
        self.inter = self.last

### copy the folder "Excel_files" from main directory to individual run directories
def clone_files():
    copytree(os.path.join(hydra.utils.get_original_cwd(),"excel_files"), "excel_files")
    copyfile(os.path.join(hydra.utils.get_original_cwd(),"sumo_files/foothill.7am.sumocfg"), "foothill.7am.sumocfg")
    'sumo_files/foothill.7am.sumocfg'


def signal_status(cfg):
    if cfg.env.name == "foothill_v1":
        # read order file
        order = pd.read_csv('excel_files/order.csv',header=0)
        order['travel_time']=order['cum_distance']/40*3600
        order = order[['signalid','phaseid','travel_time']]
        order.columns=['signalid','phase','travel_time']
        # process signal status
        res3 = []
        fname = "signal_res_test.csv" if cfg.test.status else "signal_res_benchmark.csv"
        if cfg.test.status == True:
            test_path, used_agent =  os.path.split(cfg.test.load)
        else:
            test_path, used_agent =  os.path.split("benchmark/benchmark")
        save_path = hydra.utils.to_absolute_path(os.path.join("outputs", test_path))
        fpath = os.path.join(save_path, fname)
        res = pd.read_csv(fpath, header=0)
        res = res.sort_values(by=['signalid','t']).reset_index(drop=True)
        res2 = res.groupby(['signalid','t']).mean().reset_index()

        fname = os.path.join(save_path,f"{used_agent}_temp.csv") if cfg.test.status else "excel_files/plan_temp_7am_original.csv"
        plan_temp = pd.read_csv(fname, header=0)
        
        plan_temp = plan_temp[['signalid','offset','cycle_length']]
        for i in range(8):
            temp = res2[['signalid','t','phase%s'%(i+1)]]
            temp.columns = ['signalid','t','prob']
            temp['phase']=i+1
            temp['TOD']='7am'
            temp = pd.merge(temp, plan_temp, on=['signalid'], how='left')
            res3.append(temp)
        res3 = pd.concat(res3)
        res3['mt']=(res3['t']+res3['offset'])%(res3['cycle_length'])
        res4 = pd.merge(res3, order, on=['signalid','phase'], how='left')
        res4['mt'] = (res4['mt']-res4['travel_time'])%(res4['cycle_length'])

        if cfg.test.status:
            if cfg.env.special_volume != "None":
                final_fname = f"{cfg.env.special_volume}_greentime.csv"
            else:
                final_fname = "run_greentime.csv"
        else:
            final_fname = "benchmark_greentime.csv"
        res4.to_csv(os.path.join(save_path, final_fname), index=False)
    elif cfg.env.name == "foothill_v2":
        # read order file
        order = pd.read_csv('excel_files/order.csv',header=0)
        order['travel_time']=order['cum_distance']/40*3600
        order = order[['signalid','phaseid','travel_time']]
        order.columns=['signalid','phase','travel_time']
        # process signal status
        if cfg.test.status == True:
            test_path, used_agent =  os.path.split(cfg.test.load)
        else:
            test_path, used_agent =  os.path.split("benchmark/benchmark")
        save_path = hydra.utils.to_absolute_path(os.path.join("outputs", test_path))
        for csv_num in os.listdir(save_path):
            if csv_num.find("signal_res_test_") != -1:
                index = csv_num.split("_")[-1].split(".")[0]
                res3 = []
                fname = f"signal_res_test_{index}.csv" if cfg.test.status else f"signal_res_benchmark_{index}.csv"
                fpath = os.path.join(save_path, fname)
                res = pd.read_csv(fpath, header=0)
                res = res.sort_values(by=['signalid','t']).reset_index(drop=True)
                res2 = res.groupby(['signalid','t']).mean().reset_index()

                fname = os.path.join(save_path,f"{used_agent}_temp_{index}.csv") if cfg.test.status else "excel_files/plan_temp_7am_original.csv"
                plan_temp = pd.read_csv(fname, header=0)
                
                plan_temp = plan_temp[['signalid','offset','cycle_length']]
                for i in range(8):
                    temp = res2[['signalid','t','phase%s'%(i+1)]]
                    temp.columns = ['signalid','t','prob']
                    temp['phase']=i+1
                    temp['TOD']='7am'
                    temp = pd.merge(temp, plan_temp, on=['signalid'], how='left')
                    res3.append(temp)
                res3 = pd.concat(res3)
                res3['mt']=(res3['t']+res3['offset'])%(res3['cycle_length'])
                res4 = pd.merge(res3, order, on=['signalid','phase'], how='left')
                res4['mt'] = (res4['mt']-res4['travel_time'])%(res4['cycle_length'])

                if cfg.test.status:
                    if cfg.env.special_volume != "None":
                        final_fname = f"{cfg.env.special_volume}_greentime_{index}.csv"
                    else:
                        final_fname = f"run_greentime_{index}.csv"
                else:
                    final_fname = f"benchmark_greentime_{index}.csv"
                res4.to_csv(os.path.join(save_path, final_fname), index=False)

### Create signal function very similar to Tingting's function
def create_signals(self):
    for signalid in self.all_signals.keys():
        planid = int(self.plan_temp.loc[self.plan_temp['signalid']==int(signalid), 'planid'])
        offset = int(self.plan_temp.loc[self.plan_temp['signalid']==int(signalid), 'offset'])
        cycle_length = int(self.plan_temp.loc[self.plan_temp['signalid']==int(signalid), 'cycle_length'])
        table, ring1_start, ring2_start, lead_phase, phase_type, spec_phases = define_ring_barrier(self.plan, signalid = int(signalid), planid = planid)
        if self.config.benchmark.benchmark.status == True and self.config.benchmark.plan.find('original') != -1:
            split_temp = pd.merge(self.split, table, on=['signalid','phaseid'], how='right')
            split_temp = split_temp[['signalid','phaseid','split','min_green','yellow','rc','gap_out_max']]
            split_temp = split_temp.drop_duplicates() #sig 7221 split phasing has dup phase 3 and 4, cause problem for following procedure
        else:
            split_temp = convert_split_by_ratio(self.split, table, cycle_length = cycle_length) # 4/27 update, change split by random ratio from chromosomes
        table = generate_force_off_point(table, split_temp, ring1_start, ring2_start, lead_phase)
        table, ring1_index, ring2_index, start_phases_index, barrier1_index, barrier2_index = input_conversion(table, phase_type, spec_phases)
        phase_list = create_phase_object(table, cycle_length)
        init_state = np.repeat('r', len(phase_list))
        signal_object = Signal(phase_list, ring1_index, ring2_index, start_phases_index, barrier1_index, barrier2_index, init_state, "%s"%signalid)
        signal_object.offset = offset
        signal_object.cycle_length = cycle_length

        self.all_signals[signalid] = signal_object
        # if signalid == "7216":
        #     print(table)

### Update signal function very similar to Tingting's function
def update_signals(self, signalid):
    planid = int(self.plan_temp.loc[self.plan_temp['signalid']==int(signalid), 'planid'])
    offset = int(self.plan_temp.loc[self.plan_temp['signalid']==int(signalid), 'offset'])
    cycle_length = int(self.plan_temp.loc[self.plan_temp['signalid']==int(signalid), 'cycle_length'])
    table, ring1_start, ring2_start, lead_phase, phase_type, spec_phases = define_ring_barrier(self.plan, signalid = int(signalid), planid = planid)
    if self.config.benchmark.status == True and self.config.benchmark.plan.find('original') != -1:
        split_temp = pd.merge(self.split, table, on=['signalid','phaseid'], how='right')
        split_temp = split_temp[['signalid','phaseid','split','min_green','yellow','rc','gap_out_max']]
        split_temp = split_temp.drop_duplicates() #sig 7221 split phasing has dup phase 3 and 4, cause problem for following procedure
    else:
        split_temp = convert_split_by_ratio(self.split, table, cycle_length = cycle_length) # 4/27 update, change split by random ratio from chromosomes
    table = generate_force_off_point(table, split_temp, ring1_start, ring2_start, lead_phase)
    table, ring1_index, ring2_index, start_phases_index, barrier1_index, barrier2_index = input_conversion(table, phase_type, spec_phases)
    phase_list = create_phase_object(table, cycle_length)
    init_state = np.repeat('r', len(phase_list))
    signal_object = Signal(phase_list, ring1_index, ring2_index, start_phases_index, barrier1_index, barrier2_index, init_state, "%s"%signalid)
    signal_object.offset = offset
    signal_object.cycle_length = cycle_length
    self.all_signals[signalid] = signal_object
    return table

### Initialize sumo by setting the sumo command
def traci_start(self):
    if self.sumoBinary[-3:] == 'gui':
        sumoCmd = [self.sumoBinary, "-c", self.sumoCfg, "--start", "--pedestrian.model", "striping", "--seed", str(self.seeding)]
    else:
        sumoCmd = [self.sumoBinary, "-c", self.sumoCfg, "--start", "--no-warnings", "true", "--seed", str(self.seeding)]
    traci.start(sumoCmd)

### This function is used to quickly load a new environment setting without re-initializing it again  
def traci_load(self):
    if self.sumoBinary[-3:] == 'gui':
        traci.load(["-c", self.sumoCfg, "--start", "--pedestrian.model", "striping", "--seed", str(self.seeding)])
    else:
        traci.load(["-c", self.sumoCfg, "--start", "--no-warnings", "true", "--seed", str(self.seeding)])

### This function performs a replacement of any important files (reset) before calling traci_load
def traci_sumocfg_reset(self):
    if self.config.env.env.special_volume != "None":
        ped_path = os.path.join(self.cwd,"sumo_files/baseline_files/ped_files")
        route_path = os.path.join(self.cwd,"sumo_files/baseline_files/route_files")
        print("route_path",route_path)
        for i in os.listdir(ped_path):
            if i.split('.')[2] == self.config.env.special_volume:
                pfile = os.path.join("baseline_files/ped_files",i)
        for i in os.listdir(route_path):
            if i.split('.')[2] == self.config.env.special_volume:
                rfile = os.path.join("baseline_files/route_files",i)
    elif self.config.env.env.custom_volume != "None":
        rfile = os.path.join("custom_route_files", f"{elf.config.env.custom_volume}.xml")
        print("rfile",rfile)
    else:
        route_files = os.listdir(os.path.join(self.cwd, "sumo_files/baseline_files/route_files"))
        rfile = np.random.choice(route_files)
        rfile = os.path.join("route_files", rfile)
        print("rfile from else ",rfile)

    # self.hour, self.day = sampled_route_file.split('.')[2], sampled_route_file.split('.')[3]
    tree = ET.parse("foothill.7am.sumocfg")
    root = tree.getroot()
    original_sumo_path = os.path.join(self.cwd, "sumo_files")
    root[0][0].attrib['value'] = os.path.join(original_sumo_path, root[0][0].attrib['value'])
    root[0][1].attrib['value'] = os.path.join(original_sumo_path, rfile) + ", " + os.path.join(original_sumo_path, "dynamic/ped.route.7am.xml")
    root[0][2].attrib['value'] = os.path.join(original_sumo_path, "static/osm.poly.xml") + ", " \
                                + os.path.join(original_sumo_path, "static/det.gapout.add.xml") +  ", " \
                                + os.path.join(original_sumo_path, "dynamic/ped.tls.7am.xml")
    root[4][0].attrib['value'] = os.path.join(original_sumo_path, root[4][0].attrib['value'])
    tree.write('foothill.sumocfg')
    self.sumoCfg = 'foothill.sumocfg'

### retrive subscribed values from traci_subscribe. 
'''
Example format of a subscribed dictionary

loop = {712:{edge1: value, edge2: value, edge3: value}}

'''
def traci_retrieve(state_class):

    loop, edge, lanearea, multi = None, None, None, None

    if state_class.induction:
        loop = traci.inductionloop.getAllContextSubscriptionResults()
    if state_class.edge_reduced or state_class.edge_full or state_class.edge_full_side:
        edge = traci.edge.getAllSubscriptionResults()
    if state_class.lanearea:
        lanearea = traci.lanearea.getAllSubscriptionResults()
    if state_class.multi:
        multi = traci.multientryexit.getAllSubscriptionResults()

    return loop, edge, lanearea, multi

### subscribe to traci specific classes for sensors. If else statement here depends on the state function selected.
def traci_subscribe(self, state_class):

    if state_class.induction:
        traci.inductionloop.subscribeContext("7216_p1_gapout_0", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7217_p1_gapout_0", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7218_p1_gapout_0", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7219_p1_gapout", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7503_p4_gapout_0", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7220_p1_gapout", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7221_p3_gapout_0", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7222_p4_gapout_1", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7223_p4_gapout_1", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)
        traci.inductionloop.subscribeContext("7371_p4_gapout", tc.CMD_GET_INDUCTIONLOOP_VARIABLE, 100)

    if state_class.edge_reduced:
        for i in self.edge_dic["EB_edgeIDs_reduced"]:
            if isinstance(i, tuple):
                for inner in i:
                    traci.edge.subscribe(inner,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            else:
                traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        for i in self.edge_dic["WB_edgeIDs_reduced"]:
            if isinstance(i, tuple):
                for inner in i:
                    traci.edge.subscribe(inner,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            else:
                traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
    elif state_class.edge_full:
        for i in self.EB_edgeIDs:
            traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        for i in self.WB_edgeIDs:
            traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
    elif state_class.edge_full_side:
        for i in self.EB_edgeIDs:
            traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        for i in self.WB_edgeIDs:
            traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        for i in self.NS_edgeIDs:
            traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        for i in self.SB_edgeIDs:
            traci.edge.subscribe(i,[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])

    if state_class.lanearea:
        print("traci.lanearea ",traci.lanearea.getIDList())
        lanearea_list = traci.lanearea.getIDList()
        for i in lanearea_list:
            traci.lanearea.subscribe(i)

    if state_class.multi:
        multi_list = traci.multientryexit.getIDList()
        for i in multi_list:
            traci.multientryexit.subscribe(i,[tc.LAST_STEP_VEHICLE_ID_LIST])
        self.multi_dic = {}
        for i in multi_list:
            if i not in self.multi_dic:
                self.multi_dic[i] = multientry(i)

### hardcoded setup for state function. In this function, specific edge IDs are defined manually as a class member
def setup_customs(self):
    # static csv
    self.EB_edgeIDs = ["363648686#0", "363648686#2", "363648686#3", "363648686#5", "591388269#1", "591388269#1.160", \
                    "591388269#1.258", "692123188", "692513882", "692513882.119", "692513882.255", "692513873", \
                    "692513873.472", "146626844#2", "146626844#2.308", "146626844#5.190", "146626841#1", "146626839#0", \
                    "146626839#0.73", "207412432#0.28", "207412432#3", "207412432#4", "207412432#4.140", "697666080", \
                    "207412426#0", "697666084", "697666083#0", "697666083#0.332", "697666081#0", "697666082#1", "697666085#0", \
                    "697666085#0.242", "697666085#9", "697666085#12", "697666085#13", "697666085#13.295", "697666085#15.103", \
                    "696962877#0", "632830122"]
    
    self.WB_edgeIDs = ["-632830121#1", "-696962877#4", "-696962877#1.83", "-697666085#15", "-697666085#12", "-697666085#9", \
                    "-697666085#8", "-697666085#8.761", "-697666082#1", "-697666081#1", "-697666083#6", "-697666083#6.199", \
                    "-697666084", "-207412426#1", "-697666080", "-697666080.66", "-207412432#8", "-207412432#8.103", \
                    "-207412432#3", "-207412432#0", "-207412432#0.230", "600491439#0", "146626843#0", "146626843#0.250", \
                    "146626843#0.250.97", "146626843#2", "146626842#2", "207412421-AddedOnRampEdge", "591388255#1", \
                    "692513877#1", "692513877#1", "692123191", "692513879#1", "683763253#1", "683763253#1.209", "105826050#1"]

    self.NS_edgeIDs = ["749736413#1", "749736413#1.127", "-32921656#0", "683763238#1.51", "10124969#4", "10124969#6", "139051731#16", \
                        "139051731#19", "-146626846#5", "-146626846#0", "145218802#36", "145218802#37", "-69221466#3", "-69221466#0", \
                        "-69221466#0.18", "69232295#0", "508466788#0", "10133292", "10133290#0", "10147126#4"]
    
    self.SB_edgeIDs = ["10150738", "598977856#0", "621228442#0", "621228442#1", "684464938#1", "683763247#0", "591388258#0", "-172099960#8", \
                        "-172099960#1", "10131659#1", "10131659#5", "10131659#5.37", "145216628#1", "145216628#2", "-10150792#4", "-10150792#1", "-10137014#3"]

    self.edge_dic = {"EB_edgeIDs_reduced": 
                        ["363648686#5", "591388269#1.160", ("591388269#1.258", "692123188"), ("692513882.119", "692513882.255"), "692513873.472", \
                        "146626844#5.190", "207412432#0.28", "697666080", "697666081#0", ("697666085#0.242", "697666085#9"), "697666085#15.103"],

                    "WB_edgeIDs_reduced":
                        [("-696962877#1.83", "-696962877#4"), ("-697666085#15", "-697666085#12"), ("-697666085#8.761", "-697666082#1"), \
                        ("-697666084", "-207412426#1"), ("-207412432#8.103", "-207412432#3"), "600491439#0", "146626843#2", "591388255#1", \
                        "692123191", "692513879#1", "683763253#1.209"],

                    "SB_edgeIDs_reduced":
                        ["598977856#0", "621228442#1", "684464938#1", "591388258#0", ("-172099960#8", "-172099960#1"), "10131659#5.37", \
                        ("145216628#1", "145216628#2"), ("-10150792#4","-10150792#1"), "-10137014#3"],

                    "NB_edgeIDs_reduced":
                        ["10147126#4", ("10133292", "10133290#0"), ("69232295#0", "508466788#0"), ("-69221466#0", "-69221466#0.18"), \
                        ("145218802#36", "145218802#37"), "-146626846#0", "139051731#19", "10124969#6", ("-32921656#0", "683763238#1.51"), \
                        "749736413#1.127"]}

    self.plan = {}
    self.phase_state_index = {}

    with open(os.path.join('excel_files','plan.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)
        for row in csv_reader:
            if int(row[0]) not in self.plan:
                self.plan[int(row[0])] = {}
            if int(row[1]) not in self.plan[int(row[0])]:
                self.plan[int(row[0])][int(row[1])] = {"ring1_g1": row[2].split(","), \
                                                        "ring1_g2": row[3].split(","), \
                                                        "ring2_g1": row[4].split(","), \
                                                        "ring2_g2": row[5].split(","), \
                                                        "permissive": row[6].split(","), \
                                                        "split": row[7].split(",")}

    with open(os.path.join('excel_files',"phase_state_index.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)
        for row in csv_reader:
            if row[0] not in self.phase_state_index:
                self.phase_state_index[row[0]] = {}
            if row[1] not in self.phase_state_index[row[0]]:
                self.phase_state_index[row[0]][row[1]] = {}
            if row[2] not in self.phase_state_index[row[0]][row[1]]:
                self.phase_state_index[row[0]][row[1]][row[2]] = {"end_index": row[3], \
                                                    "right_turn_index": row[4], \
                                                    "permissive_left_start_index": row[5], \
                                                    "permissive_left_end_index": row[6]}

    # dynamic csv
    self.split = pd.read_csv(os.path.join('excel_files','phase_split_7am_v2.csv'), header=0)
    self.plan_temp = pd.read_csv(os.path.join('excel_files', 'plan_temp_7am_v2.csv'), header=0)

### function that takes appends the current timing plan to observation
def pack_timing_obs(self, obs):
    
    if (len(self.state) - 74) or (len(self.state) - 22) or (len(self.state) - 111) == 41:
        table_list = []
        idx = self.split.loc[self.split['signalid'] == 7216].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g1'])
        table_list.append(self.split.loc[idx, 'r1g2'])
        table_list.append(self.split.loc[idx, 'r2g1'])
        table_list.append(self.split.loc[idx, 'r2g2'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7216].index[0]
        table_list.append(self.plan_temp.loc[idx, 'planid'])
        table_list.append(self.plan_temp.loc[idx, 'offset'])
        table_list.append(self.plan_temp.loc[idx, 'cycle_length'])

        idx = self.split.loc[self.split['signalid'] == 7217].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g1'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7217].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7218].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g1'])
        table_list.append(self.split.loc[idx, 'r2g1'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7218].index[0]
        table_list.append(self.plan_temp.loc[idx, 'planid'])
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7219].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g1'])
        table_list.append(self.split.loc[idx, 'r1g2'])
        table_list.append(self.split.loc[idx, 'r2g1'])
        table_list.append(self.split.loc[idx, 'r2g2'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7219].index[0]
        table_list.append(self.plan_temp.loc[idx, 'planid'])
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7503].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7503].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7220].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g1'])
        table_list.append(self.split.loc[idx, 'r2g1'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7220].index[0]
        table_list.append(self.plan_temp.loc[idx, 'planid'])
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7221].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g2'])
        table_list.append(self.split.loc[idx, 'r2g2'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7221].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7222].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7222].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7223].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7223].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.split.loc[self.split['signalid'] == 7371].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7371].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7274].index[0]
        table_list.append(self.plan_temp.loc[idx, 'offset'])

    elif (len(self.state) - 73) or (len(self.state) - 22) == 8:
        table_list = []
        idx = self.split.loc[self.split['signalid'] == 7216].index[0]
        table_list.append(self.split.loc[idx, 'main_ratio'])
        table_list.append(self.split.loc[idx, 'r1g1'])
        table_list.append(self.split.loc[idx, 'r1g2'])
        table_list.append(self.split.loc[idx, 'r2g1'])
        table_list.append(self.split.loc[idx, 'r2g2'])

        idx = self.plan_temp.loc[self.plan_temp['signalid'] == 7216].index[0]
        table_list.append(self.plan_temp.loc[idx, 'planid'])
        table_list.append(self.plan_temp.loc[idx, 'offset'])
        table_list.append(self.plan_temp.loc[idx, 'cycle_length'])


    return np.hstack((obs, table_list)).astype(np.float32)
