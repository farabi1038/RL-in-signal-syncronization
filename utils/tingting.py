import pandas as pd
import numpy as np
import traci
from utils.trafficLight import Phase
import timeit

def state_convert(df, signalid, phase_seq, state):

    temp = df[signalid]
    length = 0
    for i in temp.keys():
        for j in temp[i].keys():
            if int(temp[i][j]['end_index']) > length:
                length = int(temp[i][j]['end_index'])

    sumo_temp = list(np.repeat('r', length))
    sumo_phase_list = set(np.array(list(temp.keys())).astype(np.int64))
    phase_seq = set(phase_seq)
    intersect_phase = sumo_phase_list & phase_seq

    for i in intersect_phase:
        for j in temp[str(i)].keys():
            start = int(j)
            end = int(temp[str(i)][str(j)]['end_index'])
            right = int(temp[str(i)][str(j)]['right_turn_index'])
            left_start = int(temp[str(i)][str(j)]['permissive_left_start_index'])
            left_end = int(temp[str(i)][str(j)]['permissive_left_end_index'])
            phasestr = ''
            color = state[i-1] # only work for full 8 phase list, be carefull with split phasing
            for t in range(end-start):
                phasestr+=color
            sumo_temp[start:end]=phasestr
            # right movement color
            if right<900 and sumo_temp[right]=='r':
                sumo_temp[right] = 's'
            # left movement colors
            phasestr = ''
            color = state[i-1] # only work for full 8 phase list, be carefull with split phasing
            if color == 'G':
                color = 'g'
            for l in range(left_end-left_start):
                phasestr+=color
            sumo_temp[left_start:left_end]=phasestr
 
    sumo_state = ''.join(sumo_temp)
    
    return sumo_state

def ped_control(now, trigger):
    """Define the logic of a pedestrain signal.

    This signal has only pedestrain crossing and no gap out and force off logic needed.
    Simple tls program can be used and put trigger to start the switch when pedestrain occur.

    Args:
        now: system time, global time
        trigger: similar to a ped button, init as 0 from system start, check and update every iteration.

    Returns:
        trigger: updated trigger status and traffic light status change, if applied
    """

    nb_edge = ":7274_w0" # the wait edge
    sb_edge = ":7274_w1"

    index = traci.trafficlight.getPhase("7274")
    duration = traci.trafficlight.getPhaseDuration("7274")
    remaining = traci.trafficlight.getNextSwitch("7274")-now

    nb_person = traci.edge.getLastStepPersonIDs(nb_edge)
    sb_person = traci.edge.getLastStepPersonIDs(sb_edge)

    if trigger == 0:
        for person in nb_person:
            nb_next = traci.person.getNextEdge(person)
            if nb_next == ":7274_c0": # the crossing edge
                trigger = 1
                break
        for person in sb_person:
            sb_next = traci.person.getNextEdge(person)
            if sb_next == ":7274_c0":
                trigger = 1
                break
        
    if index == 0 and remaining == 0 and trigger:
        next_index = index + 1
        traci.trafficlight.setPhase("7274", next_index)
        
    if index == 0:
        trigger = 0

    return trigger

def decide_type(test):
    spec_phases = []
    status = None
    '''not need to check permissive'''
    # if test['permissive'].isnull().sum()==0: 
    #     phase_type = 'permissive'
    #     spec_phases = test.loc[0, 'permissive'].split(',')

    # if test['split'].isnull().sum()==0: # .sum() is checking how many null value, equal to 0 means not nan
    #     phase_type = 'split'
    #     spec_phases = test.loc[0, 'split'].split(',')
    # else:
    #     phase_type = 'protected/permissive'
    # spec_phases = [int(item) for item in spec_phases]
    # return phase_type, spec_phases

    if len(test['split']) == 1:
        try:
            isinstance(int(test['split'][0]), int)
            status = True
        except ValueError:
            status = False
    if status == True or len(test['split']) > 1:
        phase_type = 'split'
        spec_phases = test['split']
    else:
        phase_type = 'protected/permissive'
    spec_phases = [int(item) for item in spec_phases]
    return phase_type, spec_phases



def decide_order(df, ring):
    phaseid1 = list(df.loc[df['ring']==ring,'phaseid'])
    splits1 = list(df.loc[df['ring']==ring,'split'])
    starts1 = list(df.loc[df['ring']==ring,'ring%s_start'%ring])
    order1 = list(np.repeat(0, len(starts1)))
    ind = starts1.index(1)
    for i in range(len(starts1)):
        new_ind = (ind+i)%len(starts1)
        order1[new_ind]=(i+1)
    temp = pd.DataFrame()
    temp['phaseid']=phaseid1
    temp['split']=splits1
    temp['ring%s_start'%ring]=starts1
    temp['order']=order1
    df2 = df.merge(temp, on=['phaseid', 'split','ring%s_start'%ring], how='left')
    return df2


def define_ring_barrier(plan, signalid, planid):
    # get plan
    test = plan[signalid][planid]
    phase_type, spec_phases = decide_type(test)
    # create indicator for ring, group and barrier
    ring1_g1 = test["ring1_g1"]
    ring1_g2 = test["ring1_g2"]
    ring2_g1 = test["ring2_g1"]
    ring2_g2 = test["ring2_g2"]
    ring1 = ring1_g1 + ring1_g2
    ring2 = ring2_g1 + ring2_g2
    ring1 = [int(item) for item in ring1]
    ring2 = [int(item) for item in ring2]
    ring = [1 for item in ring1]+[2 for item in ring2]
    group = [1 for item in ring1_g1] + [2 for item in ring1_g2] + [1 for item in ring2_g1] + [2 for item in ring2_g2]
    barrier = [0 for item in ring1_g1[:-1]] + [1 for item in ring1_g1[-1]] + [0 for item in ring1_g2[:-1]] + [1 for item in ring1_g2[-1]] + \
              [0 for item in ring2_g1[:-1]] + [1 for item in ring2_g1[-1]] + [0 for item in ring2_g2[:-1]] + [1 for item in ring2_g2[-1]]
    # get lead phase and potential start phase
    pos2 = ring1.index(2)
    pos6 = ring2.index(6)
    ring1_start = 0
    ring2_start = 0
    lead_phase = 0
    if pos2 < pos6:
        ring1_start = ring1[pos2 + 1]
        lead_phase = 2
        #print 'phase 2 lead'
    elif pos2 > pos6:
        ring2_start = ring2[pos6 + 1]
        lead_phase = 6
        #print 'phase 6 lead'
    else:
        #print 'phase 2&6 together'
        if pos2 == pos6 == 0:
            ring1_start = ring1[pos2 + 1]
            lead_phase = 2
        else:
            ring1_start = ring1[pos2 + 1]
            ring2_start = ring2[pos6 + 1]


    # create dataframe
    df = pd.DataFrame()
    df['phaseid']=ring1+ring2
    df['ring']=ring
    df['group']=group
    df['barrier']=barrier
    df['signalid']=signalid
    
    return df, ring1_start, ring2_start, lead_phase, phase_type, spec_phases


def convert_split_by_ratio(split_temp, table, cycle_length):
    cols = [c for c in split_temp.columns if (('start' in c) | ('end' in c))]
    split_temp = split_temp.drop(cols, axis=1)
    
    def get_split(x):
        if x['ring']==1:
            if x['group']==1:
                ratio = x['r1g1']
            else:
                ratio = x['r1g2']
        else:
            if x['group']==1:
                ratio = x['r2g1']
            else:
                ratio = x['r2g2']

        if not np.isnan(ratio):
            split1 = int((x['group_length']-x['groupMP'])*ratio)+x['MP']
            if x['phaseid']%2==0:
                split=999 # temp
            else:
                split=split1
        else:
            split=x['group_length']
            
        return split

    split_temp = pd.merge(split_temp, table, on=['signalid','phaseid'], how='right')
    split_temp['cycle_length']=cycle_length
    split_temp['groupMP']=split_temp.groupby(['signalid','ring','group'])['MP'].transform('sum')
    split_temp['groupMaxMP']=split_temp.groupby(['signalid','group'])['groupMP'].transform('max')
    split_temp['maxMP']=split_temp.groupby(['signalid'])['groupMaxMP'].transform('min')+split_temp.groupby(['signalid'])['groupMaxMP'].transform('max')
    split_temp['main_green']=((split_temp['cycle_length']-split_temp['maxMP'])*split_temp['main_ratio']).astype('int')
    split_temp['side_green']=(split_temp['cycle_length']-split_temp['maxMP'])-split_temp['main_green']
    if min(split_temp['main_green'])<0:
        raise ValueError('Minimum cycle length is smaller than sum of minimum phase time.')
    split_temp['group_length']=0
    split_temp.loc[split_temp['group']==1, 'group_length']=split_temp.loc[split_temp['group']==1, 'groupMaxMP']+split_temp.loc[split_temp['group']==1, 'main_green']
    split_temp.loc[split_temp['group']==2, 'group_length']=split_temp.loc[split_temp['group']==2, 'groupMaxMP']+split_temp.loc[split_temp['group']==2, 'side_green']
    split_temp['split']=split_temp.apply(lambda x: get_split(x), axis=1)
    split_temp['split']=split_temp['split'].astype('int')
    split_temp['alt_split']=split_temp.groupby(['signalid','ring','group'])['split'].transform('min')
    split_temp.loc[split_temp['split']==999, 'split']=split_temp.loc[split_temp['split']==999, 'group_length']-split_temp.loc[split_temp['split']==999, 'alt_split']
    split_temp = split_temp[['signalid','phaseid','split','min_green','yellow','rc','gap_out_max']]
    split_temp = split_temp.drop_duplicates() #sig 7221 split phasing has dup phase 3 and 4, cause problem for following procedure

    return split_temp


def generate_force_off_point(df, split, ring1_start, ring2_start, lead_phase):
    
    df = pd.merge(df, split, on = ['signalid','phaseid'], how='left')
    df['ring1_start']=0
    df['ring2_start']=0
    df.loc[df['phaseid']==ring1_start,'ring1_start']=1
    df.loc[df['phaseid']==ring2_start,'ring2_start']=1
    df['FO_init'] = df.groupby(by=['ring'])['split'].cumsum()
    
    if lead_phase == 6:
        df.loc[(df['ring']==1) & (df['FO_init']>int(df.loc[df['phaseid']==lead_phase, 'split'])), 'ring1_start']=1 
    if lead_phase == 2:
        df.loc[(df['ring']==2) & (df['FO_init']>int(df.loc[df['phaseid']==lead_phase, 'split'])), 'ring2_start']=1 
    
    df = decide_order(df, 1)
    df = decide_order(df, 2)
    df['order']=df.apply(lambda x: x.order_x if np.isnan(x.order_y) else x.order_y, axis=1)
    df = df.drop(['order_x','order_y'], axis=1)
    
    df = df.sort_values(by=['ring','order'])
    df['FO_2']=df.groupby(['ring'])['split'].cumsum()
    
    # reset start index
    df['ring1_start']=0
    df.loc[(df['ring']==1)&(df['order']==1), 'ring1_start']=1
    df['ring2_start']=0
    df.loc[(df['ring']==2)&(df['order']==1), 'ring2_start']=1
    
    # adjust if phase go over y=0 point
    adjust_value = 0
    if lead_phase == 6:
        adjust_value = int(df.loc[(df['ring']==2) & (df['order']==1), 'FO_init']-df.loc[(df['ring']==2) & (df['order']==1), 'FO_2'])
        df.loc[(df['ring']==1) & (df['order']==1), 'FO_2'] = df.loc[(df['ring']==1) & (df['order']==1), 'FO_init']-adjust_value
    if lead_phase == 2:
        adjust_value = int(df.loc[(df['ring']==1) & (df['order']==1), 'FO_init']-df.loc[(df['ring']==1) & (df['order']==1), 'FO_2'])
        df.loc[(df['ring']==2) & (df['order']==1), 'FO_2'] = df.loc[(df['ring']==2) & (df['order']==1), 'FO_init']-adjust_value
    
    # reset and redo
    df.loc[df['order']>1,'FO_2']=df.loc[df['order']>1,'split']
    df = df.sort_values(by=['ring','order'])
    df['FO']=df.groupby(['ring'])['FO_2'].cumsum()
    df = df.drop(['FO_init', 'FO_2'], axis=1)
    
    return df

def input_conversion(df, phase_type, spec_phases):
    df['pseudo_phase'] = df['phaseid']
    if phase_type == 'split':
        for spec_phase in spec_phases:
            df.loc[(df['ring']==2)&(df['phaseid']==spec_phase), 'pseudo_phase']=spec_phase+4
    ring1_index = list(df.loc[df['ring']==1,'pseudo_phase'])
    ring1_index = [int(item)-1 for item in ring1_index]
    ring2_index = list(df.loc[df['ring']==2,'pseudo_phase'])
    ring2_index = [int(item)-1 for item in ring2_index]
    start_phases_index = list(df.loc[df['ring1_start']==1,'pseudo_phase'])+list(df.loc[df['ring2_start']==1,'pseudo_phase'])
    start_phases_index = [int(item)-1 for item in start_phases_index]
    barrier1_index = list(df.loc[(df['group']==1)&(df['barrier']==1),'pseudo_phase'])
    barrier1_index = [int(item)-1 for item in barrier1_index]
    barrier2_index = list(df.loc[(df['group']==2)&(df['barrier']==1),'pseudo_phase'])
    barrier2_index = [int(item)-1 for item in barrier2_index]
    return df, ring1_index, ring2_index, start_phases_index, barrier1_index, barrier2_index


def create_phase_object(df, cycle_length):
    phases = list(df['pseudo_phase'])
    force_off_point = 0
    min_green_duration = 0
    yellow_period = 0
    rc_period = 0
    gap_out_max = 0
    actual_number = 0
    phase_list = []
    for i in range(8):
        if i+1 in phases:
            force_off_point = int(df.loc[df['pseudo_phase']==i+1, 'FO'])
            min_green_duration = int(df.loc[df['pseudo_phase']==i+1, 'min_green'])
            yellow_period = int(df.loc[df['pseudo_phase']==i+1, 'yellow'])
            rc_period = int(df.loc[df['pseudo_phase']==i+1, 'rc'])
            gap_out_max = int(df.loc[df['pseudo_phase']==i+1, 'gap_out_max'])
            actual_number = int(df.loc[df['pseudo_phase']==i+1, 'phaseid'])
        phase = Phase(cycle_length, force_off_point, min_green_duration, yellow_period, rc_period, gap_out_max, actual_number)
        phase_list.append(phase)
    return phase_list
