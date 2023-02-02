'''
Traffic Light object with controller logic (gap-out and force-off)
'''
class Phase:
    
    def __init__(self, cycle_length = 150, force_off_point = 20, min_green_duration = 5, yellow_period = 3, rc_period = 2, gap_out_max = 4, actual_number=1):
        
        self.actual_number = actual_number 
        self.force_off_point = force_off_point
        self.min_green_duration = min_green_duration
        self.yellow_period = yellow_period
        self.rc_period = rc_period
        self.gap_out_max = gap_out_max
        self.state = ""  # {'G','y','r'}
        self.min_green_timer = 0
        self.gap_out_timer = 0
        self.gap_out_flag = 0
        self.since_green_ends = 0
        self.done = 0
        self.gap_out_trigger_timer = -1
        self.cycle_length = cycle_length
        self.gap_out_counter = 0
        self.force_off_counter = 0
        
    def set_state(self, state):
        
        self.state = state
        
    def get_state(self):
        
        return self.state
    
    def green_starts(self):
        
        self.set_state('G')
        self.min_green_timer = 0
        self.gap_out_timer = 0
        self.gap_out_flag = 0
        self.since_green_ends = 0
        self.done = 0
        self.gap_out_trigger_timer = -1

    def yellow_starts(self):
        
        self.set_state('y')
        self.min_green_timer = 0
        self.gap_out_timer = 0
        self.gap_out_flag = 0
        self.since_green_ends = self.yellow_period + self.rc_period - self.force_off_point -1
        self.done = 0
        self.gap_out_trigger_timer = -1

    def red_starts(self):
        
        self.set_state('r')
        self.min_green_timer = 0
        self.gap_out_timer = 0
        self.gap_out_flag = 0
        self.since_green_ends = self.yellow_period + self.rc_period - self.force_off_point -1
        self.done = 0
        self.gap_out_trigger_timer = -1
        
    def update(self, occup, t):
        
        self.time = t
        
        if self.state == 'r':
            
            self.since_green_ends += 1
            
            if self.since_green_ends == self.yellow_period + self.rc_period:
                self.done = 1
                
        if self.state == 'y':
            
            self.gap_out_timer = 0
            self.gap_out_flag = 0
            
            self.since_green_ends += 1
            
            if self.since_green_ends == self.yellow_period:
                self.set_state('r')
        
        if self.state == 'G':
            
            if occup == 0:
                self.gap_out_timer += 1
            else:
                self.gap_out_timer = 0
                self.gap_out_flag = 0
            self.min_green_timer += 1

            self.check_force_off()
            self.check_gap_out()
            
            self.gap_out_trigger_timer -= 1
            
            if self.gap_out_trigger_timer == 0:
                self.gap_out_counter += 1
                self.set_state('y')
                
    def check_force_off(self):

        FO = self.force_off_point - self.yellow_period - self.rc_period

        if FO < 0:
            FO += self.cycle_length

        if self.time == FO:
            self.force_off_counter += 1            
            self.set_state('y')
            self.gap_out_timer = 0
            self.gap_out_flag = 0
        
    def check_gap_out(self):
        
        if (self.gap_out_timer == self.gap_out_max) and (self.min_green_timer >= self.min_green_duration):
            
            self.gap_out_flag = 1
        
    def trigger_gap_out(self, wait_time):
        
        self.gap_out_trigger_timer = wait_time
        
        if wait_time == 0:
            
            self.set_state('y')
            self.gap_out_timer = 0
            self.gap_out_flag = 0
            self.gap_out_trigger_timer = -1

class Signal:
    
    def __init__(self, phase_list, ring1_index, ring2_index, start_phases_index, barrier1_index, barrier2_index, \
                 state="", signalid = ""):
        '''
        phase_list: a list of 8 phase() 
        ring1_list: a list of 4 phases
        ring2_list: a list of 4 phases
        state: a string of 8 char - 'rrrGGyyr'
        start_phases: a list of 2 phases
        barrier1: a list of 2 phases
        barrier2: a list of 2 phases
        '''
        self.offset = 0
        self.cycle_length = 0
        self.phase_list = phase_list
        self.ring1_list = [phase_list[i] for i in ring1_index]
        self.ring2_list = [phase_list[i] for i in ring2_index]
        self.state = state
        self.signalid = signalid
        self.active_phases = [phase_list[i] for i in start_phases_index]
        self.barrier1 = set([phase_list[i] for i in barrier1_index])
        self.barrier2 = set([phase_list[i] for i in barrier2_index])
        
        for i in range(len(self.phase_list)):
            self.phase_list[i].set_state(state[i])
        
        for p in self.active_phases:
            if p.force_off_point > p.yellow_period + p.rc_period:
                p.green_starts()
            elif p.force_off_point > p.rc_period:
                p.yellow_starts()
            else:
                p.red_starts()
        
    def update(self, occup_list, t):
        
        p1 = self.active_phases[0]
        p2 = self.active_phases[1]
        
        if p1.actual_number in [1,2,5,6]:
            if p2.actual_number in [3,4,7,8]:
                raise ValueError('Signal: %s, Conflict Phases: %s and %s are both active.'%(self.signalid, p1.actual_number, p2.actual_number))
        else:
            if p2.actual_number in [1,2,5,6]:
                raise ValueError('Signal: %s, Conflict Phases: %s and %s are both active.'%(self.signalid, p1.actual_number, p2.actual_number))
                
        for p in self.active_phases:
            p_occ = occup_list[self.phase_list.index(p)]
            
            p.update(p_occ,t)
        
            if p.done == 1:
                p_next_phase = self.find_the_next_phase(p)
                p_index = self.active_phases.index(p)
                self.active_phases[p_index] = p_next_phase
                p_next_phase.green_starts()
                
        p1_gap_out_flag = p1.gap_out_flag
        p2_gap_out_flag = p2.gap_out_flag
        
        if (p1 not in self.barrier1) and (p1 not in self.barrier2) and (p1_gap_out_flag == 1):
            p1.set_state('y')
            
        if (p2 not in self.barrier1) and (p2 not in self.barrier2) and (p2_gap_out_flag == 1):
            p2.set_state('y')
            
        if (set([p1, p2]) == self.barrier1) or (set([p1, p2]) == self.barrier2):
            
            if p1_gap_out_flag == p2_gap_out_flag == 1:
                
                p1_wait_time,p2_wait_time = self.find_wait_time(p1,p2)
                
                p1.trigger_gap_out(p1_wait_time)
                p2.trigger_gap_out(p2_wait_time)

    def find_wait_time(self,p1,p2):
        
        p1_total = p1.yellow_period + p1.rc_period
        p2_total = p2.yellow_period + p2.rc_period
        
        return (max(0,p2_total-p1_total),max(0,p1_total-p2_total))
            
    def find_the_next_phase(self,p):
        
        if p in self.ring1_list:
            p_index = self.ring1_list.index(p)
            next_index = (p_index+1)%len(self.ring1_list)
            return self.ring1_list[next_index]
        
        if p in self.ring2_list:
            p_index = self.ring2_list.index(p)
            next_index = (p_index+1)%len(self.ring2_list)
            return self.ring2_list[next_index]
        
    def get_state(self):
        
        state = ""
        for p in self.phase_list:
            state += p.state
            
        return state
    
    def get_phase_seq(self):
        
        phases = self.ring1_list + self.ring2_list
        seq = []
        for p in phases:
            seq.append(p.actual_number)
            
        return seq