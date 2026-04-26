#!/usr/bin/env python3
"""Three-layer RTA Controller"""

import numpy as np
from typing import Dict, Optional, Tuple

class RTAController:
    def __init__(self):
        self.d_min = 0.15
        self.zmp_margin = 0.03
        self.v_max = 1.0
        self.omega_max = 1.5
        self.r1_triggers = 0
        self.r2_triggers = 0
        self.r3_triggers = 0
    
    def check_region1(self, obs):
        if 'obstacles' in obs:
            for obs_item in obs['obstacles']:
                dist = np.sqrt((obs_item['x'] - obs['base'][0])**2 + (obs_item['y'] - obs['base'][1])**2)
                if dist < self.d_min:
                    self.r1_triggers += 1
                    return True, 'collision'
        if 'zmp_x' in obs and abs(obs['zmp_x']) > 0.27:
            self.r1_triggers += 1
            return True, 'zmp'
        if abs(obs['base'][3]) > self.v_max * 0.9:
            self.r1_triggers += 1
            return True, 'speed'
        return False, None
    
    def check_region2(self, obs, action):
        v = action.get('v', 0)
        pred_x = obs['base'][0] + v * np.cos(obs['base'][2])
        pred_y = obs['base'][1] + v * np.sin(obs['base'][2])
        risk = False
        if 'obstacles' in obs:
            for obs_item in obs['obstacles']:
                dist = np.sqrt((obs_item['x'] - pred_x)**2 + (obs_item['y'] - pred_y)**2)
                if dist < self.d_min + 0.2:
                    risk = True
                    break
        if risk:
            self.r2_triggers += 1
        return risk
    
    def check_region3(self, activations=None):
        if activations is None:
            return False, 0.0
        risk = activations.get('risk', 0)
        triggered = risk > 0.4
        if triggered:
            self.r3_triggers += 1
        return triggered, risk
    
    def get_safe_action(self, action, obs, activations=None, enable_r1=True, enable_r2=True, enable_r3=True):
        info = {'r1': False, 'r2': False, 'r3': False, 'risk': 0}
        
        # Region 3: discount action
        if enable_r3:
            r3, risk = self.check_region3(activations)
            info['r3'] = r3
            info['risk'] = risk
            if r3:
                action = {'v': action.get('v',0)*0.4, 'omega': action.get('omega',0)*0.4, 'tau': action.get('tau', np.zeros(7))*0.6}
        
        # Region 2: project to safe set
        if enable_r2:
            if self.check_region2(obs, action):
                info['r2'] = True
                action = {'v': action.get('v',0)*0.5, 'omega': action.get('omega',0)*0.5, 'tau': action.get('tau', np.zeros(7))}
        
        # Region 1: emergency brake
        if enable_r1:
            r1, reason = self.check_region1(obs)
            info['r1'] = r1
            if r1:
                action = {'v': -0.3, 'omega': 0, 'tau': np.zeros(7)}
        
        info['interventions'] = self.r1_triggers + self.r2_triggers + self.r3_triggers
        return action, info
    
    def reset(self):
        self.r1_triggers = self.r2_triggers = self.r3_triggers = 0
