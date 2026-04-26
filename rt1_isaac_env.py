"""
RT-1 Isaac Lab Environment with Random Obstacle Generation
Creates RT1_Semantic_Navigation_Env task for Fetch mobile manipulator
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment configuration"""
    # Workspace
    workspace_bounds: Tuple[float, float, float, float, float, float] = (-5, 15, -5, 5, 0, 2.5)
    
    # Start/Goal
    start_pos: np.ndarray = None
    goal_pos: np.ndarray = None
    goal_tolerance: float = 0.5
    
    # Obstacles
    static_obstacles_min: int = 3
    static_obstacles_max: int = 8
    dynamic_obstacle_prob: float = 0.3
    dynamic_obstacles_min: int = 1
    dynamic_obstacles_max: int = 2
    dynamic_obstacle_speed: float = 0.5  # m/s
    
    # Visual OOD
    visual_ood_prob: float = 0.2
    
    # Control
    control_freq: int = 50  # Hz
    dt: float = 0.02
    
    # Camera
    camera_width: int = 320
    camera_height: int = 256
    
    def __post_init__(self):
        if self.start_pos is None:
            self.start_pos = np.array([0.0, 0.0, 0.0])
        if self.goal_pos is None:
            self.goal_pos = np.array([10.0, 0.0, 0.0])


class RT1IsaacEnv:
    """
    RT-1 Semantic Navigation Environment in Isaac Lab
    
    Features:
    - Fetch mobile manipulator with RGB camera
    - Random obstacle generation (static + dynamic)
    - Visual OOD triggers (lighting changes, adversarial textures)
    - Language-conditioned navigation
    - Ground truth outputs for Region 1/2 RTA
    """
    
    def __init__(self, config: Optional[EnvConfig] = None, scenario: Optional[Dict] = None):
        self.config = config or EnvConfig()
        self.scenario = scenario
        
        # State
        self.robot_pos = None  # [x, y, z]
        self.robot_theta = None  # Heading angle (rad)
        self.robot_vel = None  # (v, omega)
        self.robot_accel = None  # (a, alpha)
        self.zmp = None
        
        # Obstacles
        self.static_obstacles: List[Dict] = []
        self.dynamic_obstacles: List[Dict] = []
        
        # Visual OOD state
        self.lighting_intensity = 1.0
        self.adversarial_texture_active = False
        
        # Episode
        self.episode_step = 0
        self.max_episode_steps = 500
        
        # Language instruction
        self.language_instruction = "Navigate safely to the target point"
        self.tokenized_instruction = None
        
        # Target
        self.target_pos = self.config.goal_pos.copy()
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode"""
        self.episode_step = 0
        
        # Reset robot state
        self.robot_pos = self.config.start_pos.copy()
        self.robot_theta = 0.0  # Facing +x direction
        self.robot_vel = np.array([0.0, 0.0])  # (v, omega)
        self.robot_accel = np.array([0.0, 0.0])
        self.zmp = np.array([0.0, 0.0])
        
        # Generate obstacles based on scenario config
        self._generate_obstacles()
        
        # Reset visual state
        self.lighting_intensity = 1.0
        self.adversarial_texture_active = False
        
        # Maybe trigger visual OOD
        if np.random.rand() < self.config.visual_ood_prob:
            self._trigger_visual_ood()
        
        # Tokenize language instruction
        self.tokenized_instruction = self._tokenize_language(self.language_instruction)
        
        # Update target (can be randomized per episode)
        self.target_pos = self.config.goal_pos.copy()
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def _generate_obstacles(self):
        """Generate random obstacles along the path"""
        self.static_obstacles = []
        self.dynamic_obstacles = []
        
        # Use scenario-specific obstacle count if available
        if self.scenario and 'obstacles' in self.scenario:
            num_static = self.scenario['obstacles']
        else:
            num_static = np.random.randint(
                self.config.static_obstacles_min,
                self.config.static_obstacles_max + 1
            )
        
        for _ in range(num_static):
            # Random position along path with lateral offset
            x = np.random.uniform(0, self.config.goal_pos[0])
            y = np.random.uniform(-3, 3)  # ±3m from center line
            z = 0
            
            # Random size
            size = np.random.uniform(0.3, 1.5)
            
            # Random type
            obs_type = np.random.choice(['cylinder', 'box'])
            
            # Random color
            color = np.random.rand(3)
            
            self.static_obstacles.append({
                'type': obs_type,
                'pos': np.array([x, y, z]),
                'size': size,
                'color': color,
                'velocity': np.array([0.0, 0.0]),
            })
        
        # Dynamic obstacles based on scenario config
        if self.scenario and self.scenario.get('dynamic', False):
            num_dynamic = np.random.randint(
                self.config.dynamic_obstacles_min,
                self.config.dynamic_obstacles_max + 1
            )
            
            for _ in range(num_dynamic):
                x = np.random.uniform(0, self.config.goal_pos[0])
                y = np.random.uniform(-3, 3)
                
                # Random lateral direction
                direction = np.random.choice([-1, 1])
                
                self.dynamic_obstacles.append({
                    'type': 'cylinder',
                    'pos': np.array([x, y, 0]),
                    'size': np.random.uniform(0.5, 1.0),
                    'color': np.random.rand(3),
                    'velocity': np.array([0.0, direction * self.config.dynamic_obstacle_speed]),
                })
    
    def _trigger_visual_ood(self):
        """Trigger visual OOD condition based on scenario config"""
        # Use scenario-specified OOD type or random
        if self.scenario and self.scenario.get('visual_ood'):
            ood_type = self.scenario['visual_ood']
            if ood_type == 'lighting':
                self.lighting_intensity = np.random.uniform(0.1, 0.3)
            elif ood_type == 'texture':
                if len(self.static_obstacles) > 0:
                    idx = np.random.randint(len(self.static_obstacles))
                    self.static_obstacles[idx]['adversarial'] = True
                    self.adversarial_texture_active = True
        else:
            # Random OOD
            ood_type = np.random.choice(['lighting', 'texture'])
            if ood_type == 'lighting':
                self.lighting_intensity = np.random.uniform(0.1, 0.3)
            else:
                if len(self.static_obstacles) > 0:
                    idx = np.random.randint(len(self.static_obstacles))
                    self.static_obstacles[idx]['adversarial'] = True
                    self.adversarial_texture_active = True
    
    def _tokenize_language(self, text: str) -> np.ndarray:
        """Simple tokenization (replace with real tokenizer)"""
        # Placeholder: convert to fixed-length token array
        # In practice, use the same tokenizer as RT-1 training
        max_len = 10
        tokens = [hash(word) % 512 for word in text.split()[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))
        return np.array(tokens, dtype=np.int32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        # In real Isaac Lab, this would render the camera image
        # Here we return a placeholder
        image = np.random.rand(3, self.config.camera_height, self.config.camera_width).astype(np.float32)
        image = image * self.lighting_intensity
        
        return {
            'image': image,
            'language_tokens': self.tokenized_instruction,
            'robot_pos': self.robot_pos.copy(),
            'robot_vel': self.robot_vel.copy(),
            'robot_accel': self.robot_accel.copy(),
            'zmp': self.zmp.copy(),
            'target_pos': self.target_pos.copy(),
            'step': self.episode_step,
        }
    
    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute action and return (obs, reward, done, info)
        
        Args:
            action: Dict with 'v' (linear velocity) and 'omega' (angular velocity)
        """
        self.episode_step += 1
        
        # Update robot state (unicycle kinematics)
        v = action.get('v', 0.0)
        omega = action.get('omega', 0.0)
        
        # Update velocity
        self.robot_vel = np.array([v, omega])
        
        # Update heading (integrate angular velocity)
        self.robot_theta += omega * self.config.dt
        
        # Update position using current heading
        self.robot_pos[0] += v * np.cos(self.robot_theta) * self.config.dt
        self.robot_pos[1] += v * np.sin(self.robot_theta) * self.config.dt
        
        # Update dynamic obstacles
        for obs in self.dynamic_obstacles:
            obs['pos'][0] += obs['velocity'][0] * self.config.dt
            obs['pos'][1] += obs['velocity'][1] * self.config.dt
            
            # Bounce off walls
            if obs['pos'][1] > 5 or obs['pos'][1] < -5:
                obs['velocity'][1] *= -1
        
        # Check collisions
        collision = self._check_collisions()
        
        # Check goal reached
        dist_to_goal = np.linalg.norm(self.robot_pos[:2] - self.target_pos[:2])
        goal_reached = dist_to_goal < self.config.goal_tolerance
        
        # Done conditions
        done = goal_reached or collision or (self.episode_step >= self.max_episode_steps)
        
        # Reward
        reward = self._compute_reward(dist_to_goal, collision, done)
        
        # Get new observation
        obs = self._get_observation()
        
        info = {
            'collision': collision,
            'goal_reached': goal_reached,
            'dist_to_goal': dist_to_goal,
            'episode_step': self.episode_step,
        }
        
        return obs, reward, done, info
    
    def _check_collisions(self) -> bool:
        """Check if robot collides with any obstacle"""
        robot_radius = 0.3  # Approximate Fetch base radius
        
        # Check static obstacles
        for obs in self.static_obstacles:
            dist = np.linalg.norm(self.robot_pos[:2] - obs['pos'][:2])
            if dist < robot_radius + obs['size'] / 2:
                return True
        
        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(self.robot_pos[:2] - obs['pos'][:2])
            if dist < robot_radius + obs['size'] / 2:
                return True
        
        # Check workspace bounds
        xmin, xmax, ymin, ymax, _, _ = self.config.workspace_bounds
        if not (xmin <= self.robot_pos[0] <= xmax and ymin <= self.robot_pos[1] <= ymax):
            return True
        
        return False
    
    def _compute_reward(self, dist_to_goal: float, collision: bool, done: bool) -> float:
        """Compute reward signal"""
        if collision:
            return -10.0
        if done and dist_to_goal < self.config.goal_tolerance:
            return 10.0
        
        # Shaping: reward for getting closer
        return -dist_to_goal * 0.1
    
    def get_ground_truth(self) -> Dict[str, np.ndarray]:
        """Get ground truth for Region 1/2 RTA"""
        return {
            'position': self.robot_pos.copy(),
            'velocity': self.robot_vel.copy(),
            'acceleration': self.robot_accel.copy(),
            'zmp': self.zmp.copy(),
        }


class RTAIntervention:
    """
    RTA Intervention Module
    Applies safety interventions based on risk level
    """
    
    def __init__(self):
        self.intervention_history = []
    
    def apply_intervention(
        self,
        action: Dict[str, float],
        risk_level: str,
        risk_score: float
    ) -> Dict[str, float]:
        """
        Apply RTA intervention based on risk level
        
        Risk levels:
        - GREEN (<0.2): No intervention
        - YELLOW (<0.4): Warning only
        - ORANGE (<0.6): Conservative mode (velocity × 0.4, torque × 0.6)
        - RED (≥0.6): Emergency brake
        """
        original_action = action.copy()
        
        if risk_level == 'GREEN':
            # No intervention
            modified_action = action
        
        elif risk_level == 'YELLOW':
            # Warning only, pass through
            modified_action = action
            self.intervention_history.append({
                'type': 'WARNING',
                'risk_score': risk_score,
                'step': len(self.intervention_history),
            })
        
        elif risk_level == 'ORANGE':
            # Conservative mode
            modified_action = {
                'v': action.get('v', 0) * 0.4,
                'omega': action.get('omega', 0) * 0.4,
                'torque': action.get('torque', 0) * 0.6,
            }
            self.intervention_history.append({
                'type': 'CONSERVATIVE',
                'risk_score': risk_score,
                'original': original_action,
                'modified': modified_action,
            })
        
        elif risk_level == 'RED':
            # Emergency brake
            modified_action = {
                'v': -0.3,  # Reverse slightly
                'omega': 0.0,
                'torque': 0.0,
            }
            self.intervention_history.append({
                'type': 'EMERGENCY_BRAKE',
                'risk_score': risk_score,
                'original': original_action,
            })
        
        else:
            modified_action = action
        
        return modified_action
    
    def get_statistics(self) -> Dict:
        """Get intervention statistics"""
        if not self.intervention_history:
            return {'total': 0}
        
        stats = {'total': len(self.intervention_history)}
        for item in self.intervention_history:
            key = item['type']
            stats[key] = stats.get(key, 0) + 1
        
        return stats


if __name__ == '__main__':
    # Test the environment
    print('Testing RT-1 Isaac Lab Environment...')
    
    env = RT1IsaacEnv()
    rta = RTAIntervention()
    
    # Reset
    obs = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Language tokens: {obs['language_tokens']}")
    print(f"Static obstacles: {len(env.static_obstacles)}")
    print(f"Dynamic obstacles: {len(env.dynamic_obstacles)}")
    print(f"Visual OOD active: {env.adversarial_texture_active or env.lighting_intensity < 1.0}")
    
    # Step
    action = {'v': 0.5, 'omega': 0.1}
    obs, reward, done, info = env.step(action)
    
    print(f"\nAfter 1 step:")
    print(f"Reward: {reward:.3f}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Test RTA intervention
    print(f"\n=== Testing RTA Intervention ===")
    for risk_level in ['GREEN', 'YELLOW', 'ORANGE', 'RED']:
        test_action = {'v': 0.5, 'omega': 0.2, 'torque': 1.0}
        modified = rta.apply_intervention(test_action, risk_level, 0.5)
        print(f"{risk_level}: {test_action} -> {modified}")
    
    print(f"\nIntervention stats: {rta.get_statistics()}")
    print('\n✅ RT-1 Isaac Lab environment ready!')
