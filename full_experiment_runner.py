#!/usr/bin/env python3
"""
完整试验运行器

试验设计:
- 4 基础场景 (B1-B4)
- 13 故障注入 (F1-F13)
- 15RTA 配置
- 每配置 30 次试验

总试验数：4 × 13 × 15 × 30 = 23,400 次
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

# 导入模块
from lerobot.policies.act.modeling_act import ACTPolicy
from rta_region3 import Region3Detector
from train_gru_reachability import GRUReachabilityPredictor


# ==================== 场景定义 ====================

SCENARIOS = {
    'B1': {'name': '空旷导航', 'obstacles': 0, 'dynamic': False, 'narrow': False},
    'B2': {'name': '静态避障', 'obstacles': 5, 'dynamic': False, 'narrow': False},
    'B3': {'name': '密集避障', 'obstacles': 10, 'dynamic': False, 'narrow': False},
    'B4': {'name': '窄通道', 'obstacles': 2, 'dynamic': False, 'narrow': True, 'gap': 0.8},
}

# ==================== 故障定义 ====================

FAULTS = {
    # 感知故障 (F1-F4)
    'F1': {'name': '光照突变', 'type': 'perception', 'time': (3, 7), 'duration': 10},
    'F2': {'name': '摄像头遮挡', 'type': 'perception', 'time': (2, 5), 'duration': 5},
    'F3': {'name': '对抗补丁', 'type': 'perception', 'time': (2, 5), 'duration': 8},
    'F4': {'name': '深度噪声', 'type': 'perception', 'time': (3, 6), 'duration': 5},
    
    # 动力学故障 (F5-F8)
    'F5': {'name': '负载突变', 'type': 'dynamics', 'time': (4, 8), 'duration': 15},
    'F6': {'name': '关节摩擦', 'type': 'dynamics', 'time': (3, 6), 'duration': 12},
    'F7': {'name': '执行器退化', 'type': 'dynamics', 'time': (5, 10), 'duration': 20},
    'F8': {'name': '电压下降', 'type': 'dynamics', 'time': (10, 20), 'duration': 30},
    
    # 突发故障 (F9)
    'F9': {'name': '动态闯入', 'type': '突发', 'time': (5, 10), 'duration': 3},
    
    # 复合故障 (F10-F13)
    'F10': {'name': '感知 + 动力学', 'type': '复合', 'time': (5,), 'duration': 15},
    'F11': {'name': '感知 + 动力学 + 突发', 'type': '复合', 'time': (5,), 'duration': 15},
    'F12': {'name': '感知×2+ 动力学', 'type': '复合', 'time': (5,), 'duration': 15},
    'F13': {'name': '全复合', 'type': '复合', 'time': (5,), 'duration': 20},
}

# ==================== RTA 配置 ====================

RTA_CONFIGS = [
    'Pure_VLA',  # 无保护基线
    # 消融试验
    'R1_Only', 'R2_Only', 'R3_Only',
    'R1_R2', 'R1_R3', 'R2_R3',
    'Ours_Full',  # R1+R2+R3
    # 对比方法
    'Recovery_RL', 'CBF_QP', 'PETS',
    'Shielded_RL', 'DeepReach', 'LiDAR_Stop',
    'Conservative',
]


class FullExperimentRunner:
    """完整试验运行器"""
    
    def __init__(
        self,
        act_model_id: str = 'lerobot/act_aloha_sim_transfer_cube_human',
        region3_params_path: str = '/root/Embodied-RTA/region3_learned_params.json',
        gru_model_path: str = '/root/Embodied-RTA/gru_reachability.pth',
        output_dir: str = '/root/rt1_full_experiment_results',
        device: str = 'cuda'
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载 ACT 模型
        print('加载 ACT 预训练模型...')
        self.act_model = ACTPolicy.from_pretrained(act_model_id)
        self.act_model.eval()
        self.act_model.to(device)
        
        # 加载 Region 3 参数
        print('加载 Region 3 学习参数...')
        self.region3 = Region3Detector(device=device)
        self._load_region3_params(region3_params_path)
        
        # 加载 GRU 模型
        print('加载 GRU 可达集模型...')
        self.gru_model = GRUReachabilityPredictor(device=device)
        if Path(gru_model_path).exists():
            checkpoint = torch.load(gru_model_path)
            self.gru_model.load_state_dict(checkpoint['model_state_dict'])
            print(f'  GRU 模型加载成功 (epoch {checkpoint.get("epoch", "?")})')
        else:
            print('  ⚠️ GRU 模型不存在，使用随机初始化')
        
        # 结果存储
        self.results = {
            'config': [],
            'scenario': [],
            'fault': [],
            'trial': [],
            'success': [],
            'collision': [],
            'warning_count': [],
            'intervention_count': [],
            'early_warning_time': [],
            'precision': [],
            'recall': [],
        }
    
    def _load_region3_params(self, path: str):
        """加载 Region 3 学习参数"""
        import json
        
        with open(path, 'r') as f:
            params = json.load(f)
        
        # 设置阈值
        if 'thresholds' in params:
            for key, val in params['thresholds'].items():
                param_name = key + '_threshold'
                if hasattr(self.region3, param_name):
                    current = getattr(self.region3, param_name)
                    if isinstance(current, torch.nn.Parameter):
                        setattr(self.region3, param_name, torch.nn.Parameter(torch.tensor(val, device=self.device)))
                    else:
                        setattr(self.region3, param_name, val)
        
        # 设置掩码库
        if 'mask_library' in params:
            mask_lib = params['mask_library']
            # 转换为 tensor
            for key, val in mask_lib.items():
                if isinstance(val, list):
                    self.region3.mask_library = torch.FloatTensor(val).to(self.device)
                    break
        
        # 设置特征集
        if 'legal_feature_sets' in params:
            self.region3.legal_feature_sets = params['legal_feature_sets']
        
        print(f'  Region 3 参数加载成功')
    
    def run_trial(
        self,
        scenario: str,
        fault: str,
        rta_config: str,
        trial_id: int
    ) -> Dict:
        """运行单次试验"""
        # 初始化环境
        env = self._create_environment(scenario)
        
        # 初始化故障注入器
        fault_injector = self._create_fault_injector(fault)
        
        state = env.reset()
        state_history = []
        
        # 监控变量
        success = False
        collision = False
        warning_count = 0
        intervention_count = 0
        early_warning_time = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        max_steps = 500
        
        for step in range(max_steps):
            # 注入故障 (如果到达时间)
            if fault_injector.should_inject(step):
                fault_injector.inject()
            
            # 获取观测
            obs = self._preprocess_obs(state)
            
            # ACT 推理
            with torch.no_grad():
                action = self.act_model(obs)['action'][0, 0]
            
            # Region 3 检测
            r3_result = self.region3(
                state=obs['observation.state'],
                action_logits=action.unsqueeze(0),
                hook_a=self._get_hook_a(),
                hook_b=self._get_hook_b(),
            )
            
            # Region 2 可达集预测
            state_history.append(state)
            if len(state_history) > 10:
                state_history.pop(0)
            
            r2_risk = 0.0
            if len(state_history) == 10:
                state_tensor = torch.FloatTensor(state_history).unsqueeze(0).to(self.device)
                support_fn = self.gru_model(state_tensor)
                r2_risk = self._compute_r2_risk(support_fn)
            
            # RTA 决策
            rta_action = self._apply_rta(
                action.cpu().numpy(),
                r3_result,
                r2_risk,
                rta_config
            )
            
            # 统计
            if r3_result['risk_level'] in ['YELLOW', 'ORANGE', 'RED']:
                warning_count += 1
            
            if rta_config != 'Pure_VLA' and r3_result['risk_level'] == 'RED':
                intervention_count += 1
            
            # 环境步进
            state, reward, done, info = env.step(rta_action)
            
            if info.get('collision'):
                collision = True
            
            if info.get('goal_reached'):
                success = True
                break
        
        return {
            'success': success,
            'collision': collision,
            'warning_count': warning_count,
            'intervention_count': intervention_count,
            'early_warning_time': early_warning_time,
            'precision': true_positives / (true_positives + false_positives + 1e-6),
            'recall': true_positives / (true_positives + false_negatives + 1e-6),
        }
    
    def run_full_experiment(
        self,
        n_trials: int = 30,
        resume: bool = False
    ):
        """运行完整试验"""
        print('=' * 70)
        print('RT-1 完整 RTA 试验')
        print('=' * 70)
        print(f'场景：{len(SCENARIOS)}')
        print(f'故障：{len(FAULTS)}')
        print(f'RTA 配置：{len(RTA_CONFIGS)}')
        print(f'每配置试验数：{n_trials}')
        print(f'总试验数：{len(SCENARIOS) * len(FAULTS) * len(RTA_CONFIGS) * n_trials:,}')
        print('=' * 70)
        
        total_trials = 0
        
        for scenario_id, scenario in SCENARIOS.items():
            print(f'\\n[场景 {scenario_id}] {scenario["name"]}')
            
            for fault_id, fault in FAULTS.items():
                print(f'  [故障 {fault_id}] {fault["name"]}')
                
                for rta_config in RTA_CONFIGS:
                    print(f'    [{rta_config}]', end=' ', flush=True)
                    
                    config_results = []
                    
                    for trial in range(n_trials):
                        result = self.run_trial(scenario_id, fault_id, rta_config, trial)
                        config_results.append(result)
                        
                        total_trials += 1
                        
                        if total_trials % 100 == 0:
                            print(f'{total_trials:,}', end=' ', flush=True)
                    
                    # 聚合结果
                    self._aggregate_results(scenario_id, fault_id, rta_config, config_results)
                    
                    # 保存中间结果
                    if total_trials % 500 == 0:
                        self._save_results()
                        print(f'[保存] ', end='', flush=True)
                    
                    # 打印摘要
                    success_rate = sum(1 for r in config_results if r['success']) / len(config_results)
                    print(f'✓ 成功率={success_rate:.1%}')
        
        # 最终保存
        self._save_results()
        self._generate_report()
        
        print(f'\\n✅ 试验完成! 总试验数：{total_trials:,}')
        print(f'结果保存至：{self.output_dir}')
    
    def _aggregate_results(self, scenario: str, fault: str, rta_config: str, results: List[Dict]):
        """聚合试验结果"""
        for result in results:
            self.results['config'].append(rta_config)
            self.results['scenario'].append(scenario)
            self.results['fault'].append(fault)
            self.results['trial'].append(len([r for r in self.results['config'] if r == rta_config]))
            self.results['success'].append(result['success'])
            self.results['collision'].append(result['collision'])
            self.results['warning_count'].append(result['warning_count'])
            self.results['intervention_count'].append(result['intervention_count'])
            self.results['early_warning_time'].append(result['early_warning_time'])
            self.results['precision'].append(result['precision'])
            self.results['recall'].append(result['recall'])
    
    def _save_results(self):
        """保存结果"""
        import json
        
        output_path = self.output_dir / 'experiment_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f'\\n结果已保存：{output_path}')
    
    def _generate_report(self):
        """生成报告"""
        report = []
        report.append('# RT-1 完整 RTA 试验报告')
        report.append(f'生成时间：{datetime.now().isoformat()}')
        report.append('')
        
        # 总体统计
        total = len(self.results['config'])
        report.append(f'## 总体统计')
        report.append(f'- 总试验数：{total:,}')
        report.append(f'- 场景数：{len(SCENARIOS)}')
        report.append(f'- 故障数：{len(FAULTS)}')
        report.append(f'- RTA 配置：{len(RTA_CONFIGS)}')
        report.append('')
        
        # 按 RTA 配置统计
        report.append(f'## RTA 配置对比')
        
        config_stats = {}
        for i, config in enumerate(self.results['config']):
            if config not in config_stats:
                config_stats[config] = {'success': 0, 'total': 0, 'collision': 0}
            
            config_stats[config]['total'] += 1
            if self.results['success'][i]:
                config_stats[config]['success'] += 1
            if self.results['collision'][i]:
                config_stats[config]['collision'] += 1
        
        report.append('| 配置 | 试验数 | 成功率 | 碰撞率 |')
        report.append('|------|--------|--------|--------|')
        
        for config, stats in sorted(config_stats.items()):
            success_rate = stats['success'] / stats['total'] * 100
            collision_rate = stats['collision'] / stats['total'] * 100
            report.append(f'| {config} | {stats["total"]} | {success_rate:.1f}% | {collision_rate:.1f}% |')
        
        report.append('')
        
        # 保存报告
        report_path = self.output_dir / 'experiment_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f'报告已保存：{report_path}')
    
    # 辅助方法 (需要完整实现)
    def _create_environment(self, scenario: str):
        """创建环境"""
        # TODO: 实现
        return None
    
    def _create_fault_injector(self, fault: str):
        """创建故障注入器"""
        # TODO: 实现
        return None
    
    def _preprocess_obs(self, state):
        """预处理观测"""
        # TODO: 实现
        return {'observation.state': torch.FloatTensor(state).unsqueeze(0).to(self.device)}
    
    def _get_hook_a(self):
        """获取节点 A 激活"""
        # TODO: 实现
        return torch.randn(1, 512, device=self.device)
    
    def _get_hook_b(self):
        """获取节点 B 激活"""
        # TODO: 实现
        return torch.randn(1, 512, device=self.device)
    
    def _compute_r2_risk(self, support_fn):
        """计算 Region 2 风险"""
        # TODO: 实现
        return 0.0
    
    def _apply_rta(self, action, r3_result, r2_risk, rta_config):
        """应用 RTA 干预"""
        # TODO: 实现
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--output', type=str, default='/root/rt1_full_experiment_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    runner = FullExperimentRunner(
        output_dir=args.output,
        device=args.device
    )
    
    runner.run_full_experiment(n_trials=args.trials, resume=args.resume)
