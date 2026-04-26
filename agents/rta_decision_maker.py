#!/usr/bin/env python3
"""
RTA 决策器 (RTA Decision Maker)

整合三层 RTA 检测 + 安全控制器
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# 简化：直接导入 (在服务器上运行时会正确配置路径)
try:
    from safe_fallback_controller import SafeFallbackController, InterventionInfo
except ImportError:
    # 如果在服务器上，使用完整路径
    import sys
    sys.path.insert(0, '/home/vipuser/Embodied-RTA/agents')
    from safe_fallback_controller import SafeFallbackController, InterventionInfo


@dataclass
class RTADecision:
    """RTA 决策结果"""
    action: Dict                    # 最终执行的动作
    rta_level: int                  # RTA 触发级别 (0/1/2/3)
    reason: str                     # 触发原因
    intervention_info: Optional[InterventionInfo]  # 干预信息
    details: Dict                   # 详细信息


class RTADecisionMaker:
    """RTA 决策器"""
    
    def __init__(self, region1_checker=None, region2_predictor=None, 
                 region3_detector=None, env_params: Dict = None):
        """
        初始化 RTA 决策器
        
        参数:
            region1_checker: Region 1 硬约束检查器
            region2_predictor: Region 2 GRU 可达性预测器
            region3_detector: Region 3 视觉 XAI 检测器
            env_params: 环境参数
        """
        self.r1 = region1_checker
        self.r2 = region2_predictor
        self.r3 = region3_detector
        
        # 安全控制器
        self.safe_controller = SafeFallbackController(env_params)
        
        # 统计信息
        self.stats = {
            'total_decisions': 0,
            'r1_triggers': 0,
            'r2_triggers': 0,
            'r3_triggers': 0,
            'no_triggers': 0,
        }
    
    def decide(self, state: Dict, vla_action: Dict) -> RTADecision:
        """
        综合决策
        
        参数:
            state: 当前状态
                - image: RGB 图像
                - history: 过去 10 帧状态
                - obstacles: 障碍物列表
                - base: 底盘状态
                - arm_q, arm_dq: 机械臂状态
            vla_action: VLA 原始动作
                - v: 线速度
                - ω: 角速度
                - τ: 关节扭矩
        
        返回:
            decision: RTA 决策结果
        """
        self.stats['total_decisions'] += 1
        
        # 1. Region 3: 感知检查 (最高优先级)
        if self.r3 is not None:
            r3_triggered, r3_score, r3_details = self.r3.detect(state.get('image'))
            if r3_triggered:
                self.stats['r3_triggers'] += 1
                
                # Region 3 干预：保守模式
                safe_action, info = self.safe_controller.decide_action(
                    vla_action, rta_level=3
                )
                
                return RTADecision(
                    action=safe_action,
                    rta_level=3,
                    reason=f'Visual anomaly detected (score={r3_score:.2f})',
                    intervention_info=info,
                    details={'region3': r3_details}
                )
        
        # 2. Region 2: 可达性检查
        if self.r2 is not None:
            # GRU 预测可达集
            reachable_set = self.r2.predict(state.get('history', []))
            
            # 检查碰撞风险
            r2_triggered, r2_info = self.r2.check_collision_risk(
                state, vla_action, reachable_set
            )
            
            if r2_triggered:
                self.stats['r2_triggers'] += 1
                
                # Region 2 干预：TTC 动态缩放
                risk_info = {
                    'reachable_set': reachable_set,
                    'obstacles': state.get('obstacles', []),
                }
                safe_action, info = self.safe_controller.decide_action(
                    vla_action, rta_level=2, risk_info=risk_info
                )
                
                return RTADecision(
                    action=safe_action,
                    rta_level=2,
                    reason=f'Future collision predicted (TTC={r2_info.get("ttc", 0):.2f}s)',
                    intervention_info=info,
                    details={'region2': r2_info, 'reachable_set': reachable_set}
                )
        
        # 3. Region 1: 硬约束检查
        if self.r1 is not None:
            r1_violations = self.r1.check_all_constraints(state, vla_action)
            if r1_violations:
                self.stats['r1_triggers'] += 1
                
                # Region 1 干预：紧急刹车
                safe_action, info = self.safe_controller.decide_action(
                    vla_action, rta_level=1
                )
                
                return RTADecision(
                    action=safe_action,
                    rta_level=1,
                    reason=f'Hard constraint violated: {r1_violations[0]}',
                    intervention_info=info,
                    details={'region1': r1_violations}
                )
        
        # 4. 无危险：执行原动作
        self.stats['no_triggers'] += 1
        
        return RTADecision(
            action=vla_action.copy(),
            rta_level=0,
            reason='No safety constraints violated',
            intervention_info=None,
            details={}
        )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats['total_decisions']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'r1_rate': self.stats['r1_triggers'] / total,
            'r2_rate': self.stats['r2_triggers'] / total,
            'r3_rate': self.stats['r3_triggers'] / total,
            'safe_rate': self.stats['no_triggers'] / total,
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_decisions': 0,
            'r1_triggers': 0,
            'r2_triggers': 0,
            'r3_triggers': 0,
            'no_triggers': 0,
        }


# 测试
if __name__ == '__main__':
    # 创建测试状态和动作
    state = {
        'image': np.random.randn(3, 224, 224),
        'history': [np.random.randn(16) for _ in range(10)],
        'obstacles': [
            {'type': 'cylinder', 'position': [3.0, 0.5, 0], 'velocity': [-1.0, 0, 0], 'size': [0.4, 1.7]}
        ],
        'base': np.array([0.0, 0.0, 0.0, 0.5, 0.3]),
        'arm_q': np.zeros(7),
        'arm_dq': np.zeros(7),
    }
    
    vla_action = {
        'v': 0.8,
        'ω': 0.5,
        'τ': np.ones(7) * 10
    }
    
    # 创建决策器 (无实际检测器，测试默认行为)
    decision_maker = RTADecisionMaker()
    
    print("=== RTA Decision Maker 测试 ===\n")
    
    # 测试 1: 无检测器，直接通过
    decision = decision_maker.decide(state, vla_action)
    print(f"测试 1 (无检测器):")
    print(f"  rta_level={decision.rta_level}")
    print(f"  reason={decision.reason}")
    print(f"  action_v={decision.action['v']:.2f}")
    
    # 测试 2: 统计信息
    stats = decision_maker.get_stats()
    print(f"\n统计信息:")
    print(f"  total={stats['total_decisions']}, no_trigger={stats['no_triggers']}")
    
    print("\n✅ 测试完成!")
