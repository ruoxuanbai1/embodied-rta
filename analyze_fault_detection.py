#!/usr/bin/env python3
"""
analyze_fault_detection.py - 分析故障检测性能指标

计算:
- 准确率 (Accuracy)
- 精准率 (Precision)
- 召回率 (Recall)
- 虚警率 (False Alarm Rate)
- F1 分数
- 检测延迟
"""

import json
import glob
import os

def load_trial_summaries(output_dir):
    """加载所有试验摘要"""
    summaries = []
    for f in sorted(glob.glob(os.path.join(output_dir, 'trial_*_summary.json'))):
        with open(f, 'r') as file:
            summaries.append(json.load(file))
    return summaries

def compute_metrics(summaries):
    """计算故障检测指标"""
    
    # 按故障类型分组
    by_fault_type = {}
    for s in summaries:
        fault_type = s['fault_type']
        if fault_type not in by_fault_type:
            by_fault_type[fault_type] = []
        by_fault_type[fault_type].append(s)
    
    print("="*70)
    print("三层 RTA 故障检测性能分析")
    print("="*70)
    
    results = {}
    
    for fault_type, trials in by_fault_type.items():
        print(f"\n{fault_type}:")
        print("-"*70)
        
        # 统计
        total_trials = len(trials)
        detected_trials = sum(1 for t in trials if t['fault_detected'])
        
        # 对于无故障基线
        if fault_type == 'none':
            # 虚警率：无故障时触发预警的比例
            avg_alert_rate = sum(t['alert_rate'] for t in trials) / len(trials)
            print(f"  试验数：{total_trials}")
            print(f"  平均预警率：{avg_alert_rate*100:.1f}%")
            print(f"  虚警率 (False Alarm Rate): {avg_alert_rate*100:.1f}%")
            print(f"  特异度 (Specificity): {(1-avg_alert_rate)*100:.1f}%")
            
            results[fault_type] = {
                'total_trials': total_trials,
                'false_alarm_rate': avg_alert_rate,
                'specificity': 1 - avg_alert_rate,
            }
        else:
            # 对于故障注入试验
            avg_alert_rate = sum(t['alert_rate'] for t in trials) / len(trials)
            avg_detection_delay = sum(t['detection_delay'] for t in trials if t['detection_delay'] is not None)
            has_detection_delay = sum(1 for t in trials if t['detection_delay'] is not None)
            
            # 真正例 (TP): 故障注入且检测到
            tp = detected_trials
            # 假负例 (FN): 故障注入但未检测到
            fn = total_trials - detected_trials
            
            # 召回率 (Recall) = TP / (TP + FN)
            recall = tp / total_trials if total_trials > 0 else 0
            
            # 精准率需要知道虚警情况，这里用预警率近似
            # 实际应该用：TP / (TP + FP)
            # 但我们的数据中 FP 是基线试验的预警率
            
            print(f"  试验数：{total_trials}")
            print(f"  检测到故障：{detected_trials}/{total_trials}")
            print(f"  召回率 (Recall): {recall*100:.1f}%")
            print(f"  平均预警率：{avg_alert_rate*100:.1f}%")
            if has_detection_delay > 0:
                print(f"  平均检测延迟：{avg_detection_delay:.1f} 步 ({avg_detection_delay*0.02:.2f} 秒)")
            else:
                print(f"  平均检测延迟：N/A (未记录)")
            
            results[fault_type] = {
                'total_trials': total_trials,
                'detected': detected_trials,
                'recall': recall,
                'avg_alert_rate': avg_alert_rate,
                'avg_detection_delay': avg_detection_delay if has_detection_delay > 0 else None,
            }
    
    # 综合指标
    print("\n" + "="*70)
    print("综合性能指标")
    print("="*70)
    
    # 获取基线虚警率
    if 'none' in results:
        baseline_far = results['none']['false_alarm_rate']
        print(f"\n基线虚警率：{baseline_far*100:.1f}%")
        
        # 计算各故障类型的检测性能
        print(f"\n{'故障类型':<25} {'召回率':<10} {'预警率':<10} {'检测延迟 (步)':<15}")
        print("-"*70)
        
        for fault_type, metrics in results.items():
            if fault_type == 'none':
                continue
            
            recall = metrics['recall']
            alert_rate = metrics['avg_alert_rate']
            delay = metrics['avg_detection_delay']
            delay_str = f"{delay:.1f}" if delay is not None else "N/A"
            
            print(f"{fault_type:<25} {recall*100:>6.1f}%    {alert_rate*100:>6.1f}%    {delay_str:>15}")
    
    # 保存结果
    output_path = os.path.join(output_dir, 'detection_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存：{output_path}")
    
    return results


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './outputs/rta_fault_tests'
    
    summaries = load_trial_summaries(output_dir)
    compute_metrics(summaries)
