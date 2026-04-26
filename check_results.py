import json
with open('/root/rt1_experiment_outputs/experiment_results.json') as f:
    data = json.load(f)

print(f'总试验数：{len(data["config"])}')
print(f'成功字段样例：{data["success"][:10]}')
print(f'碰撞字段样例：{data["collision"][:10]}')
print(f'步数字段样例：{data["steps"][:10]}')
print(f'奖励字段样例：{data["reward"][:10]}')

success_true = sum(1 for s in data['success'] if s == True or s == 'True')
success_false = sum(1 for s in data['success'] if s == False or s == 'False')
print(f'\n成功=True: {success_true}, 成功=False: {success_false}')

# 检查是否达到最大步数
avg_steps = sum(data['steps']) / len(data['steps'])
print(f'平均步数：{avg_steps:.1f} (最大 500)')
