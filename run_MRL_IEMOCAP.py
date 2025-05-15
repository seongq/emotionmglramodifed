# run_random_configs_loop.py
import random
import subprocess
import time

# 설정 가능한 값들
scripts = ['train_MRL.py']
datasets = ['IEMOCAP']

# 실행 횟수 설정
num_runs = 1000000000  # 원하는 횟수로 바꾸세요

for i in range(num_runs):
    script = random.choice(scripts)
    dataset = random.choice(datasets)
    efficient = random.choice([True, False])
    randomness = random.choice([True, False])
    seed = random.randint(1, 10000)
    cmd = ['python', script, '--Dataset', dataset, '--seed_number', str(seed)]
    if efficient:
        cmd.append('--MRL_efficient')
    if randomness:
        cmd.append("--MRL_random")
    print(f"\n[{i+1}/{num_runs}] Running: {' '.join(cmd)}")
    subprocess.run(cmd)