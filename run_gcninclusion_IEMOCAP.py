# run_random_configs_loop.py
import random
import subprocess
import time

# 설정 가능한 값들
scripts = ['train.py']
# , 'train_MKD.py']
datasets = ['IEMOCAP']


# 실행 횟수 설정
num_runs = 1000000000  # 원하는 횟수로 바꾸세요

for i in range(num_runs):
    seed = random.randint(1, 10000)
    
    
    
    for _ in range(32):
        script = random.choice(scripts)
        dataset = random.choice(datasets)
        use_lstm = random.choice([True, False])
        self_attention = random.choice([True,False])
            
        original_gcn = random.choice([True])
        graph_masking = random.choice([True, False])

        cmd = ['python', script, '--Dataset', dataset, '--seed_number', str(seed)]
        if use_lstm:
            cmd.append('--av_using_lstm')
        if self_attention:
            cmd.append("--self_attention")
        if original_gcn:
            cmd.append("--original_gcn")
        if graph_masking:
            cmd.append("--graph_masking")
        

        print(f"\n[{i+1}/{num_runs}] Running: {' '.join(cmd)}")
        
        # 실행
        subprocess.run(cmd)

        # 원하면 대기 시간 추가
        # time.sleep(1)  # 1초 대기
