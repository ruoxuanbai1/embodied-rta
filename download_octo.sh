#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rt1-isaac

cd /root
rm -rf models/octo
mkdir -p models/octo

echo "=== 下载 Octo-base ==="
python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download, login
token = os.environ.get('HF_TOKEN', '')
if not token:
    print('Error: HF_TOKEN not set')
    exit(1)
login(token=token)
print('下载 octo-base...')
snapshot_download(repo_id='rail-berkeley/octo-base', local_dir='/root/models/octo', max_workers=4)
print('完成')
PYEOF

echo ""
echo "=== 结果 ==="
ls -lh models/octo/
du -sh models/octo/
