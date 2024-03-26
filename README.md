#

### 依存環境
- Python 3.10.13
- CUDA 11.7
- torch==2.0.1
- torchvision==0.15.2
- pytorch-lightning==1.9.0
- torchmetrics==0.11.1

### Poetry

```
poetry source add torch_cu117 --priority=explicit https://download.pytorch.org/whl/cu117
poetry add torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --source torch_cu117
```