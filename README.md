# game-ai-scan


```
 pip install -r requirements.txt
```

```
 pyinstaller .\ai-scan.py --onefile --console
```

```
1. 先卸载错误的 CPU 版 PyTorch
打开你的 PowerShell，运行：
bash
运行
pip uninstall torch torchvision torchaudio -y
2. 安装适配你 4060Ti + CUDA13.1 的 GPU 版（直接复制）
bash
运行
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
为什么用 cu124？
你的驱动支持 CUDA13.1，向下兼容 CUDA12.4，这是目前 YOLO 最稳定、兼容性最好的版本。
```
