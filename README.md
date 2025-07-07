# Cont_RL
## 1.安装isaacsim，isaaclab
按照https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html 教程安装isaacsim和isaclab
## 2.下载Cont_RL库并安装
```bash
 git clone https://github.com/sjtu-cjj/Cont_RL.git
 cd ./Cont_	RL
 python -m pip install -e ./Cont_RL/source/Cont_RL  # 安装Cont_RL库
 python -m pip install -e ./rsl_rl/                 # 安装rsl_rl库
```
## 3.训练Go2平地行走
```bash
 python ./Cont_RL/scripts/rsl_rl/train.py --task=Cont-RL-Velocity-Flat-Unitree-Go2-v0 --headless --video
```
训练模型保存在./Cont_RL/logs目录下
## 4.查看训练结果
```bash
 python ./Cont_RL/scripts/rsl_rl/play.py --task=Cont-RL-Velocity-Flat-Unitree-Go2-v0
```