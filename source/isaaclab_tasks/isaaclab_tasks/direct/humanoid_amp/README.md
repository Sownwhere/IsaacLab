```
ln -s ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp ~/IsaacLab/

ln -s ~/IsaacLab/scripts/reinforcement_learning/skrl ~/IsaacLab/
```
Train:
```
(env_isaaclab) gdp@gdp:~/IsaacLab$
./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-Bw-AMP-Walk-Direct-v0 --headless

or

./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-G1-AMP-Dance-Direct-v0 --headless
```
Eval：
```
./isaaclab.sh -p ~/IsaacLab/skrl/play.py --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 32 
```
TensorBoard:
```
./isaaclab.sh -p -m tensorboard.main --logdir logs/skrl/
```
The parameters of the code in this repository have not been fine-tuned. Currently, the walk performance is acceptable, but the dance performance is quite poor. Due to personal bussiness, I will not begin to debug until summer.

The dataset and URDF files are from [Hugging Face](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset). 

Video: [Bilibili](https://www.bilibili.com/video/BV19cRvYhEL8/?vd_source=5159ce41348cd4fd3d83ef9169dc8dbc)

DeepWiki: [humanoid_amp](https://deepwiki.com/linden713/humanoid_amp)

**Contributions**, **discussions**, and stars are all welcome! ❥(^_-)
