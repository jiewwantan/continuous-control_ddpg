[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Continuous Control in Unity environment
Continuous control of double-jointed arms in a Reacher environment by DDPG agents

![Trained Agent][image1]

### Environment Details

This project works with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

The goal is to move the single or 20 double-jointed "arms" to reach and maintain a target location for as long as possible. 
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
'''
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		goal_speed -> 1.0
		goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
'''

## Installation Instruction
#### The README has instructions for installing dependencies or downloading needed files.

Python 3.6 is required. The program requires PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git  
cd deep-reinforcement-learning/python  
pip install .
```

Run the following to create drlnd kernel in ipython so that the right unity environment is loaded correctly  

```python -m ipykernel install --user --name drlnd --display-name "drlnd"```

Pytorch can be installed with the commands recommended in https://pytorch.org/ for the respective OS. Fo example for Conda package installling into a Windows environment with Python 3.6 is: 
'''
conda install pytorch -c pytorch
'''

## Getting Started

Place <mark>report.ipynb</mark> in the folder <mark>p1_navigation/</mark> together with the following two files:

1. ddpg_agent.py - contains the DDPG agent code. 
2. model.py - contains Actor and Critic neural network modules classes

The Unity Reacher environment can be downloaded from here: 

- **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Choose the environment suitable for your machine. Unzipping will create another Reacher_XXX folder. For example, if the Reacher Windows environment is downloaded, ```Reacher_Windows_x86_64``` will be created. 

Run ```p2_continuous-control/report.ipynb```

Enter the right path for the Unity Reacher environment in report.ipynb. For example for a folder consisting a 20-agent environemnt named as /Reacher_M: 

```
env = UnityEnvironment(file_name="../Reacher_M/Reacher.exe")

```

Run the remaining cell as ordered in ```report.ipynb``` to train the DDPG agent. 
