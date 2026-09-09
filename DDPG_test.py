import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from Test_env import DDPGEnvironment
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Initialize environment
num_ground_stations = 6
num_uav = 1
state_dim = num_ground_stations*3+3
action_dim = num_ground_stations+3
width = 200
height = 200
env = DDPGEnvironment(num_ground_stations,width,width, height=height)
font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')


# 定义一个函数来绘制轨迹，并减少标记的频率
def plot_trajectory(ax, x_data, y_data, z_data, label, color, marker, step=5, markersize=40):
    ax.plot([x * width for x in x_data],
            [y * width for y in y_data],
            [z * height for z in z_data],
            label=label, color=color)
    ax.scatter([x * width for x in x_data[::step]],
               [y * width for y in y_data[::step]],
               [z * height for z in z_data[::step]],
               color=color, marker=marker, s=markersize)



# Actor Network
class Actor(nn.Module):
    def __init__(self, hidden_dim=1024,hidden_dim1 = 512):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, action_dim)

    def forward(self, x):
        a = torch.relu(self.fc1(x))
        b = torch.relu(self.fc2(a))
        c = torch.tanh(self.fc3(b))  # Pendulum action space is [-1, 1]
        return c

current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + 'ddpg_actor_20240621181946.pth'

actor = Actor().to(device)
actor.load_state_dict(torch.load(actor_path))

###############优化方法#################

num_episodes = 60
num_step = 80
action_bound = [-1,1]
REWARD_BUFFER = np.empty(shape=num_episodes)
Data_rate_BUFFER = np.empty(shape=num_episodes)
Throughput_BUFFER = np.empty(shape=num_episodes)
Energy_efficiency_BUFFER = np.empty(shape=num_episodes)

UAV_trajectory_x=np.zeros((1, num_step))
UAV_trajectory_y=np.zeros((1, num_step))
UAV_trajectory_z=np.zeros((1, num_step))
GT_trajectory_x=np.zeros((1, num_step))
GT_trajectory_y=np.zeros((1, num_step))
GT_trajectory_z=np.zeros((1, num_step))
Max_reward=0
for episode_i in range(num_episodes):
    env.__init__(num_ground_stations,width,width, height=height)
    state = env.reset(0)
    episode_reward = 0
    episode_throughput=0
    episode_data_rate=0
    episode_Energy_efficiency=0

    UAV_positions_x=[]
    UAV_positions_y=[]
    UAV_positions_z=[]
    UAV_positions_x.append(state[0])
    UAV_positions_y.append(state[1])
    UAV_positions_z.append(state[2])
    GT_positions_x=[]
    GT_positions_y=[]
    GT_positions_z=[]
    GT_positions_x.append(state[3:3+num_ground_stations])
    GT_positions_y.append(state[3+num_ground_stations:3+2*num_ground_stations])
    GT_positions_z.append(state[3+2*num_ground_stations:3+3*num_ground_stations])

    for step_i in range(num_step):            
            action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            next_state, reward, done,throughput_step,data_rate_step,Energy_efficiency_step,_ = env.step(action,0)
            GT_positions_x.append(state[3:3+num_ground_stations])
            GT_positions_y.append(state[3+num_ground_stations:3+2*num_ground_stations])
            GT_positions_z.append(state[3+2*num_ground_stations:3+3*num_ground_stations])
            UAV_positions_x.append(state[0])
            UAV_positions_y.append(state[1])
            UAV_positions_z.append(state[2])
            state = next_state

            episode_reward += reward
            episode_throughput += throughput_step
            episode_data_rate += data_rate_step
            episode_Energy_efficiency += Energy_efficiency_step        

            if done:
                break
    
    REWARD_BUFFER[episode_i] = episode_reward
    Data_rate_BUFFER[episode_i] = episode_data_rate/num_step
    Throughput_BUFFER[episode_i] = episode_throughput/num_step
    Energy_efficiency_BUFFER[episode_i] = episode_Energy_efficiency/num_step

    if Max_reward<episode_reward :
        Max_reward=episode_reward

        UAV_trajectory_x=UAV_positions_x
        UAV_trajectory_y=UAV_positions_y
        UAV_trajectory_z=UAV_positions_z

        GT_trajectory_x=GT_positions_x
        GT_trajectory_y=GT_positions_y
        GT_trajectory_z=GT_positions_z
    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")


############基线方法#################
state = env.reset(1)
episode_reward = 0



episode_throughput=0
episode_data_rate=0
episode_Energy_efficiency=0
reward=0
UAV_positions_x=[]
UAV_positions_y=[]
UAV_positions_z=[]
UAV_positions_x.append(state[0])
UAV_positions_y.append(state[1])
UAV_positions_z.append(state[2])
GT_positions_x=[]
GT_positions_y=[]
GT_positions_z=[]
GT_positions_x.append(state[3:3+num_ground_stations])
GT_positions_y.append(state[3+num_ground_stations:3+2*num_ground_stations])
GT_positions_z.append(state[3+2*num_ground_stations:3+3*num_ground_stations])
for step_i in range(num_step):
            action=env.baseline(state)
            print(step_i,action)
            # print(action)
            next_state, reward, done,throughput_bs,data_rate_bs,Energy_efficiency_bs,_ = env.step(action,1)
            GT_positions_x.append(state[3:3+num_ground_stations])
            GT_positions_y.append(state[3+num_ground_stations:3+2*num_ground_stations])
            GT_positions_z.append(state[3+2*num_ground_stations:3+3*num_ground_stations])
            UAV_positions_x.append(state[0])
            UAV_positions_y.append(state[1])
            UAV_positions_z.append(state[2])
            episode_reward += reward
            episode_throughput += throughput_bs
            episode_data_rate += data_rate_bs
            episode_Energy_efficiency += Energy_efficiency_bs   
            state = next_state
print(f"Reward: {round(episode_reward, 2)}")


UAV_trajectory_x_baseline=UAV_positions_x
UAV_trajectory_y_baseline=UAV_positions_y
UAV_trajectory_z_baseline=UAV_positions_z

GT_trajectory_x_baseline=GT_positions_x
GT_trajectory_y_baseline=GT_positions_y
GT_trajectory_z_baseline=GT_positions_z
REWARD_BUFFER_baseline= np.zeros(shape=num_episodes)
Data_rate_BUFFER_baseline= np.zeros(shape=num_episodes)
Throughput_BUFFER_baseline= np.zeros(shape=num_episodes)
Energy_efficiency_BUFFER_baseline= np.zeros(shape=num_episodes)
for i in range(num_episodes):
    REWARD_BUFFER_baseline[i] = episode_reward
    Data_rate_BUFFER_baseline[i] = episode_data_rate/num_step
    Throughput_BUFFER_baseline[i] = episode_throughput/num_step
    Energy_efficiency_BUFFER_baseline[i] = episode_Energy_efficiency/num_step


####################################### 图像 #############################
# 设置相对路径
relative_path = 'plots'

# 如果目录不存在，则创建目录
if not os.path.exists(relative_path):
    os.makedirs(relative_path)

plt.clf()  # 清除当前图像
# 绘制 REWARD_BUFFER 图
plt.plot(REWARD_BUFFER, marker='o', label='DDPG')
plt.plot(REWARD_BUFFER_baseline, color='r', linestyle='--', label='Baseline')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Reward')
# 指定图例位置
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Reward.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')  # 保存图像，设置分辨率为300 DPI
#plt.show()
plt.close()  # 关闭图像窗口


plt.clf()  # 清除当前图像
# 绘制 Data_rate_BUFFER 图
plt.plot(Data_rate_BUFFER, marker='o', label='DDPG')
plt.plot(Data_rate_BUFFER_baseline, color='r', linestyle='--',label='Baseline')
plt.xlabel('Episode')
plt.ylabel('Data rate (Mbps)')
plt.title('DDPG Data Rate')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Data_rate.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight') 
#plt.show()
plt.close()  # 关闭图像窗口


plt.clf()  # 清除当前图像
# 绘制 Throughput_BUFFER 图
plt.plot(Throughput_BUFFER, marker='o', label='DDPG')
plt.plot(Throughput_BUFFER_baseline, color='r', linestyle='--', label='Baseline')
plt.xlabel('Episode')
plt.ylabel('Throughput (Mb)')
plt.title('DDPG Throughput')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Throughput.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight') 
#plt.show()
plt.close()  # 关闭图像窗口


plt.clf()  # 清除当前图像
# 绘制 Energy_efficiency_BUFFER 图
plt.plot(Energy_efficiency_BUFFER, marker='o', label='DDPG')
plt.plot(Energy_efficiency_BUFFER_baseline, color='r', linestyle='--', label='Baseline')
plt.xlabel('Episode')
plt.ylabel('Energy Efficiency (Kbps/J)')
plt.title('DDPG Energy Efficiency')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Energy_efficiency.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight') 
#plt.show()
plt.close()  # 关闭图像窗口



fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 添加一个 3D 坐标轴

# 绘制无人机飞行路径
ax.plot([x * width for x in UAV_trajectory_x],
        [y * width for y in UAV_trajectory_y],
        [z * height for z in UAV_trajectory_z],
        label='UAV Flight Path', marker='.', color='blue')

# 绘制基准无人机飞行路径
ax.plot([x * width for x in UAV_trajectory_x_baseline],
        [y * width for y in UAV_trajectory_y_baseline],
        [z * height for z in UAV_trajectory_z_baseline],
        label='UAV Baseline Path', marker='.', color='cyan')

n = num_ground_stations
# 初始化存储位置列表的列表
GT_location_x = [[] for _ in range(n)]
GT_location_y = [[] for _ in range(n)]
GT_location_z = [[] for _ in range(n)]

# 填充存储位置的列表
for slot in GT_trajectory_x:
    for i in range(n):
        GT_location_x[i].append(slot[i])
GT_location_x = [np.array(loc) for loc in GT_location_x]

for slot in GT_trajectory_y:
    for i in range(n):
        GT_location_y[i].append(slot[i])
GT_location_y = [np.array(loc) for loc in GT_location_y]

for slot in GT_trajectory_z:
    for i in range(n):
        GT_location_z[i].append(slot[i])
GT_location_z = [np.array(loc) for loc in GT_location_z]

# 定义一个颜色列表
colors = ['green', 'red', 'purple', 'orange', 'brown','pink', 'cyan', 'gray', 'olive']

# 绘图
for i in range(n):
    color = colors[i % len(colors)]  # 循环使用颜色列表
    plot_trajectory(ax, GT_location_x[i], GT_location_y[i], GT_location_z[i], f'GT{i+1} Path', color, '^', step=5)


# 设置坐标轴标签和图例
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()
ax.set_title('Combined 3D Trajectories of UAV and GT with Baseline')
save_path = os.path.join(relative_path, 'Combined 3D Trajectories of UAV and GT with Baseline.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight') 
#plt.show()

# 生成 2D 图像
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111)  # 添加一个 2D 坐标轴

# 绘制无人机飞行路径在XY平面
ax2.plot([x * width for x in UAV_trajectory_x],
        [y * width for y in UAV_trajectory_y],
        label='UAV Flight Path', marker='.', color='blue')

# 绘制基准无人机飞行路径在XY平面
ax2.plot([x * width for x in UAV_trajectory_x_baseline],
        [y * width for y in UAV_trajectory_y_baseline],
        label='UAV Baseline Path', marker='.', color='cyan')

# 绘图
for i in range(n):
    color = colors[i % len(colors)]  # 循环使用颜色列表
    ax2.plot([x * width for x in GT_location_x[i]],
             [y * width for y in GT_location_y[i]],
             label=f'GT{i+1} Path', marker='^', color=color)

# 设置坐标轴标签和图例
ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.legend()
ax2.set_title('XY Plane View of Trajectories')
save_path = os.path.join(relative_path, 'XY Plane View of Trajectories.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight') 
#plt.show()
'''
# 绘制基准地面站路径
GT_location_x_baseline = [[] for _ in range(n)]
GT_location_y_baseline = [[] for _ in range(n)]
GT_location_z_baseline = [[] for _ in range(n)]

#填充存储位置的列表
for slot in GT_trajectory_x_baseline:
    for i in range(n):
        GT_location_x_baseline[i].append(slot[i])
GT_location_x_baseline = [np.array(loc) for loc in GT_location_x_baseline]

for slot in GT_trajectory_y_baseline:
    for i in range(n):
        GT_location_y_baseline[i].append(slot[i])
GT_location_y_baseline = [np.array(loc) for loc in GT_location_y_baseline]

for slot in GT_trajectory_z_baseline:
    for i in range(n):
        GT_location_z_baseline[i].append(slot[i])
GT_location_z_baseline = [np.array(loc) for loc in GT_location_z_baseline]

# 绘图
for i in range(n):
    ax.plot([x * width for x in GT_location_x_baseline[i]],
            [y * width for y in GT_location_y_baseline[i]],
            [z * height for z in GT_location_z_baseline[i]],
            label=f'GT Baseline{i+1} Path', marker='s')'''

'''
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # 添加一个 3D 坐标轴
# 绘制无人机飞行路径
ax.plot([x * width for x in UAV_trajectory_x],
        [y * width for y in UAV_trajectory_y],
        [z * height for z in UAV_trajectory_z],
        label='UAV Flight Path', marker='.', color='blue')
# 设置坐标轴标签
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')


n=num_ground_stations
# 初始化存储位置列表的列表
GT_location_x = [[] for _ in range(n)]
GT_location_y = [[] for _ in range(n)]
GT_location_z = [[] for _ in range(n)]

# 填充存储位置的列表
for slot in GT_trajectory_x:
    for i in range(n):
        GT_location_x[i].append(slot[i])
GT_location_x = [np.array(loc) for loc in GT_location_x]

for slot in GT_trajectory_y:
    for i in range(n):
        GT_location_y[i].append(slot[i])
GT_location_y = [np.array(loc) for loc in GT_location_y]

for slot in GT_trajectory_z:
    for i in range(n):
        GT_location_z[i].append(slot[i])
GT_location_z = [np.array(loc) for loc in GT_location_z]

# 绘图
for i in range(n):
    ax.plot([x * width for x in GT_location_x[i]],
            [y * width for y in GT_location_y[i]],
            [z * height for z in GT_location_z[i]],
            label=f'GT{i+1} Path', marker='^')

# 设置图例和标题
ax.legend()
ax.set_title('Combined 3D Trajectories of UAV and GT')

plt.show()


print(REWARD_BUFFER_baseline)
print(Data_rate_BUFFER_baseline)
print(Throughput_BUFFER_baseline)
print(Energy_efficiency_BUFFER_baseline)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # 添加一个 3D 坐标轴
# 绘制无人机飞行路径
ax.plot([x * width for x in UAV_trajectory_x_baseline],
        [y * width for y in UAV_trajectory_y_baseline],
        [z * height for z in UAV_trajectory_z_baseline],
        label='UAV Flight Path', marker='.', color='blue')
# 设置坐标轴标签
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')


n=num_ground_stations
# 初始化存储位置列表的列表
GT_location_x = [[] for _ in range(n)]
GT_location_y = [[] for _ in range(n)]
GT_location_z = [[] for _ in range(n)]

# 填充存储位置的列表
for slot in GT_trajectory_x_baseline:
    for i in range(n):
        GT_location_x[i].append(slot[i])
GT_location_x = [np.array(loc) for loc in GT_location_x]

for slot in GT_trajectory_y_baseline:
    for i in range(n):
        GT_location_y[i].append(slot[i])
GT_location_y = [np.array(loc) for loc in GT_location_y]

for slot in GT_trajectory_z_baseline:
    for i in range(n):
        GT_location_z[i].append(slot[i])
GT_location_z = [np.array(loc) for loc in GT_location_z]

# 绘图
for i in range(n):
    ax.plot([x * width for x in GT_location_x[i]],
            [y * width for y in GT_location_y[i]],
            [z * height for z in GT_location_z[i]],
            label=f'GT{i+1} Path', marker='^')

# 设置图例和标题
ax.legend()
ax.set_title('Combined 3D Trajectories of UAV and GT')

plt.show()'''