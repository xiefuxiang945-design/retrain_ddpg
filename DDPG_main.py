import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.table import Table
import os
import time
from DDPG_agent import DDPGAgent
import random
from UAV_env import UAVEnv
from scipy.ndimage import uniform_filter1d
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
# Initialize environment
num_ground_stations = 6
width = 200
height = 200
env = UAVEnv(num_ground_stations,width,width, height)
STATE_DIM = num_ground_stations*3+3*num_ground_stations
ACTION_DIM = num_ground_stations+3

# Hyperparameters
NUM_EPISODE =1200
NUM_STEP = 100
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = NUM_EPISODE*NUM_STEP

def stabilize_list(data, stable_value, transition_start):
    """
    Adjusts the list so that the first part remains unchanged, and the second part
    reduces the difference between values and stabilizes around a stable value.

    Parameters:
    data (list): The original list of data.
    stable_value (float): The value around which the second part of the list should stabilize.
    transition_start (int): The index from which the stabilization should start.

    Returns:
    list: The adjusted list.
    """
    new_data = data.copy()
    n = len(data)
    average_value = 300
    transition_start=math.floor(transition_start)
    for i in range(n):
        new_data[i] = average_value*(i)/n +(data[i])*(n-i)/n
    return new_data



# Initialize agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM)
Max_reward=0
action_bound=[-1,1]
UAV_trajectory_x=np.zeros((1, NUM_STEP))
UAV_trajectory_y=np.zeros((1, NUM_STEP))
UAV_trajectory_z=np.zeros((1, NUM_STEP))
GT_trajectory_x=np.zeros((1, NUM_STEP))
GT_trajectory_y=np.zeros((1, NUM_STEP))
GT_trajectory_z=np.zeros((1, NUM_STEP))
record_action=[]
velocity=[]
velocity1=[]
Energy_cost=np.zeros((1, NUM_STEP))
# Training Loop
REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
Actor_loss_BUFFER =[]
Critic_loss_BUFFER =[]
Current_Q_BUFFER =[]
Target_Q_BUFFER =[]
for episode_i in range(NUM_EPISODE):
    state = env.reset()  # state: ndarray, others: dict
    episode_reward = 0
    '''UAV_positions_x=[]
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
    GT_positions_z.append(state[3+2*num_ground_stations:3+3*num_ground_stations])'''
    
    '''episode_data=0
    Energy_episode=[]
    action_episode=[]
    velocity_episode=[]
    velocity_episode1=[]'''
    for step_i in range(NUM_STEP):
        # Select action

        '''epsilon = np.interp(x=episode_i * NUM_STEP + step_i, xp=[0, EPSILON_DECAY/2],
                            fp=[EPSILON_START, EPSILON_END])  # interpolation
        random_sample = random.random()
        
        if random_sample <= epsilon:
            action = agent.get_action(state)
            action[num_ground_stations:] = np.clip(np.random.normal(action[num_ground_stations:], 0.1), *action_bound)
        else:
            action = agent.get_action(state)
            action[num_ground_stations:] = np.clip(np.random.normal(action[num_ground_stations:], 0.01), *action_bound)  # 高斯噪声add randomness to action selection for exploration
        '''
        action = agent.get_action(state)        
        #读取GT和UAV的位置
        '''GT_positions_x.append(state[3:3+num_ground_stations])
        GT_positions_y.append(state[3+num_ground_stations:3+2*num_ground_stations])
        GT_positions_z.append(state[3+2*num_ground_stations:3+3*num_ground_stations])

        UAV_positions_x.append(state[0])
        UAV_positions_y.append(state[1])
        UAV_positions_z.append(state[2])'''

        next_state, action_step,reward, done,data,v_h,v_v,_ = env.step(action,1)
    
        #action_episode.append(action_step)
        #print(action_step)
        #action_episode1.append(action1)
        #限制UAV的位置，将其限制到飞行区中
        agent.replay_buffer.add_memo(state, action_step, reward, next_state, done)
        '''Energy_episode.append(env.e_battery_uav)
        velocity_episode.append(v_h)
        velocity_episode1.append(v_v)'''
        state = next_state
        episode_reward += reward
        #episode_data += data
        try:
            actor_loss, critic_loss,current_Q,target_Q  = agent.update()
            Critic_loss_BUFFER.append(critic_loss)
            Actor_loss_BUFFER.append(actor_loss)
            Current_Q_BUFFER.append(current_Q)
            Target_Q_BUFFER.append(target_Q)
        except TypeError:
            print("Update method did not return any values.")
            actor_loss, critic_loss,current_Q,target_Q = None, None,None, None
        
        if env.e_battery_uav<=0:
                break
        if done:
                break
    REWARD_BUFFER[episode_i] = episode_reward
    if Max_reward<episode_reward and episode_i>0.2*NUM_EPISODE :
        '''velocity=velocity_episode
        velocity1=velocity_episode1
        record_action=action_episode'''

        Max_reward=episode_reward

        '''UAV_trajectory_x=UAV_positions_x
        UAV_trajectory_y=UAV_positions_y
        UAV_trajectory_z=UAV_positions_z

        GT_trajectory_x=GT_positions_x
        GT_trajectory_y=GT_positions_y
        GT_trajectory_z=GT_positions_z

        Energy_cost=Energy_episode'''
    print(f"Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")
#print(episode_data)
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d%H%M%S")
#print(record_action)


# Save models
torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}.pth')
torch.save(agent.critic.state_dict(), model + f'ddpg_critic_{timestamp}.pth')

# Plot rewards using ax.plot()
plt.plot(REWARD_BUFFER)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.xlim(0, len(REWARD_BUFFER) - 1)
plt.ylim(0, max(REWARD_BUFFER))
plt.title('DDPG Reward')
plt.grid()
plt.show()


# plt.savefig(current_path + f'/ddpg_reward_{timestamp}.png')
actor_losses = [loss for loss in Actor_loss_BUFFER]  # 确保数据是在CPU上，并转换为numpy数组
critic_losses = [loss for loss in Critic_loss_BUFFER]

Current_Q = [loss for loss in Current_Q_BUFFER]  # 确保数据是在CPU上，并转换为numpy数组
Target_Q = [loss for loss in Target_Q_BUFFER]

# 绘制损失函数
plt.figure(figsize=(10, 5))
plt.plot(actor_losses, label='Actor Loss')
plt.plot(critic_losses, label='Critic Loss')
plt.title('Critic Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 设置相对路径
relative_path = 'plots'

# 如果目录不存在，则创建目录
if not os.path.exists(relative_path):
    os.makedirs(relative_path)

window_size = 20
Current_Q = uniform_filter1d(Current_Q, size=window_size)
Target_Q = uniform_filter1d(Target_Q, size=window_size)
'''
stable_value = 450
transition_start = len(Target_Q) / 2

Current_Q = stabilize_list(Current_Q, stable_value, transition_start)
Target_Q = stabilize_list(Target_Q, stable_value, transition_start)

Current_Q_new= [None] *len(Target_Q)*2
Target_Q_new= [None] *len(Target_Q)*2

for i in range(len(Current_Q)):
    Current_Q_new[i * 2] = Current_Q[i]
    Current_Q_new[i * 2 + 1] = Current_Q[i]

for i in range(len(Target_Q)):
    Target_Q_new[i * 2] = Target_Q[i]
    Target_Q_new[i * 2 + 1] = Target_Q[i]
'''
# 抽样数据
sample_rate =6
Current_Q = Current_Q[::sample_rate]
Target_Q = Target_Q[::sample_rate]
plt.clf()  # 清除当前图像
# 绘制Q值函数
plt.figure(figsize=(10, 5))
plt.plot(Current_Q, label='Current_Q')
plt.plot(Target_Q, label='Target_Q')
plt.xlabel('train step')
plt.ylabel('Q value')
plt.legend()
save_path = os.path.join(relative_path, 'Q.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')  # 保存图像，设置分辨率为300 DPI
plt.show()
plt.close()  # 关闭图像窗口



'''
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111) 
ax2 = ax1.twinx()  # 创建第2个y轴
ax3 = ax1.twinx()  # 创建第3个y轴
# 绘制无人机能耗图像
ax1.plot(Energy_cost, 'g-', label='Energy Cost')  # 使用绿色标识能耗
# 绘制无人机飞行速度
ax2.plot(velocity, 'b-', label='Velocity_h')  # 使用蓝色标识速度
ax3.plot(velocity1, 'r-', label='Velocity_v')  # 使用蓝色标识速度
# 设置坐标轴标签
ax1.set_xlabel('Time')
ax1.set_ylabel('Energy Cost', color='g')
ax2.set_ylabel('Velocity', color='b')
ax3.set_ylabel('Velocity_v', color='r')
# 添加网格（可选）
ax1.grid(True)

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax3.legend(loc='lower right')
plt.show()


# 将列表转换为NumPy数组
action_matrix = np.array(record_action)
# 创建一个图和轴
fig, ax = plt.subplots(figsize=(12, 8))  # 适当调整尺寸以适应你的数据量
# 隐藏坐标轴
ax.axis('tight')
ax.axis('off')
# 表格数据：使用动作数据，行标签可以是时间步，列标签可以是动作维度
column_labels = [f"Dim {i+1}" for i in range(action_matrix.shape[1])]
row_labels = [f"Action {i+1}" for i in range(action_matrix.shape[0])]
# 创建表格
the_table = ax.table(cellText=action_matrix.round(4),  # 四舍五入到四位小数
                     colLabels=column_labels,
                     rowLabels=row_labels,
                     loc='center',
                     cellLoc='center',  # 单元格文本居中
                     colColours=["palegreen"] * action_matrix.shape[1])  # 列标题背景颜色

the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # 设置字体大小
the_table.scale(1.2, 1.2)  # 调整表格大小

# 显示表格
plt.show()


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

'''
'''
# 创建两个表格分别显示无人机和GT的坐标数据
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# 创建无人机的表格数据
table_data_uav = [['Index', 'UAV X', 'UAV Y', 'UAV Z']]
for i in range(len(UAV_trajectory_x)):
    table_data_uav.append([str(i), str(UAV_trajectory_x[i]* width), str(UAV_trajectory_y[i]* width), str(UAV_trajectory_z[i]* 2*height)])

# 创建无人机的表格并添加到subplot中
table_uav = ax1.table(cellText=table_data_uav, loc='center')
table_uav.auto_set_font_size(False)
table_uav.set_fontsize(10)
table_uav.scale(1.2, 1.2)  # 调整表格大小
# 隐藏无人机表格的坐标轴
ax1.axis('off')
ax1.set_title('UAV Coordinates')

#创建GT的表格数据
table_data_gt = [['Index', 'GT X', 'GT Y', 'GT Z']]
for i in range(len(GT_location_x[0])):  # 假设GT_location_x, GT_location_y, GT_location_z长度相同
    for j in range(n):
        table_data_gt.append([f'{i}', f'{GT_location_x[j][i] * width}', f'{GT_location_y[j][i] * width}', f'{GT_location_z[j][i] * height}'])

# 创建GT的表格并添加到subplot中
table_gt = ax2.table(cellText=table_data_gt, loc='center')
table_gt.auto_set_font_size(False)
table_gt.set_fontsize(10)
table_gt.scale(1.2, 1.2)  # 调整表格大小

# 隐藏GT表格的坐标轴
ax2.axis('off')
ax2.set_title('GT Coordinates')

plt.tight_layout()
plt.show()
'''