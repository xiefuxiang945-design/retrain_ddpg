import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from TSP_env_new import DDPGEnvironment
from torch import nn

# ========================
# Gate机制类
# ========================
class GateMechanism(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim * 2, 1)

    def forward(self, vlm_feat, isac_feat):
        g = torch.sigmoid(self.fc(torch.cat([vlm_feat, isac_feat], dim=-1)))
        fused = g * vlm_feat + (1 - g) * isac_feat
        return fused, g

def fuse_vlm_isac(vlm_feat, isac_feat, gate_module=None):
    assert vlm_feat.shape == isac_feat.shape, "输入特征维度必须相同"
    assert vlm_feat.shape[0] == 6, "应为6个GT目标"
    if gate_module is None:
        gate_module = GateMechanism(vlm_feat.shape[1])
    return gate_module(vlm_feat, isac_feat)


# ========================
# 参数设置
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_ground_stations = 6
width = 200
height = 200
num_episodes = 100
num_step = 100
env = DDPGEnvironment(num_ground_stations, width, width, height=height)

font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')



# ========================
# 绘制轨迹函数
# ========================
def plot_trajectory(ax, x_data, y_data, z_data, label, color, marker, step=5, markersize=40):
    ax.plot([x * width for x in x_data],
            [y * width for y in y_data],
            [z * height for z in z_data],
            label=label, color=color)
    ax.scatter([x * width for x in x_data[::step]],
               [y * width for y in y_data[::step]],
               [z * height for z in z_data[::step]],
               color=color, marker=marker, s=markersize)
def generate_fixed_gt_trajectory(env, num_step):
    """生成一次固定GT轨迹（不区分算法）"""
    state = env.reset(1)
    GT_x_ep = [[] for _ in range(num_ground_stations)]
    GT_y_ep = [[] for _ in range(num_ground_stations)]
    GT_z_ep = [[] for _ in range(num_ground_stations)]

    for i in range(num_ground_stations):
        GT_x_ep[i].append(state[3+i])
        GT_y_ep[i].append(state[3+num_ground_stations+i])
        GT_z_ep[i].append(state[3+2*num_ground_stations+i])

    for _ in range(num_step):
        # 这里只更新 GT 位置，不动 UAV
        action = [-1 for _ in range(num_ground_stations+3)]  # UAV 不移动
        next_state, _, _, _, _, _, _ = env.step(action, state, 1, mode=0)  # 用 ISAC 模式只是为了调用 step
        for i in range(num_ground_stations):
            GT_x_ep[i].append(next_state[3+i])
            GT_y_ep[i].append(next_state[3+num_ground_stations+i])
            GT_z_ep[i].append(next_state[3+2*num_ground_stations+i])
        state = next_state

    return GT_x_ep, GT_y_ep, GT_z_ep

# 先生成一次固定 GT 轨迹
fixed_GT_x, fixed_GT_y, fixed_GT_z = generate_fixed_gt_trajectory(env, num_step)
# ========================
# 三种测距方法
# ========================
methods = {
    "Baseline-ISAC": 0,
    "VLM": 1,
    "Fusion": 2
}

results = {}

for method_name, mode in methods.items():
    print(f"Running {method_name} ...")
    episode_rewards = []
    episode_throughputs = []
    episode_datarates = []
    episode_ees = []

    UAV_x, UAV_y, UAV_z = [], [], []

    for ep in range(num_episodes):
        state = env.reset(1)  # 这里 reset 后也会初始化 GT，但我们只用 UAV 轨迹
        total_reward = 0
        total_throughput = 0
        total_datarate = 0
        total_ee = 0

        UAV_x_ep, UAV_y_ep, UAV_z_ep = [state[0]], [state[1]], [state[2]]

        for step_i in range(num_step):
            action = env.baseline(state, mode=mode)
            next_state, reward, done, throughput, data_rate, ee, _ = env.step(action, state, 1, mode=mode)
            UAV_x_ep.append(next_state[0])
            UAV_y_ep.append(next_state[1])
            UAV_z_ep.append(next_state[2])

            total_reward += reward
            total_throughput += throughput
            total_datarate += data_rate
            total_ee += ee
            state = next_state
            if done:
                break

        episode_rewards.append(total_reward)
        episode_throughputs.append(total_throughput / num_step)
        episode_datarates.append(total_datarate / num_step)
        episode_ees.append(total_ee / num_step)

        if ep == num_episodes - 1:
            UAV_x, UAV_y, UAV_z = UAV_x_ep, UAV_y_ep, UAV_z_ep

    results[method_name] = {
        "reward": episode_rewards,
        "throughput": episode_throughputs,
        "datarate": episode_datarates,
        "ee": episode_ees,
        "uav_x": UAV_x,
        "uav_y": UAV_y,
        "uav_z": UAV_z
    }


    print(f"{method_name} -> Reward: {np.mean(episode_rewards):.2f}, "
          f"Throughput: {np.mean(episode_throughputs):.2f} Mbps, "
          f"Data rate: {np.mean(episode_datarates):.2f} Mbps, "
          f"EE: {np.mean(episode_ees):.2f} Kbps/J")


# ========================
# 绘制能效对比曲线
# ========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三条 UAV 轨迹
colors = {"Baseline-ISAC": "blue", "VLM": "orange", "Fusion": "green"}
for method_name, data in results.items():
    plot_trajectory(ax, data["uav_x"], data["uav_y"], data["uav_z"],
                    label=method_name, color=colors[method_name], marker='o')

# 绘制固定 GT 轨迹
gt_colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink']
for i in range(num_ground_stations):
    plot_trajectory(ax, fixed_GT_x[i], fixed_GT_y[i], fixed_GT_z[i],
                    label=f"GT{i+1}", color=gt_colors[i], marker='^')

ax.legend()
plt.show()