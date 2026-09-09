import math
import pandas as pd
import numpy as np
import gym
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Gate机制类
class GateMechanism(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.f_S = nn.Linear(feature_dim, 1)
        self.f_g = nn.Linear(feature_dim, 1)

    def forward(self, vlm_feat, isac_feat):
        g = torch.sigmoid(self.f_S(vlm_feat) + self.f_g(isac_feat))
        fused = g * vlm_feat + (1 - g) * isac_feat
        return fused, g

def fuse_vlm_isac(vlm_feat, isac_feat, gate_module=None):
    assert vlm_feat.shape == isac_feat.shape
    assert vlm_feat.shape[0] == 6
    feature_dim = vlm_feat.shape[1]
    if gate_module is None:
        gate_module = GateMechanism(feature_dim)
    return gate_module(vlm_feat, isac_feat)
"""seed_value = np.random.randint(0, 1000000)  # 生成一个随机种子
print("Random seed used:", seed_value)
random.seed(seed_value)
"""
seed_value = 946308
print("Random seed used:", seed_value)
random.seed(seed_value)

class DDPGEnvironment(gym.Env):
    def __init__(self, num_ground_stations, length, width, height):
        super(DDPGEnvironment, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ground_length = length
        self.ground_width = width
        self.height_max = height
        self.height_min = 100
        self.num_ground_stations = num_ground_stations
        self.gate_module = GateMechanism(feature_dim=1)
        self.vlm_features = None  # 供外部注入模拟VLM特征（如融合结果）
        self.vlm_fused_ready = False  # 初始标志，表示是否已融合过
        self.visited = np.zeros(self.num_ground_stations, dtype=bool)
        self.gate_module = GateMechanism(feature_dim=1).to(self.device)
        self.gate_optimizer = optim.Adam(self.gate_module.parameters(), lr=1e-3)
        self.gate_loss_fn = nn.MSELoss()
        self.train_gate_debug = True
        self.vel_h_max = 10
        self.vel_v_max = 10
        self.t = 0.5
        self.B = 2 * 10 ** 6
        self.e_battery_uav = 58280 / 2
        # 位置
        self.loc_uav = [0, 0, 0]
        self.ground_stations = []
        self.index = np.ones(self.num_ground_stations)
        gtx = [83.7783676828072, 195.01460128930486, 197.22939424300583, 156.77595419183504, 47.787914691092716,
               72.34721215639041]
        gty = [19.625158188947633, 128.7265491397583, 5.549427877581148, 184.97725028744216, 82.33277292670245,
               117.37779460371279]
        for i in range(self.num_ground_stations):
            '''x = random.uniform(0, 0.5*self.ground_length)
            y = random.uniform(0, 0.5*self.ground_width)'''
            x = gtx[i] / 2
            y = gty[i] / 2
            z = 0
            self.ground_stations.append(np.array([x, y, z]))

    def step(self, action, state, k):
        """
        Unified step that only uses k to choose between 4 experiment groups:
          k==0: DDPG-VLM-ISAC (train gate)
          k==1: DDPG (ISAC-only baseline)
          k==2: TSP-VLM-ISAC (train gate, same as k==0 training)
          k==3: TSP (ISAC-only baseline)
        Returns:
          next_state, reward, flag, throughput/(1e6), r_kn/(1e6), Energy_efficiency/(1e3), fusion_debug
        """
        flag = False
        r_kn = 0.0

        # ----------------- UAV / GT pos -----------------
        uav_x = state[0] * self.ground_length
        uav_y = state[1] * self.ground_length
        uav_z = state[2] * self.ground_length
        loc_uav = [uav_x, uav_y, uav_z]

        GT_location_x = state[3:3 + self.num_ground_stations] * self.ground_length
        GT_location_y = state[3 + self.num_ground_stations:3 + 2 * self.num_ground_stations] * self.ground_length
        GT_location_z = [0 for _ in range(self.num_ground_stations)]
        loc_gt = np.zeros([self.num_ground_stations, 3])
        for i in range(self.num_ground_stations):
            loc_gt[i] = [GT_location_x[i], GT_location_y[i], GT_location_z[i]]

        # GT 随机移动（沿用你的原逻辑）
        walkspeed = random.uniform(0, 3)
        for gs in loc_gt:
            jiaodu = random.uniform(0, 1) * 2 - 1
            alpha = jiaodu * np.pi
            if 0 < gs[0] + walkspeed * self.t * math.cos(alpha) < self.ground_length:
                gs[0] += walkspeed * self.t * math.cos(alpha)
            if 0 < gs[1] + walkspeed * self.t * math.sin(alpha) < self.ground_width:
                gs[1] += walkspeed * self.t * math.sin(alpha)

        # ----------------- UAV 控制参数解析 -----------------
        if k == 1:
            x_hm = action[self.num_ground_stations] + 1 + random.uniform(-0.4, 0)
        else:
            x_hm = action[self.num_ground_stations] + 1
        theta = x_hm * math.pi
        vel_horizontal = (action[self.num_ground_stations + 1] + 1) / 2
        vel_vertical = action[self.num_ground_stations + 2]

        self.distance_horizontal = vel_horizontal * self.vel_h_max * self.t
        dx_uav = self.distance_horizontal * math.cos(theta)
        dy_uav = self.distance_horizontal * math.sin(theta)
        self.distance_vertical = vel_vertical * self.vel_v_max * self.t
        dz_uav = self.distance_vertical

        if 0 <= loc_uav[0] + dx_uav <= self.ground_length:
            loc_uav[0] += dx_uav
        else:
            vel_horizontal = 0
        if 0 <= loc_uav[1] + dy_uav <= self.ground_width:
            loc_uav[1] += dy_uav
        else:
            vel_horizontal = 0
        if self.height_min <= loc_uav[2] + dz_uav <= self.height_max:
            loc_uav[2] += dz_uav
        else:
            vel_vertical = 0

        loc_uav_array = np.array(loc_uav)
        loc_gt_array = np.array([gt for gt in loc_gt])
        distances_UAV_GT_horizontal = np.zeros(loc_gt_array.shape[0])

        # 带宽归一（鲁棒）
        bandwidth = np.zeros(loc_gt_array.shape[0])
        total_bw = 0.0
        for i in range(loc_gt_array.shape[0]):
            bandwidth[i] = (action[i] + 1) / 2
            total_bw += bandwidth[i]
        if total_bw > 0:
            bandwidth = bandwidth / total_bw

        # ----------------- 噪声设置-----------------
        sigma_ddpg_baseline = 100.0
        sigma_fusion_isac = 10.0
        sigma_fusion_vlm = 10.0
        sigma_isac_baseline = 20.0
        # helper：融合函数（优先调用 fuse_vlm_isac，否则回退到 self.gate_module）
        def _fuse_arrays(vlm_arr, isac_arr):
            if getattr(self, "gate_module", None) is None:
                self.gate_module = GateMechanism(feature_dim=1)
            _gate_device = next(self.gate_module.parameters()).device
            vlm_t = torch.tensor(vlm_arr, dtype=torch.float32, device=_gate_device).view(self.num_ground_stations, -1)
            isac_t = torch.tensor(isac_arr, dtype=torch.float32, device=_gate_device).view(self.num_ground_stations, -1)
            try:
                fused_tensor, g_weights = fuse_vlm_isac(vlm_t, isac_t, self.gate_module)
            except Exception:
                fused_tensor, g_weights = self.gate_module(vlm_t, isac_t)
            return fused_tensor, g_weights

        fusion_debug = {}

        # ================= 分支：k==0,1,2,3 =================
        if k == 0:
            # DDPG-VLM-ISAC: 融合并训练 gate（DDPG 场景）
            d_true = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[:, :2], axis=1)
            isac_meas = d_true + np.random.normal(0, sigma_fusion_isac, size=(loc_gt_array.shape[0],))
            vlm_meas = d_true + np.random.normal(0, sigma_fusion_vlm, size=(loc_gt_array.shape[0],))

            fused_tensor, g_weights = _fuse_arrays(vlm_meas, isac_meas)
            fused_d = fused_tensor.view(-1).detach().cpu().numpy()

            for i in range(self.num_ground_stations):
                distances_UAV_GT_horizontal[i] = fused_d[i]
                r_kn += bandwidth[i] * self.communication(loc_uav_array[2], fused_d[i])

            fusion_debug = {
                "fused": fused_tensor.detach().cpu().numpy().reshape(self.num_ground_stations, 1),
                "g": g_weights.detach().cpu().numpy().reshape(self.num_ground_stations, 1),
                "vlm": vlm_meas.reshape(self.num_ground_stations, 1),
                "isac": isac_meas.reshape(self.num_ground_stations, 1)
            }

            # Gate 训练（与之前你希望的训练流程一致）
            try:
                d_gt = np.linalg.norm(loc_gt_array[:, :2] - loc_uav_array[:2], axis=1)
                _gate_device = next(self.gate_module.parameters()).device
                gt_tensor = torch.tensor(d_gt, dtype=torch.float32, device=_gate_device).view(self.num_ground_stations,
                                                                                              -1)
                if hasattr(self, "gate_loss_fn") and hasattr(self, "gate_optimizer"):
                    loss = self.gate_loss_fn(fused_tensor, gt_tensor)
                    self.gate_optimizer.zero_grad()
                    loss.backward()
                    self.gate_optimizer.step()
                    if getattr(self, "train_gate_debug", False):
                        print(f"[Gate训练 k=0] Loss={loss.item():.4f}, g={g_weights.view(-1).detach().cpu().numpy()}")
            except Exception as e:
                if getattr(self, "train_gate_debug", False):
                    print("[Gate训练 k=0] 跳过，原因：", e)

        elif k == 1:
            # DDPG-only
            for i in range(loc_gt_array.shape[0]):
                true_distance = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[i, :2])
                gaussian_noise = np.random.normal(0, sigma_ddpg_baseline)
                scale_noise = np.random.normal(1.0, 0.2)
                estimated_distance = true_distance * scale_noise + gaussian_noise
                distances_UAV_GT_horizontal[i] = max(0, estimated_distance)
                r_kn += bandwidth[i] * self.communication(loc_uav_array[2], distances_UAV_GT_horizontal[i])

        elif k == 2:
            # TSP-VLM-ISAC: TSP 控制 + 融合
            d_true = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[:, :2], axis=1)
            isac_meas = d_true + np.random.normal(0, sigma_fusion_isac, size=(loc_gt_array.shape[0],))
            vlm_meas = d_true + np.random.normal(0, sigma_fusion_vlm, size=(loc_gt_array.shape[0],))

            fused_tensor, g_weights = _fuse_arrays(vlm_meas, isac_meas)
            fused_d = fused_tensor.view(-1).detach().cpu().numpy()

            for i in range(self.num_ground_stations):
                distances_UAV_GT_horizontal[i] = fused_d[i]
                r_kn += bandwidth[i] * self.communication(loc_uav_array[2], fused_d[i])

            fusion_debug = {
                "fused": fused_tensor.detach().cpu().numpy().reshape(self.num_ground_stations, 1),
                "g": g_weights.detach().cpu().numpy().reshape(self.num_ground_stations, 1),
                "vlm": vlm_meas.reshape(self.num_ground_stations, 1),
                "isac": isac_meas.reshape(self.num_ground_stations, 1)
            }

            # Gate 训练（与 k==0 相同）
            try:
                d_gt = np.linalg.norm(loc_gt_array[:, :2] - loc_uav_array[:2], axis=1)
                _gate_device = next(self.gate_module.parameters()).device
                gt_tensor = torch.tensor(d_gt, dtype=torch.float32, device=_gate_device).view(self.num_ground_stations,
                                                                                              -1)
                if hasattr(self, "gate_loss_fn") and hasattr(self, "gate_optimizer"):
                    loss = self.gate_loss_fn(fused_tensor, gt_tensor)
                    self.gate_optimizer.zero_grad()
                    loss.backward()
                    self.gate_optimizer.step()
                    if getattr(self, "train_gate_debug", False):
                        print(f"[Gate训练 k=2] Loss={loss.item():.4f}, g={g_weights.view(-1).detach().cpu().numpy()}")
            except Exception as e:
                if getattr(self, "train_gate_debug", False):
                    print("[Gate训练 k=2] 跳过，原因：", e)

        elif k == 3:
            # TSP-only baseline: TSP 控制 + ISAC-only 测距
            for i in range(loc_gt_array.shape[0]):
                d_true = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[i, :2])
                meas = d_true + np.random.normal(0, sigma_isac_baseline)
                distances_UAV_GT_horizontal[i] = meas
                r_kn += bandwidth[i] * self.communication(loc_uav_array[2], meas)

            fusion_debug = {
                "fused": distances_UAV_GT_horizontal.reshape(self.num_ground_stations, 1).copy(),
                "g": np.zeros((self.num_ground_stations, 1)),
                "vlm": np.zeros((self.num_ground_stations, 1)),
                "isac": distances_UAV_GT_horizontal.reshape(self.num_ground_stations, 1).copy()
            }

        else:
            raise ValueError(f"Unsupported k value: {k}")

        # ----------------- 能耗 & reward & next_state -----------------
        self.E_uav_cost = self.flight_energy_slot(abs(vel_horizontal * self.vel_h_max),
                                                  abs(vel_vertical * self.vel_v_max))
        self.e_battery_uav = self.e_battery_uav - self.E_uav_cost
        reward = (r_kn / (10 ** 6)) / (self.E_uav_cost / self.flight_energy_slot(10.4, 0))

        next_state = self._get_state(loc_uav, loc_gt)
        throughput = r_kn * self.t
        Energy_efficiency = throughput / self.E_uav_cost

        # fusion_debug 占位（避免 KeyError）
        if not fusion_debug:
            fusion_debug = {
                "fused": np.zeros((self.num_ground_stations, 1)),
                "g": np.zeros((self.num_ground_stations, 1)),
                "vlm": np.zeros((self.num_ground_stations, 1)),
                "isac": np.zeros((self.num_ground_stations, 1))
            }

        return next_state, reward, flag, throughput / (10 ** 6), r_kn / (10 ** 6), Energy_efficiency / (
                    10 ** 3), fusion_debug

    def reset(self, k):  # ok
        self.e_battery_uav = 58280/2

        loc_uav = [0, 0, 200]
        ground_stations = []
        self.index = np.ones(self.num_ground_stations)

        # 183426
        #gtx = [83.7783676828072 / 2, 195.01460128930486 / 2, 197.22939424300583 / 2, 156.77595419183504 / 2,47.787914691092716 / 2, 72.34721215639041 / 2]
        #gty = [19.625158188947633 / 2, 128.7265491397583 / 2, 5.549427877581148 / 2, 184.97725028744216 / 2,82.33277292670245 / 2, 117.37779460371279 / 2]
        # 887115
        #gtx=[63.72676027,81.8112029,  22.59115723, 53.79039247, 27.13087413, 31.82783903]
        #gty=[87.9689825,2.76791846, 24.88115209 , 9.23983119, 33.27251303, 42.29242421]
        # 958030,good
        gtx=[36.997030855955494, 53.19896748977524, 20.019889656911783, 13.257760934502294, 54.30245664199407, 74.13939293385138]
        gty=[43.30032996341227, 92.92182147894647, 82.01102859458813, 59.79515818116455, 30.841857576617237, 67.14503554078097]
        # 946308
        gtx = [77.96764806654085, 74.18565360280212, 48.692400122184196, 33.07706148218591, 63.633839745957474,73.8319243334739]
        gty = [22.416295089579584, 73.63781588916122, 11.553216744777595, 96.98676837896325, 73.86636674953102,78.21293871678411]
        for i in range(self.num_ground_stations):
            '''x = random.uniform(0, 0.5*self.ground_length)
            y = random.uniform(0, 0.5*self.ground_width)'''
            x = gtx[i] + 100
            y = gty[i] + 100
            z = 0
            ground_stations.append(np.array([x, y, z]))
        return self._get_state(loc_uav, ground_stations)

    def _get_state(self, loc_uav, ground_stations):
        uav_normalized = np.array([loc_uav[0] / self.ground_length,
                                   loc_uav[1] / self.ground_width,
                                   loc_uav[2] / self.height_max])
        gs_positions_x = np.array([gs[0] / self.ground_length for gs in ground_stations])
        gs_positions_y = np.array([gs[1] / self.ground_width for gs in ground_stations])
        gs_positions_z = np.array([gs[2] / self.height_max for gs in ground_stations])

        state = np.concatenate((uav_normalized, gs_positions_x, gs_positions_y, gs_positions_z, uav_normalized,
                                uav_normalized, uav_normalized, uav_normalized, uav_normalized))
        return state

    def flight_energy_slot(self, vel, vel_v):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (omega ** 3) * (R ** 3)
        P1 = (1 + k) * (W ** (3 / 2) / math.sqrt(2 * rho * G))
        P2 = 11.46
        # P2=6.86
        if (1 + vel ** 4 / (4 * v_o ** 4)) <= 0 or (
                math.sqrt(1 + vel ** 4 / (4 * v_o ** 4)) - vel ** 2 / (2 * v_o ** 2)) <= 0:
            Energy_uav = self.t * (
                        P0 * (1 + 3 * vel ** 2 / U_tip ** 2) + (1 / 2) * d_o * rho * s * G * vel ** 3 + P1 * math.sqrt(
                    abs(math.sqrt(abs(1 + vel ** 4 / (4 * v_o ** 4))) - vel ** 2 / (2 * v_o ** 2))) + P2 * vel_v)
        else:
            Energy_uav = self.t * (
                        P0 * (1 + 3 * vel ** 2 / U_tip ** 2) + (1 / 2) * d_o * rho * s * G * vel ** 3 + P1 * math.sqrt(
                    math.sqrt(1 + vel ** 4 / (4 * v_o ** 4)) - vel ** 2 / (2 * v_o ** 2)) + P2 * vel_v)
        return Energy_uav

    def communication(self, h, d):
        a = 9.61
        b = 0.16
        c = 3 * 10 ** 8
        n_los = 1
        n_nlos = 20
        A = n_los - n_nlos
        P_k = 0.005  # 单位W
        N_0 = 10 ** (-169 / 10)  # -169dBm/Hz
        B = self.B
        f_c = 9 * 10 ** 8
        C = 20 * math.log10(4 * math.pi * f_c / c) + n_nlos
        p_kn = 1 + a * math.exp(a * b - b * math.atan(h / d) / math.pi * 180)
        p_kn = 1 / p_kn
        l_kn = 20 * math.log10(math.sqrt(h ** 2 + d ** 2)) + A * p_kn + C
        r_kn = B * math.log2(1 + P_k * 10 ** (-l_kn / 10) / (B * N_0))
        return r_kn

    def baseline(self, state):
        action = [-1 for i in range(self.num_ground_stations + 3)]
        uav_x = state[0] * self.ground_length
        uav_y = state[1] * self.ground_length  # 获取无人机当前位置
        loc_uav_array = [uav_x, uav_y, 100]

        nearest_gs_index = -1  # 最近的地面基站索引
        min_distance = 1000
        GT_location_x = state[3:3 + self.num_ground_stations] * self.ground_length
        GT_location_y = state[3 + self.num_ground_stations:3 + 2 * self.num_ground_stations] * self.ground_length
        GT_location_z = [0 for _ in range(self.num_ground_stations)]
        GT_location_x = [np.array(loc) for loc in GT_location_x]
        GT_location_y = [np.array(loc) for loc in GT_location_y]
        GT_location_z = [np.array(loc) for loc in GT_location_z]
        loc_gt_array = np.zeros([self.num_ground_stations, 3])
        distances_UAV_GT_horizontal = np.zeros(self.num_ground_stations)

        for i in range(self.num_ground_stations):
            loc_gt_array[i] = [GT_location_x[i], GT_location_y[i], GT_location_z[i]]
        # 遍历地面基站，找到距离最近的地面基站索引
        for i in range(self.num_ground_stations):
            distances_UAV_GT_horizontal[i] = self.index[i] * np.linalg.norm(
                loc_uav_array[:2] - loc_gt_array[i, :2])  # 假设loc_gt_array[i]包含x和y坐标
            if distances_UAV_GT_horizontal[i] < min_distance:
                min_distance = distances_UAV_GT_horizontal[i]
                nearest_gs_index = i

        if distances_UAV_GT_horizontal[nearest_gs_index] < 3:
            # print(nearest_gs_index)
            # print(distances_UAV_GT_horizontal)
            self.index[nearest_gs_index] = 1000000
            # print(self.index)

        angle = angle_from_uav(loc_uav_array[:2], loc_gt_array[nearest_gs_index, :2])
        if all(x >= 1000000 for x in self.index):
            action[self.num_ground_stations] = 0  # 差异1
            action[self.num_ground_stations + 1] = -1
            action[self.num_ground_stations + 2] = 0
            # action[nearest_gs_index]=1
        else:
            action[self.num_ground_stations] = angle / np.pi - 1  # 差异1
            # action[self.num_ground_stations+1]= min(10.0, min_distance)/self.vel_h_max*2-1
            action[self.num_ground_stations + 1] = 1
            action[self.num_ground_stations + 2] = 0
            # action[nearest_gs_index] = self.num_ground_stations
            action[nearest_gs_index] = 1
        return action

def angle_from_uav(uav, gt):
    # 计算从无人机到目标的向量
    vector = gt - uav
    # 使用arctan2计算角度，输入参数是(y, x)
    angle = np.arctan2(vector[1], vector[0])
    if angle < 0:
        angle += 2 * np.pi  # 转换角度范围从 [0, 2*pi]
    return angle

