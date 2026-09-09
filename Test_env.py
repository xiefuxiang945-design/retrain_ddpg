import math
import pandas as pd
import numpy as np
import gym
import copy
import random
seed_value = np.random.randint(0, 1000000)  # 生成一个随机种子
#print("Random seed used:", seed_value)
random.seed(seed_value)


class DDPGEnvironment(gym.Env):
    def __init__(self, num_ground_stations,length, width , height):
        super(DDPGEnvironment, self).__init__()
        self.ground_length = length
        self.ground_width = width
        self.height_max = height
        self.height_min=100
        self.num_ground_stations = num_ground_stations

        self.vel_h_max = 10
        self.vel_v_max = 10
        self.t=0.5
        self.B=2*10**6
        self.e_battery_uav = 58280/2
        #位置
        self.loc_uav = [0,0,0]
        self.ground_stations = []
        self.index=np.ones(self.num_ground_stations)
        gtx=[83.7783676828072,195.01460128930486,197.22939424300583,156.77595419183504,47.787914691092716,72.34721215639041]
        gty=[19.625158188947633, 128.7265491397583, 5.549427877581148, 184.97725028744216, 82.33277292670245, 117.37779460371279]
        for i in range(num_ground_stations):
            x= random.uniform(0, 0.5*self.ground_length)
            y = random.uniform(0, 0.5*self.ground_width)
            '''x=gtx[i]/2
            y=gty[i]/2'''
            z = 0
            self.ground_stations.append({'position': np.array([x, y, z])})


    def step(self, action,k):
        flag = False
        r_kn=0
        walkspeed=random.uniform(0, 2)
        for gs in self.ground_stations:
            jiaodu=random.uniform(0, 1)
            jiaodu=jiaodu*2-1    #print(jiaodu)
            alpha = jiaodu * np.pi   # 角度
            if (gs['position'][0]+walkspeed*self.t*math.cos(alpha) < self.ground_length) and (gs['position'][0]+walkspeed*self.t*math.cos(alpha) > 0):
                gs['position'][0]+=walkspeed*self.t*math.cos(alpha)
            if (gs['position'][1]+walkspeed*self.t*math.sin(alpha) < self.ground_width) and (gs['position'][1]+walkspeed*self.t*math.sin(alpha) > 0):
                gs['position'][1]+=walkspeed*self.t*math.sin(alpha)
        if k==0:
            loc_uav_array=np.array(self.loc_uav)
            loc_gt_array = np.array([gt['position'] for gt in self.ground_stations])
            angles = []
            for coords in loc_gt_array:
                angle = angle_from_uav(loc_uav_array, coords)
                angles.append(angle)

            # 找到最大和最小角度以及它们的差
            max_angle = max(angles)
            min_angle = min(angles)
            angle_difference = max_angle - min_angle
            x_hm= action[self.num_ground_stations]+1#差异1
            theta = x_hm/2*angle_difference+min_angle# 角度
        elif k==1:
            x_hm= action[self.num_ground_stations]+1
            theta = x_hm*np.pi# 角度
        vel_horizontal= (action[self.num_ground_stations+1]+1)/2   
        vel_vertical= action[self.num_ground_stations+2]

        # 飞行距离和位置的计算 
        self.distance_vertical=vel_vertical*self.vel_v_max*self.t
        self.distance_horizontal=vel_horizontal*self.vel_h_max*self.t
        dx_uav = self.distance_horizontal * math.cos(theta)
        dy_uav = self.distance_horizontal * math.sin(theta) 
        dz_uav = self.distance_vertical


       
        
        if self.loc_uav[0] + dx_uav<=self.ground_length and self.loc_uav[0] + dx_uav  >=0:
            self.loc_uav[0] = self.loc_uav[0] + dx_uav
        if self.loc_uav[1] + dy_uav<=self.ground_width and self.loc_uav[1] + dy_uav  >=0:
            self.loc_uav[1] = self.loc_uav[1] + dy_uav
        if self.loc_uav[2] + dz_uav<=self.height_max and self.loc_uav[2] + dz_uav >=self.height_min:
            self.loc_uav[2] = self.loc_uav[2] + dz_uav

        
        loc_uav_array=np.array(self.loc_uav)
        loc_gt_array = np.array([gt['position'] for gt in self.ground_stations])
        distances_UAV_GT_horizontal = np.zeros(loc_gt_array.shape[0])
        
        sum=0
        bandwidth=np.zeros(loc_gt_array.shape[0])
        #修正带宽分配的参数，应为和为1
        for i in range(loc_gt_array.shape[0]):
            bandwidth[i]=(action[i]+1)/2
        for i in range(loc_gt_array.shape[0]):
            sum=sum+bandwidth[i]
        for i in range(loc_gt_array.shape[0]):
            if sum==0:
                bandwidth[i]=bandwidth[i]/1
            else:
                bandwidth[i]=bandwidth[i]/sum
        for i in range(loc_gt_array.shape[0]):
            gaussian_matrix = np.random.normal(0, 1)
            distances_UAV_GT_horizontal[i] = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[i, :2]+gaussian_matrix) # 假设loc_gt_array[i]包含x和y坐标
            r_kn = r_kn + bandwidth[i]*self.communication(self.loc_uav[2],distances_UAV_GT_horizontal[i])
        
        self.E_uav_cost = self.flight_energy_slot(abs(vel_horizontal*self.vel_h_max),abs(vel_vertical*self.vel_v_max))
        
        self.e_battery_uav =self.e_battery_uav-self.E_uav_cost
        reward= (r_kn/(10**6))/(self.E_uav_cost/self.flight_energy_slot(10.4,0))

        next_state = self._get_state()
        
        throughput=r_kn*self.t
        Energy_efficiency=throughput/self.E_uav_cost
        #print(throughput)
        ##print(r_kn)
        #print(Energy_efficiency)
        return next_state, reward, flag,throughput/(10**6),r_kn/(10**6),Energy_efficiency/(10**3), {}

    def reset(self,k):#ok
        self.e_battery_uav = 58280/2
        if k==1:
            self.loc_uav = [150, 150, 150]
        else:
            self.loc_uav = [150, 150, 150]
        self.ground_stations = []
        self.index=np.ones(self.num_ground_stations)
        gtx=[83.7783676828072,195.01460128930486,197.22939424300583,156.77595419183504,47.787914691092716,72.34721215639041]
        gty=[19.625158188947633, 128.7265491397583, 5.549427877581148, 184.97725028744216, 82.33277292670245, 117.37779460371279]
        for i in range(self.num_ground_stations):
            x= random.uniform(0, 0.5*self.ground_length)
            y = random.uniform(0, 0.5*self.ground_width)
            '''x=gtx[i]/2
            y=gty[i]/2'''
            z = 0
            self.ground_stations.append({'position': np.array([x, y, z])})
        return self._get_state()

    def _get_state(self):
        uav_normalized = np.array([self.loc_uav[0] / self.ground_length, 
                           self.loc_uav[1] / self.ground_width, 
                           self.loc_uav[2] / self.height_max])
        gs_positions_x = np.array([gs['position'][0] / self.ground_length for gs in self.ground_stations])
        gs_positions_y = np.array([gs['position'][1] / self.ground_width for gs in self.ground_stations])
        gs_positions_z = np.array([gs['position'][2] / self.height_max for gs in self.ground_stations])

        state = np.concatenate((uav_normalized, gs_positions_x, gs_positions_y,gs_positions_z))
        return state

    def flight_energy_slot(self, vel,vel_v):
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
        P0=(delta/8)*rho*s*G*(omega**3)*(R**3)
        P1=(1+k)*(W**(3/2)/math.sqrt(2*rho*G))
        P2 = 11.46
        if (1 + vel**4 / (4 * v_o ** 4))<=0 or (math.sqrt(1 + vel**4 / (4 * v_o ** 4)) - vel**2 / (2 * v_o ** 2))<=0:
            Energy_uav =self.t * (P0 * (1 + 3 * vel**2 / U_tip ** 2)+(1 / 2) * d_o * rho * s * G * vel**3 + P1 * math.sqrt(abs(math.sqrt(abs(1 + vel**4 / (4 * v_o ** 4))) - vel**2 / (2 * v_o ** 2)))+P2 * vel_v)
        else:
            Energy_uav =self.t * (P0 * (1 + 3 * vel**2 / U_tip ** 2)+(1 / 2) * d_o * rho * s * G * vel**3 + P1 * math.sqrt(math.sqrt(1 + vel**4 / (4 * v_o ** 4)) - vel**2 / (2 * v_o ** 2))+P2 * vel_v)
        return Energy_uav

    def communication(self,h,d):
            a=9.61
            b=0.16
            c = 3 * 10**8  
            n_los=1
            n_nlos=20
            A = n_los - n_nlos
            P_k=0.005 #单位W
            N_0 = 10**(-169/10) #-169dBm/Hz
            B=self.B
            f_c=9*10**8
            C = 20 * math.log10(4 * math.pi * f_c / c) + n_nlos
            p_kn = 1 + a * math.exp(a * b - b * math.atan(h / d) / math.pi * 180)
            p_kn = 1 / p_kn
            l_kn = 20 * math.log10(math.sqrt(h**2 + d**2)) + A * p_kn + C  
            r_kn = B * math.log2(1 + P_k * 10**(-l_kn / 10) / (B * N_0))       
            return r_kn

    '''def baseline(self,state):
        action = [0 for i in range(self.num_ground_stations+3)]
        uav_x=state[0]*self.ground_length
        uav_y=state[1]*self.ground_length   # 获取无人机当前位置
        loc_uav_array=[uav_x,uav_y,100]
        nearest_gs_index = -1  # 最近的地面基站索引
        min_distance = 1000
        GT_location_x = state[3:3+self.num_ground_stations]*self.ground_length
        GT_location_y = state[3+self.num_ground_stations:3+2*self.num_ground_stations]*self.ground_length
        GT_location_z = [0 for _ in range(self.num_ground_stations)]
        GT_location_x = [np.array(loc) for loc in GT_location_x]
        GT_location_y = [np.array(loc) for loc in GT_location_y]
        GT_location_z = [np.array(loc) for loc in GT_location_z]
        loc_gt_array=np.zeros([self.num_ground_stations,3])
        distances_UAV_GT_horizontal = np.zeros(self.num_ground_stations)
        
        for i in range(self.num_ground_stations):
            loc_gt_array[i]=[GT_location_x[i],GT_location_y[i],GT_location_z[i]]
        # 遍历地面基站，找到距离最近的地面基站索引
        for i in range(self.num_ground_stations):
            distances_UAV_GT_horizontal[i] = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[i, :2]) # 假设loc_gt_array[i]包含x和y坐标
            if distances_UAV_GT_horizontal[i] < min_distance:
                    min_distance = distances_UAV_GT_horizontal[i]
                    nearest_gs_index = i
        print(nearest_gs_index)

        angle = angle_from_uav(loc_uav_array[:2], loc_gt_array[nearest_gs_index, :2])
       
        action[self.num_ground_stations]=angle/np.pi-1#差异1
        action[self.num_ground_stations+1]= min(10.0, min_distance)/self.vel_h_max*2-1
        #action[self.num_ground_stations+1]=1
        #action[nearest_gs_index] = self.num_ground_stations
        action[nearest_gs_index]=1
        return action'''

    def baseline(self,state):
        action = [0 for i in range(self.num_ground_stations+3)]
        uav_x=state[0]*self.ground_length
        uav_y=state[1]*self.ground_length   # 获取无人机当前位置
        loc_uav_array=[uav_x,uav_y,100]
        nearest_gs_index = -1  # 最近的地面基站索引
        min_distance = 1000
        GT_location_x = state[3:3+self.num_ground_stations]*self.ground_length
        GT_location_y = state[3+self.num_ground_stations:3+2*self.num_ground_stations]*self.ground_length
        GT_location_z = [0 for _ in range(self.num_ground_stations)]
        GT_location_x = [np.array(loc) for loc in GT_location_x]
        GT_location_y = [np.array(loc) for loc in GT_location_y]
        GT_location_z = [np.array(loc) for loc in GT_location_z]
        loc_gt_array=np.zeros([self.num_ground_stations,3])
        distances_UAV_GT_horizontal = np.zeros(self.num_ground_stations)
        
        for i in range(self.num_ground_stations):
            loc_gt_array[i]=[GT_location_x[i],GT_location_y[i],GT_location_z[i]]
        # 遍历地面基站，找到距离最近的地面基站索引
        for i in range(self.num_ground_stations):
            distances_UAV_GT_horizontal[i] = self.index[i]*np.linalg.norm(loc_uav_array[:2] - loc_gt_array[i, :2]) # 假设loc_gt_array[i]包含x和y坐标
            if distances_UAV_GT_horizontal[i] < min_distance:
                    min_distance = distances_UAV_GT_horizontal[i]
                    nearest_gs_index = i
        print(nearest_gs_index)
        if distances_UAV_GT_horizontal[nearest_gs_index]<1:
            print(distances_UAV_GT_horizontal)
            #print(distances_UAV_GT_horizontal[nearest_gs_index])
            self.index[nearest_gs_index]=1000000
            print(self.index)
        angle = angle_from_uav(loc_uav_array[:2], loc_gt_array[nearest_gs_index, :2])
        if all(x >= 1000000 for x in self.index):
            action[self.num_ground_stations+1]=0.1
        action[self.num_ground_stations]=angle/np.pi-1#差异1
        action[self.num_ground_stations+1]= min(10.0, min_distance)/self.vel_h_max*2-1
        #action[self.num_ground_stations+1]=1
        #action[nearest_gs_index] = self.num_ground_stations
        action[nearest_gs_index]=1
        return action
def angle_from_uav(uav, gt):
        # 计算从无人机到目标的向量
        vector = gt - uav
        # 使用arctan2计算角度，输入参数是(y, x)
        angle = np.arctan2(vector[1], vector[0])
        if angle < 0:
            angle += 2 * np.pi  # 转换角度范围从 [0, 2*pi]
        return angle

