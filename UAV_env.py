import math
import numpy as np
import gym
import random
from gym import spaces
#seed_value = np.random.randint(0, 1000000)  # 生成一个随机种子
#print("Random seed used:", seed_value)
random.seed(958030)

class UAVEnv(gym.Env):
    def __init__(self, num_ground_stations, ground_length, ground_width, height):
        super(UAVEnv, self).__init__()
        """
        环境参数
        """
        # 场地长宽均为100m，
        self.ground_length = ground_length 
        self.ground_width = ground_width  
        # UAV飞行高度
        self.height_max = height
        self.height_min = 100
        self.height = height
        self.num_ground_stations=num_ground_stations #GT数
        """
        通信参数
        """
        bandwidth_nums = 2# 带宽数目
        self.B = bandwidth_nums * 10 ** 6  # 带宽2MHz
        """
        运行参数
        """
        self.t=0.5

        self.vel_h_max = 10  # 水平最大飞行速度50m/s
        self.vel_v_max = 10  # 垂直飞行速度50m/s 

        self.e_battery_uav = 58280/2 # 80个numstep的全部耗能量
        # loc_uav表示UAV的位置
        x = np.random.rand()*self.ground_length
        y = np.random.rand()*self.ground_width
        z = 200
        #self.loc_uav = [x, y, z]
        self.loc_uav = [0, 0, 200]
        #################### UE用户 ####################
        
        self.ground_stations = []
        for i in range(num_ground_stations):
            x = random.uniform(0, self.ground_length)
            y = random.uniform(0, self.ground_width)
            z = 0
            self.ground_stations.append({'position': np.array([x, y, z])})

    def step(self, action,k):  
        is_terminal = False
        r_kn=0
        reward=0
        loc_gt = self.ground_stations
        r1=0
        walkspeed=random.uniform(0, 4)
        for gs in self.ground_stations:
            jiaodu=random.uniform(0, 1)
            jiaodu=jiaodu*2    #print(jiaodu)
            alpha = jiaodu * np.pi   # 角度
            if (gs['position'][0]+walkspeed*self.t*math.cos(alpha) < self.ground_length) and (gs['position'][0]+walkspeed*self.t*math.cos(alpha) > 0):
                gs['position'][0]+=walkspeed*self.t*math.cos(alpha)
            if (gs['position'][1]+walkspeed*self.t*math.sin(alpha) < self.ground_width) and (gs['position'][1]+walkspeed*self.t*math.sin(alpha) > 0):
                gs['position'][1]+=walkspeed*self.t*math.sin(alpha)
        
        loc_uav_array=np.array(self.loc_uav)
        loc_gt_array = np.array([gt['position'] for gt in loc_gt])
        if k==0:
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
            action[self.num_ground_stations]=theta/(np.pi)-1
        else:
            x_hm= action[self.num_ground_stations]+1
            theta =  x_hm * np.pi  # 角度


        vel_horizontal= (action[self.num_ground_stations+1]+1)/2   
        vel_vertical= action[self.num_ground_stations+2]

        # 飞行距离和位置的计算 
        self.distance_vertical=vel_vertical*self.vel_v_max*self.t
        self.distance_horizontal=vel_horizontal*self.vel_h_max*self.t
        dx_uav = self.distance_horizontal * math.cos(theta)
        dy_uav = self.distance_horizontal * math.sin(theta) 
        dz_uav = self.distance_vertical
        self.loc_uav[0] = self.loc_uav[0] + dx_uav
        self.loc_uav[1] = self.loc_uav[1] + dy_uav
        self.loc_uav[2] = self.loc_uav[2] + dz_uav  
        
        #位置控制函数，更倾向于在内部飞行
        if ((self.loc_uav[0] + dx_uav < self.ground_length) and (self.loc_uav[0] + dx_uav >= 0)) and ((self.loc_uav[1] + dy_uav < self.ground_width) and (self.loc_uav[1] + dy_uav >= 0)) and ((self.loc_uav[2] + dz_uav < self.height_max) and (self.loc_uav[2] + dz_uav >= self.height_min)):
            r1=0
        else:
            r1=-4

          
        # 将列表转换为 NumPy 数组
        loc_uav_array=np.array(self.loc_uav)
        loc_gt_array = np.array([gt['position'] for gt in loc_gt])
        distances_UAV_GT_horizontal = np.zeros(loc_gt_array.shape[0])
        bandwidth=np.zeros(loc_gt_array.shape[0])
        sum=0
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
            r_kn = r_kn +bandwidth[i]*self.communication(self.loc_uav[2],distances_UAV_GT_horizontal[i])
        self.E_uav_cost = self.flight_energy_slot(abs(vel_horizontal*self.vel_h_max),abs(vel_vertical*self.vel_v_max))
        
        self.e_battery_uav =self.e_battery_uav-self.E_uav_cost
        reward=reward+ (r_kn/(10**6))/(self.E_uav_cost/self.flight_energy_slot(10.4,0)) +r1*2
        #print((r_kn/(10**7))/(self.E_uav_cost/self.flight_energy_slot(10.4,0)))
        next_state = self._get_state()


        data=r_kn*self.t

        return next_state,action, reward, is_terminal,data,vel_horizontal*self.vel_h_max,vel_vertical*self.vel_v_max,{}

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
        if ((self.loc_uav[0] < self.ground_length) and (self.loc_uav[0]  >= 0)) and ((self.loc_uav[1]  < self.ground_width) and (self.loc_uav[1]  >= 0)) and ((self.loc_uav[2]  < self.height_max) and (self.loc_uav[2]  >= self.height_min)):
            r_kn=r_kn
        else:
            r_kn=0 
        return r_kn

    def reset(self):#ok
        #self.e_battery_uav = 250000/8  # uav电池电量: 500kJ

        self.e_battery_uav = 58280/2
        #self.loc_uav = [x, y, z]
        self.loc_uav = [5, 5, 190]
        self.ground_stations = []
        #183426
        #gtx=[83.7783676828072/2,195.01460128930486/2,197.22939424300583/2,156.77595419183504/2,47.787914691092716/2,72.34721215639041/2]
        #gty=[19.625158188947633/2, 128.7265491397583/2, 5.549427877581148/2, 184.97725028744216/2, 82.33277292670245/2, 117.37779460371279/2]
        #887115
        #gtx=[63.72676027,81.8112029,  22.59115723, 53.79039247, 27.13087413, 31.82783903]
        #gty=[87.9689825,2.76791846, 24.88115209 , 9.23983119, 33.27251303, 42.29242421]
        #958030,good
        gtx=[36.997030855955494, 53.19896748977524, 20.019889656911783, 13.257760934502294, 54.30245664199407, 74.13939293385138]
        gty=[43.30032996341227, 92.92182147894647, 82.01102859458813, 59.79515818116455, 30.841857576617237, 67.14503554078097]
        #946308
        #gtx=[77.96764806654085, 74.18565360280212, 48.692400122184196, 33.07706148218591, 63.633839745957474, 73.8319243334739]
        #gty=[22.416295089579584, 73.63781588916122, 11.553216744777595, 96.98676837896325, 73.86636674953102, 78.21293871678411]
        for i in range(self.num_ground_stations):
            '''x = random.uniform(0, 0.5*self.ground_length)
            y = random.uniform(0, 0.5*self.ground_width)'''
            x=gtx[i]+100
            y=gty[i]+100
            z = 0
            self.ground_stations.append({'position': np.array([x, y, z])})
        '''for i in range(self.num_ground_stations):
            x = random.uniform(100, 200)
            y = random.uniform(100, 200)
            z = 0
            self.ground_stations.append({'position': np.array([x, y, z])})
            '''
            
        return self._get_state()

    def _get_state(self):
        # 归一化无人机的位置
        uav_normalized = np.array([round(self.loc_uav[0] / self.ground_length,4), 
                           round(self.loc_uav[1] / self.ground_width, 4),
                           round(self.loc_uav[2] / self.height_max,4)])

        # 归一化地面站的x坐标
        gs_positions_x = np.array([round(gs['position'][0] / self.ground_length ,4)for gs in self.ground_stations])
        gs_positions_y = np.array([round(gs['position'][1] / self.ground_length ,4)for gs in self.ground_stations])
        gs_positions_z = np.array([round(gs['position'][2] / self.height_max ,4) for gs in self.ground_stations])
        # 合并所有的数组
        state = np.concatenate((uav_normalized,gs_positions_x, gs_positions_y,gs_positions_z,uav_normalized,uav_normalized,uav_normalized,uav_normalized,uav_normalized ))
        return state

def angle_from_uav(uav, gt):
        # 计算从无人机到目标的向量
        vector = gt - uav
        # 使用arctan2计算角度，输入参数是(y, x)
        angle = np.arctan2(vector[1], vector[0])
        if angle < 0:
            angle += 2 * np.pi  # 转换角度范围从 [0, 2*pi]
        return angle