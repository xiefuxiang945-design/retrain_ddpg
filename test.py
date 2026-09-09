import math
import random
import numpy as np
import gym
from UAV_env import UAVEnv
import matplotlib.pyplot as plt

seed_value = np.random.randint(0, 1000000)  # 生成一个随机种子
print("Random seed used:", seed_value)
random.seed(seed_value)
#x =np.zeros([6])
#y =np.zeros([6])
x=[[] for i in range(6) ]
y=[[] for i in range(6)]
for i in range(6):
    x[i] = random.uniform(0, 100)
    y[i] = random.uniform(0, 100)
print(x,y)

'''walkspeed=np.zeros([100])
for i in range(100):
    walkspeed[i]=random.uniform(2, 10)
np.savetxt('walkspeed.txt', walkspeed, fmt='%.2f')
'''
#jiaodu=random.uniform(0, 1)
'''100 1 7193013.7381424100 11 7160988.0976577100 21 7077655.2485643100 31 6947160.4709257100 41 6775010.5993994100 51 6566620.3265722100 61 6325969.2661284100 71 6054724.8600145100 81 5752084.9553046100 91 5415460.287009100 101 5041955.4648047100 111 4630409.6029941100 121 4183534.8452844100 131 3709500.1431971100 141 3222260.3443988100 151 2740159.1251042100 161 2282902.3995157100 171 1867760.9951137100 181 1506384.5182727100 191 1203410.234644100 201 957133.8014564100 211 761528.3559854100 221 608537.0707676100 231 489869.5640698100 241 398057.1807749100 251 326883.0229831100 261 271419.7355402100 271 227877.6037643100 281 193392.0798021100 291 165816.7126027
110 1 6692709.4155288110 11 6666650.5339612110 21 6598668.0888072110 31 6491664.7395268110 41 6349660.8908818110 51 6176911.9296693110 61 5977030.0263827110 71 5752305.533725110 81 5503382.1710586110 91 5229382.0879825110 101 4928507.8792296110 111 4599065.9381067110 121 4240755.7635302110 131 3855960.4565749110 141 3450683.4407398110 151 3034757.552394110 161 2621066.2395914110 171 2223794.5597281110 181 1856112.7889754110 191 1528005.0231203110 201 1244959.2547811110 211 1007863.0728335110 221 813918.2289146110 231 658040.7573342110 241 534210.5998926110 251 436461.490365110 261 359448.8549757110 271 298679.6979308110 281 250528.9110426110 291 212147.4858131
120 1 6244049.4335426120 11 6222523.2856121120 21 6166260.4081474120 31 6077345.8337605120 41 5958758.5991917120 51 5813818.6548952120 61 5645607.1991428120 71 5456471.0271704120 81 5247704.6401275120 91 5019476.7822325120 101 4771035.6262311120 111 4501189.2643844120 121 4209013.2273748120 131 3894683.5226283120 141 3560277.3883787120 151 3210340.0091914120 161 2852009.6906102120 171 2494555.6995412120 181 2148327.1750009120 191 1823312.9573221120 201 1527689.3549595120 211 1266776.2329375120 221 1042675.3262631120 231 854595.1546664120 241 699629.4490984120 251 573670.2454219120 261 472203.1984038120 271 390865.3594524120 281 325759.7469725120 291 273581.3582428
130 1 5839140.3858172130 11 5821133.6956976130 21 5774005.2249965130 31 5699286.3487762130 41 5599217.9376968130 51 5476393.689135130 61 5333371.6447724130 71 5172318.5660475130 81 4994744.9725543130 91 4801375.4557728130 101 4592182.4722106130 111 4366593.2146508130 121 4123857.6654253130 131 3863540.3518354130 141 3586068.7749362130 151 3293241.5453369130 161 2988577.5471156130 171 2677387.3256583130 181 2366483.6977111130 191 2063527.1586588130 201 1776110.7044651130 211 1510789.8584455130 221 1272303.8158681130 231 1063178.1629453130 241 883766.3464646130 251 732642.9773806130 261 607177.100218130 271 504112.8089018130 281 420042.2606969130 291 351727.2552854
140 1 5471792.7762349140 11 5456569.3753308140 21 5416685.9039023140 31 5353289.1328877140 41 5268089.0714429140 51 5163122.8732907140 61 5040489.9943251140 71 4902097.5213299140 81 4749451.9682864140 91 4583527.2339795140 101 4404729.6839346140 111 4212971.4132989140 121 4007851.6747967140 131 3788933.638531140 141 3556088.5757654140 151 3309862.6890108140 161 3051805.55045140 171 2784688.4094961140 181 2512542.5907672140 191 2240469.9823512140 201 1974222.2187692140 211 1719606.1084861140 221 1481831.2384171140 231 1264945.002199140 241 1071480.8036386140 251 902379.8254131140 261 757163.8939902140 271 634273.7424011140 281 531465.7985044140 291 446178.7181028
150 1 5137051.230308150 11 5124063.4870703150 21 5090012.8398399150 31 5035772.7721336150 41 4962665.7319051150 51 4872303.1275003150 61 4766401.6369855150 71 4646599.8793673150 81 4514298.6749287150 91 4370544.7731039150 101 4215973.1170124150 111 4050817.1458095150 121 3874990.5290436150 131 3688236.9035734150 141 3490336.297203150 151 3281347.820857150 161 3061858.3935847150 171 2833198.3283881150 181 2597579.4456732150 191 2358113.845422150 201 2118685.1747547150 211 1883670.4241003150 221 1657545.3564773150 231 1444440.9018859150 241 1247737.6427922150 251 1069779.8447468150 261 911758.4346829150 271 773765.7878727150 281 654983.0352662150 291 553937.9455166
160 1 4830877.1493563160 11 4819709.3326089160 21 4790415.3969195160 31 4743670.6941817160 41 4680510.547782160 51 4602219.4936152160 61 4510201.7418726160 71 4405848.0822638160 81 4290414.3749054160 91 4164925.069621160 101 4030112.4843391160 111 3886399.3204843160 121 3733928.3344096160 131 3572639.2194732160 141 3402388.4261096160 151 3223102.6831325160 161 3034951.3935745160 171 2838517.2940778160 181 2634939.8375581160 191 2426003.3954181160 201 2214144.694703160 211 2002362.7481931160 221 1794030.3668878160 231 1592627.0646279160 241 1401433.5926784160 251 1223241.311992160 261 1060128.9661572160 271 913343.6591312160 281 783296.907373160 291 669659.9618967
170 1 4549928.1470987170 11 4540259.0574857170 21 4514887.8613762170 31 4474343.6747093170 41 4419446.1312418170 51 4351227.235822170 61 4270838.5731796170 71 4179453.7231872170 81 4078175.93023170 91 3967960.2244627170 101 3849557.6394096170 111 3723487.2174148170 121 3590039.3330399170 131 3449311.5550858170 141 3301275.7622288170 151 3145872.4225869170 161 2983124.7726461170 171 2813262.1591826170 181 2636838.3761337170 191 2454828.1211664170 201 2268683.7329533170 211 2080336.3253823170 221 1892131.2492705170 231 1706697.6151719170 241 1526764.146692170 251 1354946.0516084170 261 1193536.0166593170 271 1044333.3864829170 281 908537.8265721170 291 786719.3132084
180 1 4291400.9274651180 11 4282978.560976180 21 4260874.1547018180 31 4225507.1551568180 41 4177533.0594494180 51 4117787.3215515180 61 4047217.9068759180 71 3966812.9999743180 81 3877530.6444189180 91 3780236.6850705180 101 3675656.4848895180 111 3564344.6927748180 121 3446675.9860558180 131 3322858.2708893180 141 3192968.2977117180 151 3057007.9910862180 161 2914977.946393180 171 2766962.4932016180 181 2613218.5537215180 191 2454258.4839601180 201 2290915.6187975180 211 2124380.9602532180 221 1956201.0083686180 231 1788230.6156101180 241 1622540.9447886180 251 1461290.3594345180 261 1306573.7617925180 271 1160271.361159180 281 1023919.1284353180 291 898619.3815897
190 1 4052916.9089127190 11 4045540.8622633190 21 4026180.537264190 31 3995171.9502528190 41 3953044.1906978190 51 3900478.525854190 61 3838258.7106221190 71 3767216.8780432190 81 3688179.6560674190 91 3601918.9758578190 101 3509111.5157203190 111 3410309.9822852190 121 3305928.5649962190 131 3196243.9720679190 141 3081412.4860464190 151 2961502.4512348190 161 2836540.4910809190 171 2706568.5238461190 181 2571707.3085004190 191 2432220.8895915190 201 2288575.0985504190 211 2141482.4959152190 221 1991926.1710624190 231 1841156.0230626190 241 1690653.7704892190 251 1542066.9311797190 261 1397116.8974674190 271 1257491.0735235190 281 1124732.613181190 291 1000142.4546553
4.376583037478579 3.817671319116424 0.5589117183621544
''''''
a = 9.61
b = 0.16
c = 3 * 10**8  
n_los = 1
n_nlos = 20
A = n_los - n_nlos
P_k = 0.005 # 单位W
N_0 = 10**(-169/10) #-169dBm/Hz

B = 2 * 10 ** 6
f_c=9*10**8
C = 20 * math.log10(4 * math.pi * f_c / c) + n_nlos

heights = range(100, 200, 10)
distances = range(1, 300, 10)

p_kn_list = []
l_kn_list = []
r_kn_list = []

for h in heights:
    p_kn_row = []
    l_kn_row = []
    r_kn_row = []
    for d in distances:
        p_kn = 1 + a * math.exp(a * b - b * math.atan(h / d) / math.pi * 180)
        p_kn = 1 / p_kn
        l_kn = 20 * math.log10(math.sqrt(h**2 + d**2)) + A * p_kn + C  
        r_kn = B * math.log2(1 + P_k * 10**(-l_kn / 10) / (B * N_0))
        r_kn=round(r_kn, 7)
        print(h,d,r_kn,end='')
        p_kn_row.append(p_kn)
        l_kn_row.append(l_kn)
        r_kn_row.append(r_kn)
    print()
    p_kn_list.append(p_kn_row)
    l_kn_list.append(l_kn_row)
    r_kn_list.append(r_kn_row)



# 转换为numpy数组以便于绘图
p_kn_array = np.array(p_kn_list)
l_kn_array = np.array(l_kn_list)
r_kn_array = np.array(r_kn_list)
'''
'''
fig, ax = plt.subplots()

# Hide axes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False) 
ax.set_frame_on(False)

# Create table
table = ax.table(cellText=r_kn_array, loc='center', cellLoc='center', colLabels=['1', '11','21','31','41','51','61','71','81','91','101','111','121','131','141','151','161','171','181','191'])

# Adjust layout
table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(12)

plt.show()
'''
'''

# 绘制p_kn
plt.figure(figsize=(12, 6))
plt.imshow(p_kn_array, extent=[1, 200, 100, 200], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='p_kn')
plt.xlabel('Distance (d)')
plt.ylabel('Height (h)')
plt.title('p_kn vs Distance and Height')
plt.show()

# 绘制l_kn
plt.figure(figsize=(12, 6))
plt.imshow(l_kn_array, extent=[1, 200, 100, 200], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='l_kn')
plt.xlabel('Distance (d)')
plt.ylabel('Height (h)')
plt.title('l_kn vs Distance and Height')
plt.show()

# 绘制r_kn
plt.figure(figsize=(12, 6))
plt.imshow(r_kn_array, extent=[1, 200, 100, 200], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='r_kn')
plt.xlabel('Distance (d)')
plt.ylabel('Height (h)')
plt.title('r_kn vs Distance and Height')
plt.show()



'''

#walkspeed=2#从r_kn的角度来看，UAV到gt的水平距离越远越好，但是垂直距离越近越好
'''for vel in range(0,200,5):
            
            vel=vel/10
        
            vel_v=0/10    
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

            Energy_uav =0.5 * (P0 * (1 + 3 * vel**2 / U_tip ** 2)+(1 / 2) * d_o * rho * s * G * vel**3 + P1 * math.sqrt(
                    math.sqrt(1 + vel**4 / (4 * v_o ** 4)) - vel**2 / (2 * v_o ** 2))+P2 * vel_v)
            
            print(vel,vel_v,Energy_uav,end =" ")
            print()
'''

'''
gaussian_matrix = np.random.normal(0, 1)
print(gaussian_matrix)
loc_uav=[100,100,150]
loc_uav_array=np.array(loc_uav)
loc_gt=[32,157,0]
loc_gt_array=np.array(loc_gt)
distances_UAV_GT_horizontal = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[:2] + gaussian_matrix) 
distances_UAV_GT_horizontalx = np.linalg.norm(loc_uav_array[:2] - loc_gt_array[:2])+gaussian_matrix 
print(distances_UAV_GT_horizontal)
print(distances_UAV_GT_horizontalx)
'''
'''
ground_stations=[]
for i in range(6):
    x = np.random.rand()*200
    y = np.random.rand()*200
    z = 0
    ground_stations.append({'position': np.array([x, y, z])})
print(ground_stations)
for gs in ground_stations:
    jiaodu=np.random.rand()
    #print(jiaodu)
    jiaodu=jiaodu*2-1
    #print(jiaodu)
    alpha = jiaodu * np.pi   # 角度
    print(walkspeed*2*math.cos(alpha))
    print(walkspeed*2*math.sin(alpha))
    gs['position'][0]+=walkspeed*2*math.cos(alpha)
    gs['position'][1]+=walkspeed*2*math.sin(alpha)
print(ground_stations)

'''


'''

x 
r=200
y
r=200
z
r=500

如果超出限制
r1=-2000

r/E=300/2=150

reward大概是惩罚的0.5

'''

'''
import matplotlib.pyplot as plt
import os

# 假设你的坐标数据存储在这三个数组中
UAV_trajectory_x = [1, 2, 3]
UAV_trajectory_y = [4, 5, 6]
UAV_trajectory_z = [7, 8, 9]

# 指定保存图像的文件夹路径
save_folder = 'path/to/your/folder'
# 确保文件夹存在，不存在则创建
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 指定图像文件名
file_name = 'UAV_trajectory.png'
# 构建完整的文件路径
file_path = os.path.join(save_folder, file_name)

# 创建一个三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制轨迹
ax.plot(UAV_trajectory_x, UAV_trajectory_y, UAV_trajectory_z, label='UAV Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# 保存图像到指定路径
plt.savefig(file_path)

# 显示图像（可选）
plt.show()
'''
'''
action = [1,4,6,5,2,9,7,8,3]
sorted_actionspace=action[:6]
sorted_indices = np.argsort(sorted_actionspace)
print(sorted_actionspace)
print(sorted_indices)
print(type(sorted_indices))

print(sorted_indices[5])


sorted_action=action
for i in range(6):
            
            if i==0:
             j=sorted_indices[i]
             sorted_action[j]=0.76
            elif i <=2 :
                j=sorted_indices[i]
                sorted_action[j]=0.32/(2)**(i)
            else: 
                j=sorted_indices[i]
                sorted_action[j]=0

print(sorted_action)'''

'''for i in range(100,300,1):
    reward=(225-i)/90+(50/i)
    print(i,reward)'''
'''def angle_from_uav(uav, gt):
        # 计算从无人机到目标的向量
        vector = gt - uav
        # 使用arctan2计算角度，输入参数是(y, x)
        angle = np.arctan2(vector[1], vector[0])
        if angle < 0:
            angle += 2 * np.pi  # 转换角度范围从 [0, 2*pi]
        return angle

num_ground_stations=6
e_battery_uav = 58280/2
#self.loc_uav = [x, y, z]
loc_uav = [150, 150, 180]
ground_stations = []
for i in range(num_ground_stations):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            z = 0
            ground_stations.append({'position': np.array([x, y, z])})

loc_uav_array=np.array(loc_uav)
loc_gt_array = np.array([gt['position'] for gt in ground_stations])
angles = []
for coords in loc_gt_array:
    angle = angle_from_uav(loc_uav_array, coords)
    angles.append(angle)

# 找到最大和最小角度以及它们的差
max_angle = max(angles)
min_angle = min(angles)
angle_difference = max_angle - min_angle
        
'''
'''x_hm= action[self.num_ground_stations]+1#差异1
theta =  x_hm * np.pi /4 +np.pi  # 角度
action=[1,1,1,1,1,1,1,1,-1]
x_hm= action[num_ground_stations]+1#差异1
theta = x_hm/2*angle_difference+min_angle# 角度
action[num_ground_stations]=theta/(np.pi)-1
vel_horizontal= (action[num_ground_stations+1]+1)/2   
vel_vertical= action[num_ground_stations+2]
print(max_angle,min_angle,angle_difference)
print(action)
'''