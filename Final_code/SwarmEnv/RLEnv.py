
from gymnasium import Wrapper
import gymnasium as gym
import numpy as np
from Algorithm.Image_process import Image_process
import math
from collections import deque
class CustomEnv(Wrapper):
    def __init__(self, worker_id=0):
        base_env = DynamicsEnv()
        super().__init__(base_env)
        self.IP = Image_process()
        self.current_goal = [0, 0]
        self.episode_length = 50
        self.reference_points = None
        self.t = 0
        self.cv_show = True
        self.flow_x = 0
        self.flow_y = 0
        self.previous_obs = None
        self.pos = None
        self.arrive_step = 0
        self.cur_dis = 0 
        self.pre_dis = 0
        self.flow_x_real = 0
        self.flow_y_real = 0
        self.flow_x_real1 = 0
        self.flow_y_real1 = 0
        self.theta_list = deque(maxlen=1)
        self.previous_obs_list = []
        self.previous_obs = None
        self.v_max = 0
        self.flow_x_real_cp = 0 
        self.flow_y_real_cp = 0

    def step(self, action):

        # 解包复合动作
        ptr = 0
        action_raw = action[ptr:ptr + 2]  # 前2维：原始动作
        ptr += 2
        phi, psi = action[ptr], action[ptr + 1]
        ptr += 2
        step = action[ptr:ptr + 1]
        continue_step = 0
        
        arrive_step = 0
        self.flow_x_real = self.flow_x_real_cp
        self.flow_y_real = self. flow_y_real_cp
        # self.flow_x_real += np.random.normal(-0.2, 0.2)
        # self.flow_y_real += np.random.normal(-0.2, 0.2)
        for index in range(1):
            phi = np.clip(phi, 0, self.v_max)
            obs, reward, done, info = self.env.step(np.array([phi, psi, self.flow_x_real, self.flow_y_real]))
        density, center, variance, radius, res, out_view = self.IP.ProcessRawImage(obs) # 从图像中获取状态
        results = [density, center, variance, radius, res]
        self.IP.t = self.t
        if self.cv_show:
            visualized_image = self.IP.visualize_results(obs, results, self.reference_points, phi, psi, self.flow_x_real, self.flow_y_real)
        center = np.append(center, 0)
        self.pos = center[:2]


        # 合成观测量
        def clamp_angle_diff(a, b):
            """调整角度a，使其与b的差值不超过π"""
            diff = a - b
            if diff > np.pi:
                a -= 2 * np.pi  # 逆时针方向调整
            elif diff < -np.pi:
                a += 2 * np.pi  # 顺时针方向调整
            return a
        desire_goal = math.atan2((self.current_goal[1] - center[1]), (self.current_goal[0] - center[0]))
        desire_goal = clamp_angle_diff(desire_goal, psi)
        current_dis = math.sqrt(((self.current_goal[0] - center[0]))**2+((self.current_goal[1] - center[1]))**2)
        vector1 = np.array([self.current_goal[0]-center[0], self.current_goal[1]-center[1]])
        x2target = vector1[0]
        y2target = vector1[1]
        self.flow_x_real1 = 10*self.flow_x_real
        self.flow_y_real1 = 10*self.flow_y_real
        obs_new = np.concatenate([[x2target, y2target], [phi], [math.sin(psi), math.cos(psi)],
                                    [self.flow_x_real1, self.flow_y_real1, self.v_max]])
        

        # 奖励计算
        reward_f= 0
        # self.previous_obs_list.append(obs_new)
       
        # if len(self.previous_obs_list)>=2:
        #     self.previous_obs = self.previous_obs_list[-2]
        if step == 1:
            x_move = -obs_new[0] - self.previous_obs[0] + self.current_goal[0]
            y_move = -obs_new[1] - self.previous_obs[1] +self.current_goal[1]
        else:
            x_move = -obs_new[0] + self.previous_obs[0]
            y_move = -obs_new[1] + self.previous_obs[1]
        v1 = [x_move, y_move]
        v2 = [obs_new[0], obs_new[1]]
        theta_v12 = self.angle_between_vectors(v1, v2)
        if np.isnan(theta_v12):
            theta_v12 = 0
        # 计算当前距离和上一个周期的距离
        self.cur_dis = math.sqrt(obs_new[1]*obs_new[1]+obs_new[0]*obs_new[0])
        #self.pre_dis = math.sqrt(self.previous_obs[1]*self.previous_obs[1]+self.previous_obs[0]*self.previous_obs[0])
        reward_dis = -0.005 * (self.cur_dis)
        # 计算执行动作后的运动方向与期望运动方向的差异，差异越小，奖励越大
        self.theta_list.append(theta_v12)
        avr_theta_v12 = sum(self.theta_list)/len(self.theta_list)
        reward_heading = (-1.0 *avr_theta_v12/3*min(math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]),2))
        reward_heading = max(reward_heading, -5)
        reward_path_following = reward_dis+reward_heading

        self.previous_obs = obs_new
        print(f"theta:{avr_theta_v12*180/math.pi}")
      
        reward = 0       
        reward += reward_path_following

        # episode 结束标志
        flow_x_rec = self.flow_x_real
        flow_y_rec = self.flow_y_real
        if self.cur_dis < 5:
            self.arrive_step = self.arrive_step + 1
        else:
            self.arrive_step = 0
        if self.arrive_step > 15:
            done = False
            self.arrive_step = 0
        if step == self.episode_length-1:
            done = True
        if out_view:
            done = True
        info = {
            "continue_step": continue_step,  # 存储自定义值
            "arrive_step": arrive_step,
            "pos": self.pos,
            "dones": done,
            "dis": self.cur_dis,
            "reward_flocking": 0,
            "reward_dis": reward_dis,
            "reward_heading": reward_heading,
            "reward_final": 0,
            "flow":self.flow_x,
            "flow_x": flow_x_rec,
            "flow_y": flow_y_rec
        }
        truncated = False    # 时间超限等
        return obs_new, reward, done, truncated, info

    def reset(self,seed=None, options=None):
        obs = self.env.reset()
        self.previous_obs_list=[]
        density, center, variance, radius, res, _ = self.IP.ProcessRawImage(obs)
        # 根据不同的需求初始化流速，模式0顺流，模式1完全逆流， 模式2不完全逆流
        mode = 1
        x_sym = self.reference_points[0][0] - center[0]
        y_sym = self.reference_points[0][1] - center[1]
        x_sign = np.sign(x_sym)
        y_sign = np.sign(y_sym)
        direction = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]])
        random_flow = np.random.uniform(0, self.flow_x)
        random_flow = self.flow_x
        mode = np.random.randint(0, 3)
        #mode =0
        if mode == 0 or mode == 1:
            self.flow_x_real = x_sign*direction[mode, 0]*np.random.uniform(0, random_flow)
            self.flow_y_real = -y_sign*direction[mode, 1]*np.sqrt(random_flow*random_flow - self.flow_x_real*self.flow_x_real)
        else:
            minusorplus = np.random.uniform(0, 1)
            if minusorplus>0.5:
                self.flow_x_real = x_sign*direction[2, 0]*np.random.uniform(0, random_flow)
                self.flow_y_real = -y_sign*direction[2, 1]*np.sqrt(random_flow*random_flow -self.flow_x_real*self.flow_x_real)
            else:
                self.flow_x_real = x_sign*direction[3, 0]*np.random.uniform(0, random_flow)
                self.flow_y_real = -y_sign*direction[3, 1]*np.sqrt(random_flow*random_flow - self.flow_x_real*self.flow_x_real)
    
        # 重置其他值
        # self.flow_x_real = 0
        # self.flow_y_real = 1
        self.pos = center
        center = np.append(center, 0)
        obs_new = np.array([
            center[0], center[1],
            0,
            math.sin(0), math.cos(0),
            self.flow_x_real1, self.flow_y_real1,
            self.v_max
        ])
        info = {}
        self.flow_x_real_cp = self.flow_x_real
        self.flow_y_real_cp = self. flow_y_real
        self.previous_obs = obs_new
        return obs_new, info

    def angle_between_vectors(self, v1, v2):
        """
        计算两个向量之间的夹角（0到pi之间）

        参数:
        v1, v2 -- 输入向量（列表或numpy数组）

        返回:
        夹角（弧度制）
        """
        v1 = np.array(v1)
        v2 = np.array(v2)

        # 计算向量的点积
        dot_product = np.dot(v1, v2)

        # 计算向量的模
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 计算夹角的余弦值
        cos_theta = dot_product / (norm_v1 * norm_v2)

        # 处理可能的浮点误差（确保cos_theta在[-1, 1]范围内）
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # 计算夹角（弧度）
        angle = np.arccos(cos_theta)

        return angle

    def set_first_flag(self, value: bool):
        self.first_flag = value

    def set_flow_x(self, value):
        self.flow_x = value

    def set_previous_obs(self, value):
        self.previous_obs = value

    def set_current_goal(self, value):
        self.current_goal = value

    def set_episode_length(self, value):
        self.episode_length = value

    def set_reference_points(self, value):
        self.reference_points = value

    def get_robot_pos(self):
        return self.pos

    def set_cv_show(self, value):
        self.cv_show = value
    def set_v_max(self, value):
        self.v_max = value


class DynamicsEnv(gym.Env):
    def __init__(self, phi=0, psi=0):
        self.v = phi
        self.psi = psi
        self.obs = np.array([316,217],dtype=np.float32)
        self.disturb = np.zeros(2)
        self.reward = 0
        self.done = False
        self.info = None
        self.action_space = None 
        self.observation_space = None 
        self.reward_range = None
        self.metadata  = None
        self.seed = None


    def step(self, action):
        # 状态转移
        self.v = action[0]
        self.psi = action[1]
        self.flow_x = action[2]
        self.flow_y = action[3]
        self.obs[0] = self.obs[0] + 10*(self.v*math.cos(self.psi) + self.flow_x/6)
        self.obs[1] = self.obs[1] + 10*(self.v*math.sin(self.psi) - self.flow_y/6)
        return self.obs, self.reward, self.done, self.info
    


    def reset(self, seed=None, options=None):
        self.phi = 0
        self.psi = 0
        self.obs = np.array([316,217],dtype=np.float32)
        return self.obs
    


