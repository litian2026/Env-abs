import numpy as np
import math

class PathFollowing:

    def __init__(self, num_envs, path_type=1, point_num=1):
        self.path_type = path_type
        self.point_num = point_num
        self.reference_points = self.generateRefer()
        self.reference_points = self.generate_trajectory('sin', 400)
        self.index = 0
        self.max_points = 8
        self.t = np.zeros(num_envs)
        self.current = np.zeros(num_envs)
    def generateRefer(self):
        x = 350
        y = 215
        r = 5
        points = np.array([[x, y + r],
                           [x + r, y - r],
                           [x + r, y],
                           [x, y],
                           [x - r, y],
                           [x - r, y + r],
                           [x, y + r],
                           [x + r, y + r]])
        points_x = points[:, 0]
        points_y = points[:, 1]
        return points

    def getCurrentPoint(self, swarm_center, index):

        self.t += 1
        if self.current[index] != len(self.reference_points[index]) - 1:
            temp = self.reference_points[index][int(self.current[index]), :]
            diff = math.sqrt((swarm_center[0]-temp[0])**2+(swarm_center[1]-temp[1])**2)
            if diff <= 5:
                self.current[index] += 1
        #self.index = 0
        return self.reference_points[index][int(self.current[index]), :]


    def generate_trajectory(self, traj_type, total_time, dt=0.1, **kwargs):
        t = np.arange(0, total_time, dt)
        initial_point = [315, 210]
        if traj_type == 'line':

            choice = np.random.randint(0, 2)
            if choice:
                rand_x = np.random.uniform(-100, -60)
            else:
                rand_x = np.random.uniform(60, 100)
            choice = np.random.randint(0, 2)
            if choice:
                rand_y = np.random.uniform(-100, -60)
            else:
                rand_y = np.random.uniform(60, 100)


            # rand_x = 30
            # rand_y = 40
            start = np.array(kwargs.get('start', [initial_point[0]+rand_x-1, initial_point[1]+rand_y-1]))
            end = np.array(kwargs.get('end', [initial_point[0]+rand_x, initial_point[1]+rand_y]))
            traj = np.outer(t/total_time, end-start)+start
            #traj = 5*np.ones((int(total_time/dt), 2))
        elif traj_type == 'circle':
            center = np.array(kwargs.get('center', initial_point))
            radius = np.array(kwargs.get('radius', 2))
            angular_speed = np.array(kwargs.get('angular_speed', 2*np.pi/total_time))
            radius = np.random.uniform(30, 50)
            radius = 60
            x = center[0]+radius*np.sin(angular_speed*t)
            y = center[1]+radius*np.cos(angular_speed*t)
            traj = np.stack([x, y], axis=1)
        elif traj_type == 'sin':
            center = np.array(kwargs.get('center', initial_point))
            amplitude = kwargs.get('amplitude', 1.0)
            frequency = kwargs.get('frequency', 1.0)
            frequency = 1/400
            amplitude = 20
            vx = kwargs.get('vx', 1.0)
            vx = 0.1
            x = vx * t + center[0]
            y = amplitude * np.cos(2 * np.pi * frequency * t) + center[1]
            traj = np.stack([x, y], axis=1)
        elif traj_type == 'eight':
            scale = kwargs.get('scale', 2.0)  
            speed = kwargs.get('speed', 2*np.pi / total_time)  
            x = scale * np.sin(speed * t)
            y = scale * np.sin(speed * t) * np.cos(speed * t) 
            traj = np.stack([x, y], axis=1)
        elif traj_type == 'zigzag':
            amplitude = kwargs.get('amplitude', 1.0)  
            periods = kwargs.get('periods', 5)  
            direction = kwargs.get('direction', 'x')  
            sawtooth = amplitude * signal.sawtooth(2 * np.pi * periods * t / total_time)
            if direction == 'x':
                x = t / total_time * 5  
                y = sawtooth
            else:
                x = sawtooth
                y = t / total_time * 5  
            traj = np.stack([x, y], axis=1)
        else:

            raise ValueError(f"Unknown trajectory type: {traj_type}")
        return traj

    def generate_trajectory_re(self, center, total_time, index, dt=0.1, **kwargs):
        t = np.arange(0, total_time, dt)
        initial_point = np.array(center[0:2])
            # rand_x = 30
            # rand_y = 40
        if self.reference_points[index][100, 0]-center[0]>0:
            rand_x = np.random.uniform(40,50)
        else:
            rand_x = np.random.uniform(-50,-40)
        if self.reference_points[index][100, 1] -center[1]>0:
            rand_y = np.random.uniform(40, 50)
        else:
            rand_y = np.random.uniform(-50,  -40)

        start = np.array(kwargs.get('start', [initial_point[0] + rand_x - 1, initial_point[1] + rand_y - 1]))
        end = np.array(kwargs.get('end', [initial_point[0] + rand_x, initial_point[1] + rand_y]))
        traj = np.outer(t / total_time, end - start) + start
            # traj = 5*np.ones((int(total_time/dt), 2))
        self.reference_points[index] = traj
        
        return rand_y


