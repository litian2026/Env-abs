import cv2
import numpy as np
import math
import time

class Image_process:
    def __init__(self, t=5):
        self.t = t
        # 设置视频参数
        self.width, self.height = 640, 480  # 根据您的图像尺寸调整
        self.fps = 30.0
        self.output_filename = 'output_video.mp4'

        # 创建VideoWriter对象
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
        self.out = cv2.VideoWriter(self.output_filename, self.fourcc, self.fps, (self.width, self.height))
        self.symbol = 1

    # def ProcessRawImage(self, image):
    #
    #     if image.dtype == np.float32 or image.dtype == np.float64:
    #         image_data = (image * 255).astype(np.uint8)
    #
    #     # 转换为BGR格式,opencv中表示方法
    #     if len(image_data.shape) == 3 and image_data.shape[2] == 3:
    #         img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    #     else:
    #         print("Unexpected image format")
    #         return None, None, None
    #     # 处理黑色粒子群
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     _, binary = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
    #
    #     # 反转图像使机器人像素为白色(255)
    #     inverted = cv2.bitwise_not(gray)
    #
    #     # 找到所有机器人像素(白色)的坐标
    #     y_coords, x_coords = np.where(inverted > 0)
    #     robot_pixels = len(x_coords)
    #
    #     if robot_pixels == 0:
    #         return 0.0, (0, 0), 0.0, 0.0
    #
    #     # 计算中心位置(质心)
    #     center_x = np.mean(x_coords)
    #     center_y = np.mean(y_coords)
    #     center = (int(round(center_x)), int(round(center_y)))
    #
    #     # 计算位置方差
    #     variance_x = np.var(x_coords)
    #     variance_y = np.var(y_coords)
    #     variance = (variance_x + variance_y) / 2  # 平均方差
    #
    #     # 计算每个点到中心的距离
    #     distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    #
    #     # 计算包含85%机器人的半径
    #     sorted_distances = np.sort(distances)
    #     eighty_fifth_percentile = int(0.90 * len(sorted_distances))
    #     radius = sorted_distances[eighty_fifth_percentile]
    #
    #     # 计算密度(机器人像素数/分布面积)
    #     if radius > 0:
    #         area = math.pi * radius ** 2
    #         density = robot_pixels / area
    #     else:
    #         density = 0.0
    #
    #     return density, center, variance, radius
    def ProcessRawImage(self, obs):

        # if image_data.dtype == np.float32 or image_data.dtype == np.float64:
        #     image_data = (image_data * 255).astype(np.uint8)

        # # 转换为BGR格式,opencv中表示方法
        # if len(image_data.shape) == 3 and image_data.shape[2] == 3:
        #     img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        # else:
        #     print("Unexpected image format")
        #     return None, None, None
        # # 处理黑色粒子群
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # valid_contours = []
        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     contour_center = (x + w // 2, y + h // 2)
        #     if 50 < contour_center[1] < 600:
        #         valid_contours.append(contour)
        # if len(valid_contours) == 0:
        #     return 0.0, (-500, -500), 20.0, 20.0, (-500, 500), True
        # largest_contour = max(valid_contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # center1 = (x + w // 2, y + h // 2)

        # rotated_rect = cv2.minAreaRect(largest_contour)
        # (center_x, center_y), (w, h), angle = rotated_rect  # 直接返回中心坐标
        # center = (center_x, center_y)
        # # 反转图像使机器人像素为白色(255)
        # # inverted = cv2.bitwise_not(gray)

        # # 找到所有机器人像素(白色)的坐标
        # y_coords, x_coords = np.where(binary > 0)
        # robot_pixels = len(x_coords)

        # if robot_pixels == 0:
        #     return 0.0, (0, 0), 0.0, 0.0, True

        # # 计算中心位置(质心)
        # #center_x = np.mean(x_coords)
        # #center_y = np.mean(y_coords)
        # #center = (int(round(center_x)), int(round(center_y)))

        # # 计算位置方差
        # variance_x = np.var(x_coords)
        # variance_y = np.var(y_coords)
        # variance = (variance_x + variance_y) / 2  # 平均方差

        # # 计算每个点到中心的距离
        # distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # # 计算包含85%机器人的半径
        # sorted_distances = np.sort(distances)
        # eighty_fifth_percentile = int(0.90 * len(sorted_distances))
        # radius = sorted_distances[eighty_fifth_percentile]
        # result = image_data.copy()
        # # 绘制粒子群中心
        # cv2.circle(result, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)  # 绿色点表示中心
        # cv2.circle(result, (int(350), int(225)), 5, (255, 0, 0), -1)
        # # 计算密度(机器人像素数/分布面积)
        # if radius > 0:
        #     area = math.pi * radius ** 2
        #     density = robot_pixels / area
        # else:
        #     density = 0.0

        # 实时输出所有粒子的真实平均位置与图像识别位置的差异
        density = 1
        variance = 1
        radius = 10
        center = obs
        center1 = obs
    


        return density, center, variance, radius, center1, False

    def visualize_results(self, image_data, results, reference_points, phi, psi, flow_x, flow_y):
        #if image_data is None:
        # 如果没有输入图像，创建一个默认的黑色背景
        image_data = np.ones((480, 640, 3), dtype=np.float32)
        density, center, variance, radius, binary = results
        result = image_data.copy()

        # 绘制中心点和参考点
        cv2.circle(result, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)  # 绿点
        #cv2.circle(result, (int(binary[0]), int(binary[1])), 1, (0, 0, 255), -1)  # 蓝点
        cv2.circle(result, (350, 225), 1, (255, 0, 0), -1)  # 蓝点
        # 绘制psi的方向和水流方向
        cv2.arrowedLine(result, (int(center[0]), int(center[1])), 
                (int(center[0] + 100 * phi* math.cos(psi)), int(center[1] + 100*phi * math.sin(psi))), 
                (255, 255, 0), 2, tipLength=0.5) 
        cv2.arrowedLine(result, (int(center[0]), int(center[1])), 
                (int(center[0] + flow_x*60), int(center[1] - flow_y*60)), 
                (0, 255, 255), 2, tipLength=0.5) 
        # 绘制参考轨迹
        if reference_points is not None:
            for point in reference_points:
                x, y = int(point[0]), int(point[1])  # 确保坐标是整数
                cv2.circle(result, (x, y), 1, (0, 255, 255), -1)  # 黄色点
            cv2.circle(result, (int(reference_points[self.t][0]), int(reference_points[self.t][1])), 1, (0, 0, 255), -1)  # 黄色点
        # # 可选：绘制二值化结果（调试用）
        # #binary_rgb = cv2.cvtColor(binary[:, :, 0], cv2.COLOR_GRAY2BGR)
        combined = np.hstack([result, result])

        # 显示到独立窗口
        window_name = "OpenCV Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
        cv2.resizeWindow(window_name, 800, 600)  # 设置初始窗口大小（宽×高）
        cv2.imshow(window_name, result)
        cv2.waitKey(1)  # 非阻塞式刷新
        #time.sleep(0.1)


        return combined

if __name__ =="__main__":
    # 创建环境
    env_name = "../model_new/Josh2DEndlessShooter.exe"
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(env_name, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=1)
    env = UnityToGymWrapper(unity_env)
    # 获取图像
    IP = Image_process()
    obs = env.reset()
    density, center, variance, radius = IP.ProcessRawImage(obs)
    debug = 1
