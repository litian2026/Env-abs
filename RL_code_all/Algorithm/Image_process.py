import cv2
import numpy as np
import math
import time

class Image_process:
    def __init__(self, t=5):
        self.t = t
        self.width, self.height = 640, 480  
        self.fps = 30.0
        self.output_filename = 'output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        self.out = cv2.VideoWriter(self.output_filename, self.fourcc, self.fps, (self.width, self.height))
        self.symbol = 1

    def ProcessRawImage(self, obs):
        density = 1
        variance = 1
        radius = 10
        center = obs
        center1 = obs
        return density, center, variance, radius, center1, False

    def visualize_results(self, image_data, results, reference_points, phi, psi, flow_x, flow_y):

        image_data = np.ones((480, 640, 3), dtype=np.float32)
        density, center, variance, radius, binary = results
        result = image_data.copy()
        cv2.circle(result, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)  
        cv2.circle(result, (350, 225), 1, (255, 0, 0), -1)  
        cv2.arrowedLine(result, (int(center[0]), int(center[1])), 
                (int(center[0] + 50 * phi* math.cos(psi)), int(center[1] + 50*phi * math.sin(psi))), 
                (255, 255, 0), 2, tipLength=0.5) 
        cv2.arrowedLine(result, (int(center[0]), int(center[1])), 
                (int(center[0] + flow_x*60), int(center[1] - flow_y*60)), 
                (0, 255, 255), 2, tipLength=0.5) 
        if reference_points is not None:
            for point in reference_points:
                x, y = int(point[0]), int(point[1])  
                cv2.circle(result, (x, y), 1, (0, 255, 255), -1) 
            cv2.circle(result, (int(reference_points[self.t][0]), int(reference_points[self.t][1])), 1, (0, 0, 255), -1)  
        combined = np.hstack([result, result])
        window_name = "OpenCV Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  
        cv2.resizeWindow(window_name, 800, 600)  
        cv2.imshow(window_name, result)
        cv2.waitKey(1)  
        #time.sleep(0.1)


        return combined

