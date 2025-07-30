import rclpy
import cv2   
import mediapipe as mp
import time
import signal
import sys
import math
import numpy as np
from rclpy.node import Node
from media_pipe_ros2_msg.msg import HandPoint,MediaPipeHumanHand,MediaPipeHumanHandList
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


mp_drawing = mp.solutions.drawing_utils     # 绘制手部关键点
mp_hands = mp.solutions.hands               # 导入手部关键点检测模块
cap = cv2.VideoCapture(0)                   

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 30)             # 设置帧率 (FPS)

# 显示实际摄像头参数
print(f"摄像头宽度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"摄像头高度: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"摄像头帧率: {cap.get(cv2.CAP_PROP_FPS)}")


# 手势识别常量
GESTURE_CONSTANTS = {
    'INVALID_ANGLE': 65535.0,
    'THR_ANGLE_CURVE': 35.0,
    'THR_ANGLE_STRAIGHTEN': 65.0,
    'MAX_ANGLE': 180.0
}

# 手势定义
GESTURE_PATTERNS = {
    'one': [True, False, True, True, True],      # 只有食指伸直
    'two': [True, False, False, True, True],     # 食指和中指伸直
    'three': [True, False, False, False, True],  # 食指、中指、无名指伸直
    'four': [True, False, False, False, False],  # 除拇指外都伸直
    'five': [False, False, False, False, False], # 所有手指伸直
    'six': [False, True, True, True, False],     # 拇指和小指弯曲
    'seven':[False, False, True, True, True], 
    'eight':[False, False, False, True, True], 
    'nine':[False, False, False, False, True], 
    'ten': [True, True, True, True, True],  # 所有弯曲
    'good': [False, True, True, True, True],     # 拇指弯曲，其他伸直
    'fuck': [True, True, False, True, True],     # 中指弯曲
    'ok': [True, True, False, False, False],      # 拇指和食指弯曲，其他伸直
    'crazy':[False, False, True, True, False],
}



class HandsPublisher(Node):

    def __init__(self):
        super().__init__('mediapipe_publisher')
        self.publisher_ = self.create_publisher(MediaPipeHumanHandList, '/mediapipe/human_hand_list', 10)
        # 添加图像发布器
        self.image_publisher_ = self.create_publisher(Image, '/mediapipe/hands_image', 10)
        self.bridge = CvBridge()
    
    
        # 这个方法用来计算两个二维向量之间的角度
    def vector_2d_angle(self, v1, v2):  
        # 分别提取两个向量的x和y分量  
        v1_x, v1_y = v1[0], v1[1]  
        v2_x, v2_y = v2[0], v2[1]  
        
        # 计算向量的模长
        v1_norm = math.sqrt(v1_x ** 2 + v1_y ** 2)
        v2_norm = math.sqrt(v2_x ** 2 + v2_y ** 2)
        
        # 检查向量是否为零向量
        if v1_norm == 0 or v2_norm == 0:
            return GESTURE_CONSTANTS['INVALID_ANGLE']
        
        try:  
            # 计算点积
            dot_product = v1_x * v2_x + v1_y * v2_y
            # 计算夹角的余弦值
            cos_angle = dot_product / (v1_norm * v2_norm)
            # 限制余弦值在[-1, 1]范围内
            cos_angle = max(-1.0, min(1.0, cos_angle))
            # 使用反余弦函数计算角度，并将结果从弧度转换为角度  
            angle_ = math.degrees(math.acos(cos_angle))
        except (ValueError, ZeroDivisionError):  
            return GESTURE_CONSTANTS['INVALID_ANGLE']
        
        # 如果计算出的角度大于180度，设置为特殊值  
        if angle_ > GESTURE_CONSTANTS['MAX_ANGLE']:  
            angle_ = GESTURE_CONSTANTS['INVALID_ANGLE']  
        return angle_
    
    def hand_angle(self, hand_landmarks):
        '''
        获取对应手相关向量的二维角度,根据角度确定手势
        '''
        if not hand_landmarks or not hasattr(hand_landmarks, 'landmark') or len(hand_landmarks.landmark) < 21:
            return [GESTURE_CONSTANTS['INVALID_ANGLE']] * 5
        
        # 提取关键点坐标
        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        
        # 定义手指关键点索引
        finger_indices = {
            'thumb': [0, 2, 3, 4],      # 拇指
            'index': [0, 6, 7, 8],       # 食指
            'middle': [0, 10, 11, 12],   # 中指
            'ring': [0, 14, 15, 16],     # 无名指
            'pinky': [0, 18, 19, 20]     # 小指
        }
        
        angle_list = []
        
        for finger_name, indices in finger_indices.items():
            # 计算手指角度
            v1 = landmarks[indices[0]] - landmarks[indices[1]]  # 手掌到第二关节
            v2 = landmarks[indices[2]] - landmarks[indices[3]]  # 第二关节到指尖
            
            angle_ = self.vector_2d_angle(v1, v2)
            angle_list.append(angle_)
        
        return angle_list
    
    def detect_gesture(self, angle_list):
        '''
        二维约束的方法定义手势
        '''
        if GESTURE_CONSTANTS['INVALID_ANGLE'] in angle_list:
            return "None"
        
        # 将角度转换为布尔值（伸直/弯曲）
        finger_states = []
        for angle in angle_list:
            if angle < GESTURE_CONSTANTS['THR_ANGLE_CURVE']:
                finger_states.append(False)   # 伸直（角度小）
            elif angle > GESTURE_CONSTANTS['THR_ANGLE_STRAIGHTEN']:
                finger_states.append(True)  # 弯曲（角度大）
            else:
                finger_states.append(None)  # 默认弯曲
        
        # 模式匹配
        for gesture_name, pattern in GESTURE_PATTERNS.items():
            if finger_states == pattern:
                return gesture_name
        
        return "None"
    


    def draw_text(self, image, text, position, font_scale=1.0, text_color=(255, 255, 255), thickness=2):
        """
        在图像上绘制带背景框的文字
        Args:
            image: 要绘制的图像
            text: 要绘制的文字
            position: 文字位置 (x, y)
            font_scale: 字体大小
            text_color: 文字颜色 (B, G, R)
            bg_color: 背景颜色 (B, G, R)
            thickness: 文字粗细
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 绘制文字
        cv2.putText(image, text, position, font, font_scale, text_color, thickness)
        
        return image


    def getimage_callback(self):
        mediapipehumanlist = MediaPipeHumanHandList() 
        mediapipehuman = MediaPipeHumanHand()
        points = HandPoint()

        # 性能监控
        frame_count = 0
        start_time = time.time()
        fps = 0.0  # 初始化FPS为0
            

        with mp_hands.Hands(    # 创建手部关键点检测对象
                static_image_mode=False,        # 是否静态图像模式
                min_detection_confidence=0.7,   # 最小检测置信度
                min_tracking_confidence=0.7,    # 最小跟踪置信度
                max_num_hands=2) as hands:      # 最大手数
            
            while cap.isOpened():
                
                success, image = cap.read()
                if not success:
                    print("Sem camera.")
                    time.sleep(0.1)
                    continue

                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧计算一次FPS
                    current_time = time.time()
                    fps = 30 / (current_time - start_time)
                    start_time = current_time
                    print(f"FPS: {fps:.2f}")
                

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # 水平翻转图像后，将图像从BGR格式转换为RGB格式
                image.flags.writeable = False
                results = hands.process(image)# 处理图像，返回手部关键点
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                    # 在图像上显示FPS和状态信息
                status_text = f"FPS: {fps:.1f}"
                self.draw_text(image, status_text, (500, 30), 0.7, (255, 255, 255), 2)
                
                imageHeight, imageWidth, _ = image.shape
                
                if results.multi_hand_landmarks != None:# 如果检测到手部关键点
                    hand_number_screen = 0
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):# 遍历手部关键点
                        if handedness.classification[0].label == "Right":# 如果手部关键点是右手
                            mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)# 绘制手部关键点及连线
                            
                            # 添加手势识别
                            angle_list = self.hand_angle(hand_landmarks)
                            gesture = self.detect_gesture(angle_list)
                            print(f"右手手势: {gesture}")
                            
                            # 在图像上绘制手势文字
                            if gesture != "None":
                                text = f"Right: {gesture}"
                                self.draw_text(image, text, (50, 50), 1.0, (255, 255, 255), 2)
                            
                            index_point = 0 # 初始化索引
                            
                            for point in mp_hands.HandLandmark:# 遍历手部关键点
                                normalizedLandmark = hand_landmarks.landmark[point] # 归一化手部关键点
                                mediapipehuman.right_hand_key_points[index_point].name = str(point) # 将手部关键点名称赋值
                                mediapipehuman.right_hand_key_points[index_point].x = normalizedLandmark.x
                                mediapipehuman.right_hand_key_points[index_point].y = normalizedLandmark.y
                                mediapipehuman.right_hand_key_points[index_point].z = normalizedLandmark.z                                
                                if hand_number_screen == 0:
                                    mediapipehuman.left_hand_key_points[index_point].name = str(point) 
                                    mediapipehuman.left_hand_key_points[index_point].x = 0.0
                                    mediapipehuman.left_hand_key_points[index_point].y = 0.0
                                    mediapipehuman.left_hand_key_points[index_point].z = 0.0
                                index_point = index_point +1
                            hand_number_screen = hand_number_screen +1

                        elif handedness.classification[0].label == "Left":
                            mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            # 添加手势识别
                            angle_list = self.hand_angle(hand_landmarks)
                            gesture = self.detect_gesture(angle_list)
                            print(f"左手手势: {gesture}")
                            
                            # 在图像上绘制手势文字
                            if gesture != "None":
                                text = f"Left: {gesture}"
                                self.draw_text(image, text, (50, 100), 1.0, (255, 255, 255), 2)
                            
                            index_point = 0
                            
                            for point in mp_hands.HandLandmark:
                                
                                normalizedLandmark = hand_landmarks.landmark[point]  
                                points.name = str(point)                            
                                mediapipehuman.left_hand_key_points[index_point].name = str(point) 
                                mediapipehuman.left_hand_key_points[index_point].x = normalizedLandmark.x
                                mediapipehuman.left_hand_key_points[index_point].y = normalizedLandmark.y 
                                mediapipehuman.left_hand_key_points[index_point].z = normalizedLandmark.z
                                
                                if hand_number_screen == 0:
                                    mediapipehuman.right_hand_key_points[index_point].name = str(point) 
                                    mediapipehuman.right_hand_key_points[index_point].x = 0.0
                                    mediapipehuman.right_hand_key_points[index_point].y = 0.0
                                    mediapipehuman.right_hand_key_points[index_point].z = 0.0
                                index_point = index_point +1
                            hand_number_screen = hand_number_screen +1

                    mediapipehumanlist.human_hand_list.right_hand_key_points = mediapipehuman.right_hand_key_points
                    mediapipehumanlist.human_hand_list.left_hand_key_points = mediapipehuman.left_hand_key_points
                    mediapipehumanlist.num_humans = 1
                    self.publisher_.publish(mediapipehumanlist)
                else: # responsavel por mandar 0 nos topicos quando as duas maos nao estao na tela
                    index_point = 0
                    for point in mp_hands.HandLandmark:                          
                                                                                          
                        mediapipehuman.right_hand_key_points[index_point].name = str(point) 
                        mediapipehuman.right_hand_key_points[index_point].x = 0.0
                        mediapipehuman.right_hand_key_points[index_point].y = 0.0
                        mediapipehuman.right_hand_key_points[index_point].z = 0.0 
                        mediapipehuman.left_hand_key_points[index_point].name = str(point) 
                        mediapipehuman.left_hand_key_points[index_point].x = 0.0
                        mediapipehuman.left_hand_key_points[index_point].y = 0.0
                        mediapipehuman.left_hand_key_points[index_point].z = 0.0
                        index_point = index_point + 1 

                    mediapipehumanlist.human_hand_list.right_hand_key_points = mediapipehuman.right_hand_key_points
                    mediapipehumanlist.human_hand_list.left_hand_key_points = mediapipehuman.left_hand_key_points
                    mediapipehumanlist.num_humans = 1
                    self.publisher_.publish(mediapipehumanlist)

                

                # cv2.imshow('MediaPipe Hands', image)
                
                # 发布处理后的图像
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                    self.image_publisher_.publish(img_msg)
                except Exception as e:
                    print(f"发布图像时出错: {e}")
                
                # 添加多种退出方式
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):  # Q键退出
                    print("按Q键退出程序")
                    break
                elif key == 27:  # ESC键退出
                    print("按ESC键退出程序")
                    break
                elif key == ord('x') or key == ord('X'):  # X键退出
                    print("按X键退出程序")
                    break

def main(args=None):
    rclpy.init(args=args)

    hands_publisher = HandsPublisher()
    
    try:
        hands_publisher.getimage_callback()
    except KeyboardInterrupt:
        print("\n程序被Ctrl+C中断")
    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        # 确保资源被正确释放
        cap.release()
        cv2.destroyAllWindows()
        hands_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()