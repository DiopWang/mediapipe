import rclpy
import cv2
import math
import numpy as np  
import time
import rclpy
import mediapipe as mp
from rclpy.node import Node
from media_pipe_ros2_msg.msg import  MediaPipeHumanPoseList                            
from mediapipe.python.solutions.pose import PoseLandmark

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
NAME_POSE = [
    (PoseLandmark.NOSE), (PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE), (PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_INNER), ( PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE_OUTER), ( PoseLandmark.LEFT_EAR),
    (PoseLandmark.RIGHT_EAR), ( PoseLandmark.MOUTH_LEFT),
    (PoseLandmark.MOUTH_RIGHT), ( PoseLandmark.LEFT_SHOULDER),
    (PoseLandmark.RIGHT_SHOULDER), ( PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW), ( PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_WRIST), ( PoseLandmark.LEFT_PINKY),
    (PoseLandmark.RIGHT_PINKY), ( PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_INDEX), ( PoseLandmark.LEFT_THUMB),
    (PoseLandmark.RIGHT_THUMB), ( PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_HIP), ( PoseLandmark.LEFT_KNEE),
    (PoseLandmark.RIGHT_KNEE), ( PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE), ( PoseLandmark.LEFT_HEEL),
    (PoseLandmark.RIGHT_HEEL), ( PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_FOOT_INDEX)
]

# 姿态识别常量
POSE_CONSTANTS = {
    'INVALID_ANGLE': 65535.0,
    'THR_ANGLE_CURVE': 35.0,
    'THR_ANGLE_STRAIGHTEN': 65.0,
    'MAX_ANGLE': 180.0
}


cap = cv2.VideoCapture(0)
# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # 设置图像高度
cap.set(cv2.CAP_PROP_FPS, 30)             # 设置帧率 (FPS)
# 显示实际摄像头参数
print(f"摄像头宽度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"摄像头高度: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"摄像头帧率: {cap.get(cv2.CAP_PROP_FPS)}")


class PosePublisher(Node):

    def __init__(self):
        super().__init__('mediapipe_pose_publisher')
        self.publisher_ = self.create_publisher(MediaPipeHumanPoseList, '/mediapipe/human_pose_list', 10)
    
    
    #     def vector_2d_angle(self, v1, v2):  
    #     # 分别提取两个向量的x和y分量  
    #         v1_x, v1_y = v1[0], v1[1]  
    #         v2_x, v2_y = v2[0], v2[1]  
            
    #         # 计算向量的模长
    #         v1_norm = math.sqrt(v1_x ** 2 + v1_y ** 2)
    #         v2_norm = math.sqrt(v2_x ** 2 + v2_y ** 2)
            
    #         # 检查向量是否为零向量
    #         if v1_norm == 0 or v2_norm == 0:
    #             return POSE_CONSTANTS['INVALID_ANGLE']
            
    #         try:  
    #             # 计算点积
    #             dot_product = v1_x * v2_x + v1_y * v2_y
    #             # 计算夹角的余弦值
    #             cos_angle = dot_product / (v1_norm * v2_norm)
    #             # 限制余弦值在[-1, 1]范围内
    #             cos_angle = max(-1.0, min(1.0, cos_angle))
    #             # 使用反余弦函数计算角度，并将结果从弧度转换为角度  
    #             angle_ = math.degrees(math.acos(cos_angle))
    #         except (ValueError, ZeroDivisionError):  
    #             return POSE_CONSTANTS['INVALID_ANGLE']
            
    #         # 如果计算出的角度大于180度，设置为特殊值  
    #         if angle_ > POSE_CONSTANTS['MAX_ANGLE']:  
    #             angle_ = POSE_CONSTANTS['INVALID_ANGLE']  
    #         return angle_
    
    # def pose_angle(self, pose_landmarks):
    #     '''
    #     获取对应手相关向量的二维角度,根据角度确定手势
    #     '''
    #     if not pose_landmarks or not hasattr(pose_landmarks, 'landmark') or len(pose_landmarks.landmark) < 21:
    #         return [POSE_CONSTANTS['INVALID_ANGLE']] * 5
        
    #     # 提取关键点坐标
    #     landmarks = np.array([[lm.x, lm.y] for lm in pose_landmarks.landmark])
        
    #     # 定义手指关键点索引
    #     finger_indices = {
    #         'thumb': [0, 2, 3, 4],      # 拇指
    #         'index': [0, 6, 7, 8],       # 食指
    #         'middle': [0, 10, 11, 12],   # 中指
    #         'ring': [0, 14, 15, 16],     # 无名指
    #         'pinky': [0, 18, 19, 20]     # 小指
    #     }
        
        # def detect_pose(self, angle_list):
        #     '''
        #     二维约束的方法定义手势
        #     '''
        #     if POSE_CONSTANTS['INVALID_ANGLE'] in angle_list:
        #         return "None"
            
        #     # 将角度转换为布尔值（伸直/弯曲）
        #     finger_states = []
        #     for angle in angle_list:
        #         if angle < POSE_CONSTANTS['THR_ANGLE_CURVE']:
        #             finger_states.append(False)   # 伸直（角度小）
        #         elif angle > POSE_CONSTANTS['THR_ANGLE_STRAIGHTEN']:
        #             finger_states.append(True)  # 弯曲（角度大）
        #         else:
        #             finger_states.append(None)  # 默认弯曲
            
        #     # 模式匹配
        #     for pose_name, pattern in POSE_PATTERNS.items():
        #         if finger_states == pattern:
        #             return pose_name
            
        #     return "None"
        
    
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
        mediapipehumanposelist = MediaPipeHumanPoseList() 
        
        # 性能监控初始化
        frame_count = 0
        start_time = time.time()
        fps = 0.0

        with mp_pose.Pose(
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as pose:
            while cap.isOpened():

                success, image = cap.read()
                if not success:
                    print("Sem camera.")
                    
                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧计算一次FPS
                    current_time = time.time()
                    fps = 30 / (current_time - start_time)
                    start_time = current_time
                    print(f"FPS: {fps:.2f}")
                            
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)# 翻转图像                
                image.flags.writeable = False
                results = pose.process(image)# 处理图像
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                imageHeight, imageWidth, _ = image.shape
                
                # 在图像上显示FPS和状态信息
                status_text = f"FPS: {fps:.1f}"
                self.draw_text(image, status_text, (10, 30), 0.7, (255, 255, 255), 2)
                
                # 显示检测状态
                if results.pose_landmarks:
                    status_text = "Pose Detected"
                    self.draw_text(image, status_text, (10, 60), 0.7, (0, 255, 0), 2)
                else:
                    status_text = "No pose detected"
                    self.draw_text(image, status_text, (10, 60), 0.7, (0, 0, 255), 2)

                # 绘制姿势关键点
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)# 绘制关键点及连接线
  
                
                if results.pose_landmarks != None:
                    index_pose = 0
                    for pose_landmarks in (results.pose_landmarks.landmark):
                        print(index_pose)
                        mediapipehumanposelist.human_pose_list[index_pose].name = str(NAME_POSE[index_pose])
                        mediapipehumanposelist.human_pose_list[index_pose].x = pose_landmarks.x
                        mediapipehumanposelist.human_pose_list[index_pose].y = pose_landmarks.y
                        mediapipehumanposelist.human_pose_list[index_pose].visibility = pose_landmarks.visibility# 设置可见性
                        index_pose = index_pose +1

                    mediapipehumanposelist.num_humans = 1
                    self.publisher_.publish(mediapipehumanposelist)
                else: # responsavel por mandar 0 nos topicos quando corpo nao esta na tela
                    index_pose = 0
                    for point in mp_pose.PoseLandmark:                          
                                                                                          
                        mediapipehumanposelist.human_pose_list[index_pose].name = str(NAME_POSE[index_pose])
                        mediapipehumanposelist.human_pose_list[index_pose].x = 0.0
                        mediapipehumanposelist.human_pose_list[index_pose].y = 0.0
                        mediapipehumanposelist.human_pose_list[index_pose].visibility = 0.0
                        index_pose = index_pose +1

                
                    mediapipehumanposelist.num_humans = 1
                    self.publisher_.publish(mediapipehumanposelist)

                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break        

def main(args=None):
    rclpy.init(args=args)

    pose_publisher = PosePublisher()
    
    try:
        pose_publisher.getimage_callback()    
    except KeyboardInterrupt:
        print("\n程序被Ctrl+C中断")
    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        # 确保资源被正确释放
        cap.release()
        cv2.destroyAllWindows()
        pose_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    