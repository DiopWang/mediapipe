#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.log_timer = self.create_timer(1.0, self.log_callback) # 1Hz
        self.bridge = CvBridge()
        # self.cap = cv2.VideoCapture(1)  # 打开摄像头，0为默认摄像头
        self.cap = cv2.VideoCapture(0)  # 打开摄像头，0为默认摄像头
        # 尝试不同的摄像头索引
        # for camera_index in [1, 0]:  # 先尝试video1，再尝试video0
        #     self.cap = cv2.VideoCapture(camera_index)
        #     if self.cap.isOpened():
        #         self.get_logger().info(f'成功打开摄像头 /dev/video{camera_index}')
        #         break
        #     else:
        #         self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            self.get_logger().error('无法打开任何摄像头设备')
            raise RuntimeError('没有可用的摄像头')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # 将OpenCV图像转换为ROS 2图像消息
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(img_msg)
            
    def log_callback(self):
        if self.cap.isOpened():
            self.get_logger().info('Publishing image')
        else:
            self.get_logger().error('Camera is not opened')
            

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()