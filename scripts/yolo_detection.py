#!/usr/bin/env python3
import socket,os,struct, time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolov7_ros_msgs.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge


class ImageStreamerNode(Node):
    def __init__(self):
        super().__init__("image_node")

        self.model = YOLO("yolov8m.pt")
        self.model.to("cuda")

        # Change some class labels for our case
        self.custom_labels = {
            60: "table",    # dining table -> table
            62: "monitor"  # tv -> monitor
        }     

        self.image_sub = self.create_subscription(Image, '/cf_1/image', self.callback, 50)
        self.detection_publisher = self.create_publisher(BoundingBoxes, "/yolo/detections", 50)
        self.cv_bridge = CvBridge()



    def callback(self, img_msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        detections = []
        results = self.model.predict(source=cv_image, conf=0.5)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls_id = int(box.cls[0].item())  # Class ID
                label = self.custom_labels.get(cls_id, result.names[cls_id])

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                detections.append([center_x, center_y, conf, label, cls_id])
                # Draw bounding box
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(cv_image, f"{label}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw a dot at the center of the detection
                # cv2.circle(cv_image, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

            cv2.imshow("YOLOv8 Detections", cv_image)
            cv2.waitKey(1)

            detection_msg = self.create_detection_msg(detections, img_msg)
            self.detection_publisher.publish(detection_msg)
            

            
    def create_detection_msg(self, detections, img_msg):   
        detection_boxes_msg = BoundingBoxes()
        detection_boxes_msg.header = img_msg.header
        if detections:
            for i, detection in enumerate(detections):
                # Bounding box
                bbox = BoundingBox()
                bbox.probability = detection[2]
                bbox.x_center = detection[0]
                bbox.y_center= detection[1]
                bbox.id = detection[4]
                bbox.class_name = detection[3]
                detection_boxes_msg.bounding_boxes.append(bbox)
        return detection_boxes_msg


def main(args=None):
    rclpy.init(args=args)
    node = ImageStreamerNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()