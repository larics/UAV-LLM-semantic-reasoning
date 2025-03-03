#!/usr/bin/env python3
import socket,os,struct, time
import numpy as np
import cv2
import yaml
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from ultralytics import YOLO
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from yolov7_ros_msgs.msg import BoundingBox, BoundingBoxes


class ImageStreamerNode(Node):
    def __init__(self):
        super().__init__("image_node")

        self.model = YOLO("yolov8m.pt")
        self.model.to("cuda")

        self.custom_labels = {
            60: "table",    # dining table -> table
            62: "monitor"  # tv -> monitor
        }     

        self.detection_publisher = self.create_publisher(BoundingBoxes, "/yolo/detections", 10)

        # declare config path parameter
        self.declare_parameter(
            name="config_path",
            value=os.path.join(
                    get_package_share_directory('crazyflie'),
                    'config',
                    'aideck_streamer.yaml'
                )
        )

        config_path = self.get_parameter("config_path").value
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # declare topic names
        self.declare_parameter(
            name="image_topic",
            value=config["image_topic"],
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to publish to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value=config["camera_info_topic"],
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        # declare aideck ip and port
        self.declare_parameter(
            name='deck_ip',
            value=config['deck_ip'],        
            )
        
        self.declare_parameter(
            name='deck_port',
            value=config['deck_port'],        
        )

        # define variables from ros2 parameters
        image_topic = (
            self.get_parameter("image_topic").value
        )
        self.get_logger().info(f"Image topic: {image_topic}")

        info_topic = (
            self.get_parameter("camera_info_topic").value
        )
        self.get_logger().info(f"Image info topic: {info_topic}")


        # create messages and publishers
        #self.image_msg = Image()
        #self.camera_info_msg = self._construct_from_yaml(config)
        #self.image_publisher = self.create_publisher(Image, image_topic, 10)
        #self.info_publisher = self.create_publisher(CameraInfo, info_topic, 10)

        # set up connection to AI Deck
        deck_ip = self.get_parameter("deck_ip").value
        deck_port = int(self.get_parameter("deck_port").value)
        self.get_logger().info("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((deck_ip, deck_port))
        self.get_logger().info("Socket connected")
        self.image = None
        self.rx_buffer = bytearray()

        # set up timers for callbacks
        timer_period = 0.1
        self.rx_timer = self.create_timer(timer_period, self.receive_callback)


    def _rx_bytes(self, size):
        data = bytearray()
        while len(data) < size:
            data.extend(self.client_socket.recv(size-len(data)))
        return data

    def receive_callback(self):
        # first get the info
        packetInfoRaw = self._rx_bytes(4)
        [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)

        # receive the header
        imgHeader = self._rx_bytes(length - 2)
        [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

        # if magic is correct, get new image
        if magic == 0xBC:
            imgStream = bytearray()

            while len(imgStream) < size:
                packetInfoRaw = self._rx_bytes(4)
                [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
                chunk = self._rx_bytes(length - 2)
                imgStream.extend(chunk)

            raw_img = np.frombuffer(imgStream, dtype=np.uint8)
            raw_img = raw_img.reshape((height, width))
            raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
            detections = []
            results = self.model.predict(source=raw_img_rgb, conf=0.5)
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
                    cv2.rectangle(raw_img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(raw_img_rgb, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Draw a dot at the center of the detection
                    #cv2.circle(raw_img_rgb, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

            cv2.imshow("YOLOv8 Detections", raw_img_rgb)
            cv2.waitKey(1)
            #print(detections)
            detection_msg = self.create_detection_msg(detections)
            self.detection_publisher.publish(detection_msg)

            self.image = raw_img

            
    def create_detection_msg(self, detections):   
        detection_boxes_msg = BoundingBoxes()
        if detections:
            for i, detection in enumerate(detections):
                # bounding box
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