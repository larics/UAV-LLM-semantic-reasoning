import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from message_filters import Subscriber
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from yolov7_ros_msgs.msg import BoundingBoxes

from detect_project_tools import *
from tf_transformations import euler_from_quaternion
import numpy as np

from sklearn.cluster import DBSCAN
from visualization_tools import plot_3d_clusters, plot_2d_xy_clusters
from timer_tool import TrajTimer


# from pycrazyswarm import Crazyswarm
from crazyflie_py import *

class CrazyflieObject(Node):
    def __init__(self, yaml_path,  octree, target_class='couch', publish_cast=False, publish_hits=False, publish_path=True):
        super().__init__('crazyflie_object')

        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.drone = self.swarm.allcfs.crazyflies[0]     # Crazyflie object for controll of the aircraft
        self.trajTimer = TrajTimer(self)
        self.pose = PoseStamped().pose
        self.path = Path()
        self.path.header.frame_id = 'map'

        self.publish_cast = publish_cast
        self.publish_hits = publish_hits
        self.publish_path = publish_path

        self.octree = octree
        # Camera's origin & yaw in world coordinates
        self.camera_origin_world = None
        self.camera_roll = None # radians
        self.camera_pitch = None # radians
        self.camera_yaw = None # radians
        self.K, self.dist_coeffs, self.image_w, self.image_h = load_camera_params(yaml_path)
        
        self.target_class = target_class    # Target class for detection
        self.hit_points_by_class = {}
        # DBSCAN hyperparameters
        self.eps = 0.3
        self.min_samples = 10

        # Subscribe to the pose and detections topics
        self.pose_sub = self.create_subscription(PoseStamped, '/cf_1/pose', self.pose_callback, 5)
        self.boxes_sub = self.create_subscription(BoundingBoxes, '/yolo/detections', self.detection_callback, 5)
        # Publish path, cast direction and hits
        self.pub_cast = self.create_publisher(PointCloud2, '/camera/detections_direction', 5)
        self.pub_hits = self.create_publisher(PointCloud2, '/camera/ray_hits', 5)
        self.pub_path = self.create_publisher(Path, '/cf_1/path', 5)

        # Flags for searching
        self.object_found = False
        self.space_scanning = False
        self.searching_flag = False
        self.started_traj = False
        self.first_pass = True
        self.counter = 0
    

    def pose_callback(self, msg):
        """Callback function to process received position data"""
        self.pose = msg.pose
        self.camera_origin_world = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64)
        self.camera_roll, self.camera_pitch, self.camera_yaw = self.get_roll_pitch_yaw(msg.pose)
        if self.publish_path:
            new_pose = PoseStamped()
            new_pose.header = msg.header
            new_pose.pose = msg.pose
            self.path.poses.append(new_pose)

            # Publish the path
            self.path.header.stamp = self.get_clock().now().to_msg()
            self.pub_path.publish(self.path)



    def detection_callback(self, boxes_msg):

        if (self.camera_origin_world is None) or (self.camera_yaw is None) or not self.searching_flag:
            return

        # We'll collect 3D points from all bounding boxes that match
        direction_points = []
        hit_points = []
        classes_of_hits = []
        self.counter += 1

        for box in boxes_msg.bounding_boxes:
            u_center = int(box.x_center)
            v_center = int(box.y_center)

            # 1) Convert (u, v) -> direction in camera coords
            d_cam = pixel_to_3d_direction(u_center, v_center, self.K, self.dist_coeffs)
            # 2) Convert camera direction -> world direction
            R_wc = camera_to_world_rotation(self.camera_roll, self.camera_pitch, self.camera_yaw)
            d_world = R_wc @ d_cam 

            end = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            hit_point = self.octree.octree.castRay(
                origin=self.camera_origin_world,
                direction=d_world,
                end=end,
                ignoreUnknownCells=True,
            )
            if hit_point:
                hit_points.append(end)
                # Store this in a dictionary keyed by class_name
                class_name = box.class_name
                if class_name not in self.hit_points_by_class:
                    self.hit_points_by_class[class_name] = []
                self.hit_points_by_class[class_name].append(end.copy())
                classes_of_hits.append(class_name)

                # Visualize direction of cast to the object
                dist = np.linalg.norm(self.camera_origin_world - end)
                t_vals = np.arange(0.0, dist, 0.1)  # Point every 0.1m
                points_xyz = []
                for t in t_vals:
                    pw = self.camera_origin_world + t * d_world
                    points_xyz.append(pw)
                points_xyz = np.array(points_xyz, dtype=np.float32)
                direction_points.append(points_xyz)

        
        if self.started_traj and not self.object_found and not self.space_scanning:
            if self.counter >= 50:
                #print("provjeravam")
                self.object_found = self.check_for_object(self.target_class)
                self.counter = 0 
                

        # ---- Publish the points as pointcloud
        if len(direction_points) > 0 and self.publish_cast:
            cat_points = np.concatenate(direction_points, axis=0)
            pc2_msg = xyz_array_to_pointcloud2(self, cat_points, frame_id='map')
            self.pub_cast.publish(pc2_msg)

        # ---- Publish the hits as pointcloud
        if len(hit_points) > 0 and self.publish_hits:
            hits_xyz = np.array(hit_points, dtype=np.float32)
            pc2_hits_msg = xyz_array_to_pointcloud2(self, hits_xyz, frame_id='map')
            self.pub_hits.publish(pc2_hits_msg)


    def stop_trajectory(self):
        self.started_traj = False
        self.drone.goTo(np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z]), self.camera_yaw, 3.0)
 

    def check_for_object(self, target):
        for class_name, hits in self.hit_points_by_class.items():
            num_of_hits = np.array(hits, dtype=np.float64)
            # If the target class is found stop the trajectory
            if class_name == target and num_of_hits.shape[0]:     
                class_centers = self.cluster_and_average()
                if target in class_centers.keys():
                    self.object_found = True
                    self.stop_trajectory()
                    x_coord = round(class_centers[class_name][0][0], 2)
                    y_coord = round(class_centers[class_name][0][1], 2)
                    z_coord = round(class_centers[class_name][0][2], 2)
                    print("{} found at x: {} y: {} z: {}".format(class_name, x_coord, y_coord, z_coord))
                    return True
        #print("not found")
        return False


    def get_position(self):
        """Returns the latest known position of the drone"""
        return (self.pose.position.x, self.pose.position.y, self.pose.position.z)


    def get_roll_pitch_yaw(self, pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)
        return (roll, pitch, yaw)

    def cluster_and_average(self):
        """
        For each class, run DBSCAN on all its 3D hits.
        Then compute the average of each cluster.
        Return a dict: class_name -> list of cluster centroids.
        """

        class_centers = {}  # class_name -> [ [cx,cy,cz], [cx2,cy2,cz2], ... ]

        for class_name, hits in self.hit_points_by_class.items():
            points_arr = np.array(hits, dtype=np.float64)
            if points_arr.shape[0] < self.min_samples:
                # Not enough points to form a cluster
                continue

            db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = db.fit_predict(points_arr)

            unique_labels = set(labels)
            centers = []
            for lbl in unique_labels:
                if lbl == -1:   # outlier in DBSCAN
                    continue
                # Average all points in this cluster
                cluster_points = points_arr[labels == lbl]
                centroid = cluster_points.mean(axis=0)
                centers.append(centroid)
            if len(centers) > 0:
                class_centers[class_name] = centers

        return class_centers



if __name__ == '__main__':
    main()
