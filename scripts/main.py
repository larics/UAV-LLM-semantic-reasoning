#!/usr/bin/env python3
import numpy as np
import heapq
from pathlib import Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import traceback

from octomap_tools import OctoMap
from path_planning import a_star_path_planning
from create_trajectory import create_polynomial_trajectory_from_points
from visualization_tools import plot_trajectory_from_csv, visualize_path_and_octomap
from timer_tool import Timer
from search_methods import *
from LLM_tools import describe_space, decide_movement, define_target_object

from crazyflie_object import *
from crazyflie_py.uav_trajectory import Trajectory


TAKEOFF_DURATION = 5.0
HOVER_DURATION = 5.0
SPEED = 0.65               # Desired velocity along the path
TIMESCALE = 2.0


def wait_for_input():
    user_input = input("Press Enter to execute trajectory or 'q' to quit: ")
    if user_input.lower() == 'q':
        exit()

def start_ros_node(node):
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

def remove_visited(objects, visited, epsilon):
    filtered_objects = {}
    
    for obj, points in objects.items():
        if obj in visited:
            visited_points = visited[obj]
            # Keep only the points that are not close to any visited point
            filtered_points = [p for p in points if not any(np.linalg.norm(np.array(p) - np.array(vp)) < epsilon for vp in visited_points)]
            if filtered_points:
                filtered_objects[obj] = filtered_points
        else:
            filtered_objects[obj] = points  # No visited points for this object, keep all
    return filtered_objects

def LLM_schedule_call(interval, interesting_object_thread, cf, decide_movement, target_description, remove_visited, visited):
    epsilon = 0.3
    while True:
        time.sleep(interval)
        class_centers = cf.cluster_and_average()
        class_centers = remove_visited(class_centers, visited, epsilon)
        interesting_object_flag, object_name, obj_coord, explanation = decide_movement(class_centers, target_description)
        #print(class_centers)
        #print(interesting_object_flag)
        #print(explanation)
        #print(object_name, obj_coord)
        if interesting_object_flag:
            cf.stop_trajectory()
            if object_name not in visited:
                visited[object_name] = []
            visited[object_name].append(obj_coord)
            interesting_object_thread[0] = True
            interesting_object_thread[1] = obj_coord
            interesting_object_thread[2] = object_name
            break
            
    

def main(yaml_path, octomap_path):    
    rclpy.init()
    octomap_file = octomap_path
    octree = OctoMap()
    octree.load_octomap(octomap_file)
    
    object_searching = True
    space_scanning = False
    search_withouth_LLM = True
    first_pass = True
    interesting_object_thread = [False, [0,0,0]]
    visited = {}
    timer = Timer()

    ''' Start '''
    target_description = "Where is my banana?"
    target_class, _ = define_target_object(target_description)
    print(f"The target object is a {target_class}")
    scan_height = 2.5   # Height on which the wall is scanned
    
    try:
        cf = CrazyflieObject(yaml_path, octree, target_class)
        centroid, min_point, max_point, wall_points = get_wall_and_center(octree.octree, scan_height)
        cf.space_scanning = space_scanning

        # Run in background
        ros_thread = threading.Thread(target=start_ros_node, args=(cf,), daemon=True)
        ros_thread.start()
        cf.drone.takeoff(targetHeight=scan_height, duration=TAKEOFF_DURATION)
        cf.timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)   

        while not cf.object_found:
            if object_searching:
                timer.__enter__()
                if first_pass:
                    first_pass = False
                    cf.searching_flag = True
                    first_lookaround(cf)
                    end_search = cf.check_for_object(target_class)
                    if end_search: 
                        cf.object_found = True
                        timer.__exit__()
                        break
                    class_centers = cf.cluster_and_average()
                    interesting_object_flag, object_name, obj_coord, explanation = decide_movement(class_centers, target_description)
                    #print(class_centers)
                    #print(interesting_object_flag)
                    #print(explanation)
                    #print(object_name, obj_coord)
                    if interesting_object_flag:
                        do_thorough_search(cf, SPEED, TIMESCALE, object_name, obj_coord, min_point, max_point, higher=1.0, radius=2.0, wall_clearance=1.0, visualize=False)
                        if object_name not in visited:
                            visited[object_name] = []
                        visited[object_name].append(obj_coord)
                        end_search = cf.check_for_object(target_class)
                        if end_search: 
                            cf.object_found = True
                            timer.__exit__()
                            break
                
                scheduler_thread = threading.Thread(
                    target=LLM_schedule_call,
                    args=(10, interesting_object_thread, cf, decide_movement, target_description, remove_visited, visited),
                    daemon=True
                )
                scheduler_thread.start()
                
                do_initial_scan(cf, octree, SPEED, TIMESCALE, scan_height, scan_type='wall', visualize=False)
                if cf.object_found:
                    timer.__exit__()
                    break
                if interesting_object_thread[0]:
                    scheduler_thread.join()
                    interesting_object_thread = [False, [0,0,0]]
                    do_thorough_search(cf, SPEED, TIMESCALE, interesting_object_thread[2], interesting_object_thread[1], min_point, max_point, higher=1.0, radius=2.5, wall_clearance=1.0, visualize=False)
                    end_search = cf.check_for_object(target_class)
                    if end_search: 
                        cf.object_found = True
                        break
                else:
                    scheduler_thread.join()

            elif space_scanning:
                cf.space_scanning = True
                cf.searching_flag = True
                do_initial_scan(cf, octree, SPEED, TIMESCALE, scan_height, scan_type='wall', visualize=False)
                do_initial_scan(cf, octree, SPEED, TIMESCALE, scan_height, scan_type='circle', visualize=False)
                class_centers = cf.cluster_and_average()
                description = describe_space(class_centers)
                print(description)
                plot_2d_xy_clusters(class_centers, min_point, max_point, octree.top_down_map)
                cf.object_found = True
                break

            elif search_withouth_LLM:
                timer.__enter__()
                if first_pass:
                    first_pass = False
                    cf.searching_flag = True
                    first_lookaround(cf)
                    
                do_initial_scan(cf, octree, SPEED, TIMESCALE, scan_height, scan_type='wall', visualize=False)
                timer.__exit__()
                break


        print("Done successfully!")
        cf.drone.land(targetHeight=0.04, duration=3)
        cf.timeHelper.sleep(TAKEOFF_DURATION+3.0)

        cf.destroy_node()
        rclpy.shutdown()

    except Exception as e:
        print("Stopping...")
        print(e)
        traceback.print_exc()
        cf.destroy_node()
        rclpy.shutdown()
        scheduler_thread.join()


if __name__ == "__main__":
    yaml_path = "~/UAV-control-using-semantics-and-LLM/config/camera_params.yaml"
    octomap_path = "~/UAV-control-using-semantics-and-LLM/data/crazysim_octomap.bt"
    main(yaml_path, octomap_path)
