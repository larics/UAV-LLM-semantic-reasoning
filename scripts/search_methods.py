import octomap
import numpy as np
import math
from visualization_tools import *
from create_trajectory import create_polynomial_trajectory_from_points
from path_planning import a_star_path_planning
from crazyflie_py.uav_trajectory import Trajectory


def get_wall_and_center(octree, z_level):
    """
    Slices the OctoMap at a given z_level, returns center, min, max points and wall points.
    """
    sliced_octree = octomap.OcTree(octree.getResolution())

    for node in octree.begin_leafs(maxDepth = 16):
        if node.getCoordinate()[2] >= z_level and node.getCoordinate()[2] < z_level + octree.getResolution() and node.getDepth() == octree.getTreeDepth():
           sliced_octree.updateNode(node.getKey(), False)

    points = []
    for node in sliced_octree.begin_leafs():
        points.append(np.array([node.getCoordinate()[1], node.getCoordinate()[0]]))
    points = np.array(points)

    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    centroid = np.array([mean_x, mean_y])
    centroid3d = np.concatenate((centroid, np.array([z_level])), axis=0)

    angles = np.arange(0, 360, 4)  # Angles from 0 to 359 degrees
    directions = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a)), 0] for a in angles])
    hit_points = []
    for direction in directions:
        end = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        hit_point = octree.castRay(
            origin=centroid3d,
            direction=direction,
            end=end,
            ignoreUnknownCells=True,
        )
        if hit_point:
            hit_points.append(end[:-1])

    hit_points =  np.c_[np.array(hit_points), np.ones(np.array(hit_points).shape[0])*z_level]
    min_point = np.min(hit_points, axis=0)
    max_point = np.max(hit_points, axis=0)

    # Reduce number of points
    wall_pruned = [hit_points[0]]
    for idx in range(1, len(hit_points)):
        prev = wall_pruned[-1]
        curr = hit_points[idx]
        dist = np.linalg.norm(prev - curr)
        if dist >= 1.5:
            wall_pruned.append(curr)
        else:
            continue

    return centroid, min_point, max_point, np.array(wall_pruned)


def reorder_circle(points):
    n = len(points)
    if n < 2:
        return points
    # Distances between consecutive points, wrap-around with (i+1)%n
    dists = [round(math.hypot(points[i+1][0] - p[0], points[i+1][1] - p[1]), 2) for i, p in enumerate(points[:-1])]
    
    if not np.all([i == dists[0] for i in dists]):
        i_max = max(range(n-1), key=dists.__getitem__)
        points_rolled = np.roll(points, shift=-(i_max+1), axis=0)
        return points_rolled
    else:
        return points

def generate_circle_points(center, radius, num_points=36):
    cx, cy = center
    waypoints = []

    # Generate points around the circle
    for i in range(num_points):
        theta = (i / num_points) * 2 * np.pi  # Convert to radians
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        waypoints.append([x, y])

    return np.array(waypoints)


def gen_circle_scan_waypoints(cf, octree, height):
    centroid, min_point, max_point, _ = get_wall_and_center(octree, height)
    radius = (min(int(max_point[0] - min_point[0]), int(max_point[1] - min_point[1])) // 2) - 2
    waypoints = generate_circle_points(centroid, radius=radius, num_points=18)
    waypoints3d = np.c_[waypoints, np.ones(waypoints.shape[0])*height]
    centroid3d = np.concatenate((centroid, np.array([height])), axis=0)

    # Make the first point the one closest to the drone
    closest = np.array(waypoints3d[0])
    origin = np.array(cf.get_position())
    for i in waypoints3d:
        if np.linalg.norm(i - origin) < np.linalg.norm(closest - origin):
            closest = i
    point_dict = {tuple(pt): i for i, pt in enumerate(waypoints3d)}

    waypoints3d = np.roll(waypoints3d, shift=-point_dict[tuple(closest)], axis=0)
    return waypoints3d, min_point, max_point, centroid3d


def gen_wall_following_waypoints(cf, octree, height, scale_factor = 0.5):
    centroid, min_point, max_point, wall_points = get_wall_and_center(octree, height)
    centroid3d = np.concatenate((centroid, np.array([height])), axis=0)
    
    points_centered = wall_points - centroid3d
    points_scaled_centered = points_centered * scale_factor
    wall_points_scaled = points_scaled_centered + centroid3d

    # Make the first point the one closest to the drone
    closest = np.array(wall_points_scaled[0])
    origin = np.array(cf.get_position())
    for i in wall_points_scaled:
        if np.linalg.norm(i - origin) < np.linalg.norm(closest - origin):
            closest = i
    point_dict = {tuple(pt): i for i, pt in enumerate(wall_points_scaled)}

    wall_points_scaled = np.roll(wall_points_scaled, shift=-point_dict[tuple(closest)], axis=0)

    return wall_points_scaled, min_point, max_point, centroid3d


def do_initial_scan(cf, octree, SPEED, TIMESCALE, scan_height, scan_type='circle', visualize=False):
    print(f"Beginning {scan_type} scan ...")
    if scan_type == 'circle':
        circle_scan, min_point, max_point, centroid = gen_circle_scan_waypoints(cf, octree.octree, scan_height)
        points_to_circle = np.array(a_star_path_planning(cf.get_position(), circle_scan[0], octree, resolution=0.25, max_iterations=int(10e6), clearance=0.2, step=0.1))
        if points_to_circle.shape[0] == 1:
            all_points = np.concatenate((points_to_circle, circle_scan), axis=0)
        else:
            all_points = np.concatenate((points_to_circle[:-1], circle_scan), axis=0)
        all_points = np.concatenate((points_to_circle[:-1], circle_scan), axis=0)
        yaw_direction = centroid

    elif scan_type == 'wall':
        wall_scan, min_point, max_point, centroid = gen_wall_following_waypoints(cf, octree.octree, scan_height)
        points_to_wall = np.array(a_star_path_planning(cf.get_position(), wall_scan[0], octree, resolution=0.25, max_iterations=int(10e6), clearance=0.2, step=0.1))
        if points_to_wall.shape[0] == 1:
            all_points = np.concatenate((points_to_wall, wall_scan), axis=0)
        else:
            all_points = np.concatenate((points_to_wall[:-1], wall_scan), axis=0)
        all_points = np.concatenate((points_to_wall[:-1], wall_scan), axis=0)
        yaw_direction = None

    end_yaw = create_polynomial_trajectory_from_points(
        points = all_points,
        speed  = SPEED,
        csv_filename = "temp_trajectory_scan.csv",
        yaw_direction = yaw_direction,
        start_yaw = cf.camera_yaw
    )
    
    if visualize:
        #visualize_path_and_octomap()
        plot_trajectory_from_csv("temp_trajectory_scan.csv", octree.occupied_nodes)

    trajectory = Trajectory()
    trajectory.loadcsv("temp_trajectory_scan.csv")
    cf.drone.uploadTrajectory(0, 0, trajectory)
    cf.timeHelper.sleep(1.0)
    cf.drone.startTrajectory(0, timescale=TIMESCALE, relative = False)
    cf.started_traj = True      # important!
    cf.timeHelper.sleep(3.0)
    cf.trajTimer.sleep_while_moving()
    cf.timeHelper.sleep(1.0)
    #cf.timeHelper.sleep(trajectory.duration * TIMESCALE + 5.0)
    cf.started_traj = False
    print("Scanning done.")


def first_lookaround(cf):
    print("First scan around ...")
    cf.drone.goTo(np.array([cf.pose.position.x, cf.pose.position.y, cf.pose.position.z]), cf.camera_yaw-1.6, 8.0)
    cf.timeHelper.sleep(8.0)
    cf.drone.goTo(np.array([cf.pose.position.x, cf.pose.position.y, cf.pose.position.z]), cf.camera_yaw+1.6, 8.0)
    cf.timeHelper.sleep(8.0)


def do_thorough_search(cf, SPEED, TIMESCALE, object_name, coordinates, min_point, max_point, higher, radius, wall_clearance, visualize):
    print(f"Starting thorough search around {object_name} at [{coordinates[0], coordinates[1], coordinates[2]}]...")
    height = coordinates[2] + higher
    x_min, y_min, x_max, y_max = (min_point[0]+wall_clearance, min_point[1]+wall_clearance, max_point[0]-wall_clearance, max_point[1]-wall_clearance)
    waypoints2d = generate_circle_points(coordinates[:-1], radius=radius, num_points=18)

    waypoints3d = np.c_[waypoints2d, np.ones(waypoints2d.shape[0])*height]
    waypoints3d = [point for point in waypoints3d if cf.octree.is_free_with_clearance(point[0], point[1], point[2], clearance=0.2, step=0.1)]
    waypoints3d = reorder_circle(np.array([(x, y, z) for x, y, z in waypoints3d if x_min <= x <= x_max and y_min <= y <= y_max]))

    #Make the first point the one closest to the drone
    if np.linalg.norm(waypoints3d[-1] - cf.get_position()) < np.linalg.norm(waypoints3d[0] - cf.get_position()):
        waypoints3d = np.flipud(waypoints3d)
    
    points_to_traj = a_star_path_planning(cf.get_position(), waypoints3d[0], cf.octree, resolution=0.25, max_iterations=int(10e6), clearance=0.2, step=0.1)

    all_points = np.concatenate((points_to_traj[:-1], waypoints3d), axis=0)

    end_yaw = create_polynomial_trajectory_from_points(
        points = all_points,
        speed  = SPEED,
        csv_filename = "temp_trajectory_scan.csv",
        yaw_direction = coordinates,
        start_yaw = cf.camera_yaw
    )
    
    if visualize:
        plot_trajectory_from_csv("temp_trajectory_scan.csv", cf.octree.occupied_nodes)

    trajectory = Trajectory()
    trajectory.loadcsv("temp_trajectory_scan.csv")
    cf.drone.uploadTrajectory(0, 0, trajectory)
    cf.timeHelper.sleep(1.0)
    cf.drone.startTrajectory(0, timescale=TIMESCALE, relative = False)
    cf.started_traj = True      # important!
    cf.timeHelper.sleep(3.0)
    cf.trajTimer.sleep_while_moving()
    cf.timeHelper.sleep(1.0)
    #cf.timeHelper.sleep(trajectory.duration * TIMESCALE + 5.0)
    cf.started_traj = False
    
    print("Thorough search done.")


if __name__ == "__main__":
    height = 2.5
    centroid, min_point, max_point, wall_points = get_wall_and_center(octree.octree, height)
    do_thorough_search(cf, SPEED, TIMESCALE, [9.02700043, 2.68869238, 1.04099998], min_point, max_point, higher=1.0, radius=2.0, wall_clearance=1.0, visualize=True)
    