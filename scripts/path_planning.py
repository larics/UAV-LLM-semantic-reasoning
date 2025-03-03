import numpy as np
import heapq


def world_to_grid(point, origin, resolution):
    """
    Convert continuous world coordinates to discrete grid coordinates.
    """
    return tuple(int((point[i] - origin[i]) // resolution) for i in range(3))


def grid_to_world(grid_coord, origin, resolution):
    """
    Convert grid coordinates back to continuous world coordinates for the center of that cell.
    """
    return tuple(origin[i] + grid_coord[i]*resolution + resolution/2.0 for i in range(3))


def get_3d_neighbors_26(coord):
    """
    Return the 26-connected neighbors for a 3D grid cell coordinate (x, y, z) except (0,0,0)
    """
    x0, y0, z0 = coord
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((x0 + dx, y0 + dy, z0 + dz))
    return neighbors
    

def euclidean_distance_3d(p1, p2):
    """Return the Euclidean distance between two 3D points or grid coords."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)


def a_star_path_planning(start_xyz, goal_xyz, octree, resolution=0.2, max_iterations=100000, clearance=0.2, step=0.1):
    """
    3D A* that uses 26-connected neighbors and Euclidean distance as both
    the cost for moving between neighbors and the heuristic to the goal.
    """

    # Define an origin for the grid (could also compute bounding box from the OctoMap)
    origin = (0.0, 0.0, 0.0)

    # Convert start/goal to grid
    start_grid = world_to_grid(start_xyz, origin, resolution)
    goal_grid  = world_to_grid(goal_xyz,  origin, resolution)

    g_score = {start_grid: 0.0}
    f_score = {start_grid: euclidean_distance_3d(start_grid, goal_grid)}  # heuristic
    came_from = {}

    # Use a priority queue (min-heap) for the open set
    open_set = []
    heapq.heappush(open_set, (f_score[start_grid], start_grid))

    visited = set()

    # A* search loop
    for _ in range(max_iterations):
        if not open_set:
            break

        # Pop node with lowest f_score
        _, current = heapq.heappop(open_set)

        if current == goal_grid:
            # Reconstruct path
            path_points = reconstruct_path_3d(came_from, current, start_grid, origin, resolution)
            cleaned_points = prune_path(path_points, octree, clearance, step)
            return cleaned_points

        visited.add(current)

        # Expand neighbors (26-connected in 3D)
        for nbr in get_3d_neighbors_26(current):
            if nbr in visited:
                continue

            nbr_world = grid_to_world(nbr, origin, resolution)
            wx, wy, wz = nbr_world
            if not octree.is_free_with_clearance(wx, wy, wz, clearance, step):
                continue

            current_world = grid_to_world(current, origin, resolution)
            move_cost = euclidean_distance_3d(current_world, nbr_world)

            temp_g = g_score[current] + move_cost

            if nbr not in g_score or temp_g < g_score[nbr]:
                came_from[nbr] = current
                g_score[nbr] = temp_g

                # Heuristic 
                h = euclidean_distance_3d(nbr_world, goal_xyz)
                f_score[nbr] = temp_g + h
                heapq.heappush(open_set, (f_score[nbr], nbr))

    return None     # No path found


def reconstruct_path_3d(came_from, current, start_grid, origin, resolution):
    """Reconstruct the path from 'current' back to the start."""
    path = []
    while current in came_from:
        path.append(grid_to_world(current, origin, resolution))
        current = came_from[current]
    # Add the start node
    path.append(grid_to_world(start_grid, origin, resolution))
    path.reverse()
    return path


def line_of_sight_free(p1, p2, octree, clearance, step):
    num_steps = int(max(3, euclidean_distance_3d(p1, p2) // 0.3))
    for t in np.linspace(0, 1, num_steps):
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        point_on_line = p1_arr + t * (p2_arr - p1_arr)
        if not octree.is_free_with_clearance(point_on_line[0], point_on_line[1], point_on_line[2], clearance, step):
            return False    # Obstacle in the way
    return True   # No obstacles in the way


def prune_path(waypoints, octree, clearance, step):
    """
    Attempts to reduce the number of waypoints by skipping
    intermediate ones, as long as there's no obstacle between.
    """

    if not waypoints or len(waypoints) <= 3:
        return waypoints

    pruned = [waypoints[0]]  # always keep the first
    i = 0
    while i < len(waypoints) - 1:
        if len(pruned) >= len(waypoints) - 2:  # Stop pruning if only 3 points remain
            break
        j = len(waypoints) - 1
        found_skip = False
        # Try skipping directly to a farther point
        while j > i+1:
            if len(pruned) + (len(waypoints) - j) <= 3:
                break
            if line_of_sight_free(waypoints[i], waypoints[j], octree, clearance, step):
                # We can skip intermediate points
                pruned.append(waypoints[j])
                i = j
                found_skip = True
                break
            j -= 1
        if not found_skip:
            # No skip found, just move one step
            pruned.append(waypoints[i+1])
            i += 1

    # Ensure the last point (goal) is included
    if pruned[-1] != waypoints[-1]:
        pruned.append(waypoints[-1])

    # Second pass: Ensure no final segments > 1.5 by subdividing where necessary
    final_path = [pruned[0]]
    for idx in range(1, len(pruned)):
        prev = final_path[-1]
        curr = pruned[idx]
        dist = euclidean_distance_3d(prev, curr)
        if dist <= 1.5:
            # Segment is already short enough
            final_path.append(curr)
        else:
            # Subdivide into chunks of <= 1.5
            segments = int(dist // 1.5) + 1
            prev_np = np.array(prev)
            curr_np = np.array(curr)
            for s in range(1, segments + 1):
                new_point = prev_np + (curr_np - prev_np) * (s / segments)
                final_path.append(tuple(new_point))
    
    return np.array(final_path)




if __name__ == "__main__":
    start_xyz = (0.0, 0.0, 0.8)
    goal_xyz  = (4.0, 4.0, 1)
    octomap_file = "/root/diplomski/pitar_sredina.bt"

