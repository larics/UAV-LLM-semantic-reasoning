import octomap
import numpy as np


class OctoMap():
    def __init__(self):
        self.octree = None
        self.occupied_nodes = []
        self.top_down_map = None

    def load_octomap(self, file_path):
        # Load the OctoMap
        self.octree = octomap.OcTree(file_path.encode())
        for node in self.octree.begin_leafs():
            if self.octree.isNodeOccupied(node):
                self.occupied_nodes.append((node.getX(), node.getY(), node.getZ()))
        self.create_topdown_map()


    def is_free(self, x, y, z):
        """
        Check if (x, y, z) is free in the OctoMap.
        """
        # Arrange the coordinates into numpy array
        coord = np.empty((3,), dtype=float) 
        coord[:] = [x, y, z]

        # Search which node contains this point
        node = self.octree.search(coord)

        try:
            occupation = self.octree.isNodeOccupied(node)
            return not occupation
        except octomap.NullPointerException:
            return True

    def is_free_with_clearance(self, x, y, z, clearance=0.5, step=0.1):
        point = [x, y, z]
        bbx_min = [point[0] - clearance, point[1] - clearance, point[2] - clearance]
        bbx_max = [point[0] + clearance, point[1] + clearance, point[2] + clearance]
        occupied_nodes = []

        # Iterate through nodes within the bounding box
        for node in self.octree.begin_leafs_bbx(np.array(bbx_min), np.array(bbx_max)):
            if self.octree.isNodeOccupied(node):
                occupied_nodes.append((node.getX(), node.getY(), node.getZ()))
        
        if not occupied_nodes:
            return True
        else:
            return False


    def create_topdown_map(self, resolution=0.04, occupancy_threshold=0.5):
        points = np.array(self.occupied_nodes)
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        # Determine grid size
        x_min, x_max = min_point[0], max_point[0]
        y_min, y_max = min_point[1], max_point[1]
        width = int(np.ceil((x_max - x_min) / resolution))
        height = int(np.ceil((y_max - y_min) / resolution))
        
        # Initialize grid to free space (0)
        occupancy_grid = np.ones((height, width), dtype=np.uint8)

        for point in points: 
            x, y, z = (point[0], point[1], point[2])
            
            # Map the (x, y) coordinate to grid indices
            i = int((x - x_min) / resolution)
            j = int((y - y_min) / resolution)
            
            # Ensure indices are within bounds
            if 0 <= i < width and 0 <= j < height:
                occupancy_grid[j, i] = 0  # Mark as occupied

        self.top_down_map = occupancy_grid