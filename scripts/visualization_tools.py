#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def evaluate_polynomial(coeffs, t):
    """
    Given a list of 8 coefficients [a0..a7] for a 7th-degree polynomial:
      p(t) = a0 + a1*t + a2*t^2 + ... + a7*t^7
    returns p(t) at the scalar time t.
    """
    order = len(coeffs)
    powers = np.array([t**i for i in range(order)])
    return np.dot(coeffs, powers)


def load_and_evaluate(csv_filename, dt=0.01):
    """
    Loads the CSV that defines the piecewise polynomial trajectory.
    Returns:
      time_vals, x_vals, y_vals, z_vals, yaw_vals
    each is a numpy array representing the piecewise evaluation over the full trajectory.
    """
    durations = []
    x_coeffs_list = []
    y_coeffs_list = []
    z_coeffs_list = []
    yaw_coeffs_list = []

    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if rows[0][0].lower().startswith("duration"):
        rows = rows[1:]  # skip header row

    for row in rows:
        # Row format = 1 duration + 4*8 coeffs = 33 columns
        # Convert to float
        floats = list(map(float, row))
        duration = floats[0]
        x_coeffs = floats[1:9]
        y_coeffs = floats[9:17]
        z_coeffs = floats[17:25]
        yaw_coeffs = floats[25:33]

        durations.append(duration)
        x_coeffs_list.append(x_coeffs)
        y_coeffs_list.append(y_coeffs)
        z_coeffs_list.append(z_coeffs)
        yaw_coeffs_list.append(yaw_coeffs)

    # piecewise evaluate
    time_vals = []
    x_vals = []
    y_vals = []
    z_vals = []
    yaw_vals = []

    total_time = 0.0

    for i, duration in enumerate(durations):
        # We'll step from t=0..duration in increments of dt
        t_local = 0.0
        while t_local <= duration:
            t_global = total_time + t_local
            # Evaluate polynomials
            x_ = evaluate_polynomial(x_coeffs_list[i], t_local)
            y_ = evaluate_polynomial(y_coeffs_list[i], t_local)
            z_ = evaluate_polynomial(z_coeffs_list[i], t_local)
            yaw_ = evaluate_polynomial(yaw_coeffs_list[i], t_local)

            time_vals.append(t_global)
            x_vals.append(x_)
            y_vals.append(y_)
            z_vals.append(z_)
            yaw_vals.append(yaw_)

            t_local += dt

        total_time += duration

    # convert to numpy arrays
    time_vals = np.array(time_vals)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)
    yaw_vals = np.array(yaw_vals)

    return time_vals, x_vals, y_vals, z_vals, yaw_vals


def plot_trajectory_from_csv(csv_file, occupied_cells=None, sampling_step=0.1):
    time_vals, x_vals, y_vals, z_vals, yaw_vals = load_and_evaluate(csv_file, sampling_step)

    # Create a figure with subplots
    fig = plt.figure(figsize=(10, 8))

    # 1) X, Y, Z vs Time
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time_vals, x_vals, label='x')
    ax1.plot(time_vals, y_vals, label='y')
    ax1.plot(time_vals, z_vals, label='z')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.set_title('Positions vs Time')
    ax1.legend()
    ax1.grid(True)

    # 2) Yaw vs Time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_vals, yaw_vals, label='yaw', color='orange')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Yaw [rad]')
    ax2.set_title('Yaw vs Time')
    ax2.legend()
    ax2.grid(True)

    # 3) 2D XY plot
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_vals, y_vals, 'b-')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_title('XY Top-Down View')
    ax3.grid(True)
    ax3.axis('equal')  # so it doesn't distort the circle or shape

    # 4) 3D Trajectory
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(x_vals, y_vals, z_vals, 'r-')
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_zlabel('Z [m]')
    ax4.set_title('3D Trajectory View')
    if occupied_cells:
        x = [p[0] for p in occupied_cells]
        y = [p[1] for p in occupied_cells]
        z = [p[2] for p in occupied_cells]
        ax4.scatter(x, y, z, c='blue', s=5, marker='.')

    plt.tight_layout()
    plt.show()



def visualize_path_and_octomap(path, occupied_cells=None):
# Extract x, y, z coordinates
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    z = [p[2] for p in path]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c='red', linewidth=2, marker='.')

    if occupied_cells:
        x = [p[0] for p in occupied_cells]
        y = [p[1] for p in occupied_cells]
        z = [p[2] for p in occupied_cells]

        ax.scatter(x, y, z, c='blue', s=15, marker='.')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('3D Points Visualization')
    # Show the plot
    plt.show()


def plot_3d_clusters(class_centers):
        """
        Plot the cluster-averaged hits in 3D. Each class has a separate color.
        class_centers: dict { class_name: [centroid1, centroid2, ...], ... }
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Clustered Hit Points (Averaged)")

        # Distinguish classes by color
        colors = [
            'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta',
            'yellow', 'black', 'brown'
        ]
        color_map = {}

        class_names = sorted(class_centers.keys())
        for i, class_name in enumerate(class_names):
            c = colors[i % len(colors)]
            color_map[class_name] = c

        for class_name, centers in class_centers.items():
            c = color_map[class_name]
            centers = np.array(centers)
            ax.scatter(centers[:,0], centers[:,1], centers[:,2], c=c, label=class_name, s=50)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()


def plot_2d_xy_clusters(class_centers, min_point, max_point, ground_truth):
        """
        Plot the cluster-averaged hits in 2D (X vs. Y). Each class has a separate color.
        """
        x_min, x_max, y_min, y_max = (min_point[0], max_point[0], min_point[1], max_point[1])

        fig, ax = plt.subplots()
        ax.set_title("Clustered Hit Points (Averaged) - 2D X-Y")

        # Invert axes for the required orientation
        #ax.invert_yaxis()  # Y-axis increases to the left
        ax.imshow(ground_truth, cmap='gray', origin='lower', extent=(x_min, x_max, y_min, y_max))

        # Same color logic
        colors = [
            'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta',
            'yellow', 'black', 'brown'
        ]
        color_map = {}
        class_names = sorted(class_centers.keys())
        for i, class_name in enumerate(class_names):
            c = colors[i % len(colors)]
            color_map[class_name] = c

        for class_name, centers in class_centers.items():
            c = color_map[class_name]
            centers = np.array(centers)
            # Plot x vs. y
            ax.scatter(centers[:,0], centers[:,1], c=c, label=class_name, s=50)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.yaxis.set_label_position('right')

        # Show grid and legend
        ax.grid(True)
        ax.legend()
        plt.show()

if __name__ == "__main__":
    csv_file = "my_3dpath_trajectory.csv" 
    dt = 0.01  # time step for sampling
    plot_trajectory_from_csv("temp_trajectory_scan.csv")
