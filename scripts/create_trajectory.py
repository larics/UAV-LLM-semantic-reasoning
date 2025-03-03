#!/usr/bin/env python3
import numpy as np
import csv

def angle_wrap(theta):
    #To avoid jumps from pi to -pi
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return vec / norm

def poly_terms(t, deriv=0):
    coefs = np.zeros(8)
    for n in range(8):
        if n >= deriv:
            # Factor = n*(n-1)*... for 'deriv' times
            factor = 1
            for k in range(deriv):
                factor *= (n - k)
            power = n - deriv
            coefs[n] = factor * (t ** power)
    return coefs

def fit_7th_poly_1D(p0, v0, p1, v1, T):
    A = np.zeros((8, 8))
    b = np.zeros(8)

    # p(0) = p0
    A[0, :] = poly_terms(0.0, deriv=0)
    b[0] = p0
    # p(T) = p1
    A[1, :] = poly_terms(T, deriv=0)
    b[1] = p1
    # p'(0) = v0
    A[2, :] = poly_terms(0.0, deriv=1)
    b[2] = v0
    # p'(T) = v1
    A[3, :] = poly_terms(T, deriv=1)
    b[3] = v1
    # p''(0)=0
    A[4, :] = poly_terms(0.0, deriv=2)
    # p''(T)=0
    A[5, :] = poly_terms(T, deriv=2)
    # p'''(0)=0
    A[6, :] = poly_terms(0.0, deriv=3)
    # p'''(T)=0
    A[7, :] = poly_terms(T, deriv=3)

    coeffs = np.linalg.solve(A, b)
    return coeffs.tolist()

def fit_yaw_7th_poly(yaw0, yaw1, T):
    """
    7th-degree polynomial for yaw from yaw0->yaw1
    with zero derivative at endpoints.
    """
    return fit_7th_poly_1D(yaw0, 0.0, yaw1, 0.0, T)

def create_polynomial_trajectory_from_points(points, speed, csv_filename, yaw_direction=None, start_yaw=None):
    """
    Given:
      - 'points': list of (x, y, z) defining the path to follow.
      - 'speed': constant speed for each segment (m/s).
      - 'csv_filename': the name of the output CSV.
      - 'yaw_direction': optional (x, y, z) coordinate that the drone should always face.
        If None, the drone's yaw will be in the direction of travel.
      - 'start_yaw': optional initial yaw value.
    """

    end_yaw = None
    n = len(points)
    if n < 2:
        print("Need at least 2 points!")
        return

    # Compute distances and durations for each segment
    segments = [] 
    for i in range(n-1):
        p0 = np.array(points[i])
        p1 = np.array(points[i+1])
        diff = p1 - p0
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            print(f"Points {i} and {i+1} are the same! Skipping segment.")
            continue
        direction_3d = diff / dist  # unit vector
        T = dist / speed
        segments.append((p0, p1, dist, direction_3d, T))

    # etermine velocity at each waypoint
    velocities = [None]*n

    if segments:
        velocities[0] = segments[0][3] * speed  # direction_3d * speed
    if len(segments) > 0:
        velocities[n-1] = segments[-1][3] * speed

    for i in range(1, n-1):
        dir_prev = segments[i-1][3]
        dir_next = segments[i][3]
        avg_dir = dir_prev + dir_next
        avg_dir = normalize(avg_dir)
        velocities[i] = avg_dir * speed

    # Fit polynomials for each segment and write CSV rows
    rows = []
    if yaw_direction is not None:
        yaw_direction = np.array(yaw_direction)

    for i in range(len(segments)):
        p0, p1, dist, direction_3d, T = segments[i]
        idx_start = i
        idx_end   = i + 1

        # boundary velocities
        v0 = velocities[idx_start]
        v1 = velocities[idx_end]

        x0, y0, z0 = p0
        vx0, vy0, vz0 = v0
        x1, y1, z1 = p1
        vx1, vy1, vz1 = v1

        # Yaw based on yaw_direction
        if yaw_direction is not None:
            if i == 0 and start_yaw is not None:
                yaw0 = start_yaw
            else:
                dir_to_target0 = yaw_direction - p0
                yaw0 = np.arctan2(dir_to_target0[1], dir_to_target0[0])

            dir_to_target1 = yaw_direction - p1
            yaw1 = np.arctan2(dir_to_target1[1], dir_to_target1[0])
        else:
            # Yaw follows velocity direction
            if i == 0 and start_yaw is not None:
                yaw0 = start_yaw
            else:
                yaw0 = np.arctan2(vy0, vx0)
            yaw1 = np.arctan2(vy1, vx1)

        # No large 2π jumps
        diff_yaw = angle_wrap(yaw1 - yaw0)
        yaw1 = yaw0 + diff_yaw
        end_yaw = yaw1

        # Fit polynomials
        x_poly = fit_7th_poly_1D(x0, vx0, x1, vx1, T)
        y_poly = fit_7th_poly_1D(y0, vy0, y1, vy1, T)
        z_poly = fit_7th_poly_1D(z0, vz0, z1, vz1, T)
        yaw_poly = fit_yaw_7th_poly(yaw0, yaw1, T)

        # One row - duration + 4*8 coeffs
        row = [T] + x_poly + y_poly + z_poly + yaw_poly
        rows.append(row)

    headers = [
        "duration",
        "x^0","x^1","x^2","x^3","x^4","x^5","x^6","x^7",
        "y^0","y^1","y^2","y^3","y^4","y^5","y^6","y^7",
        "z^0","z^1","z^2","z^3","z^4","z^5","z^6","z^7",
        "yaw^0","yaw^1","yaw^2","yaw^3","yaw^4","yaw^5","yaw^6","yaw^7"
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([f"{val:.6f}" for val in row])

    #print(f"Trajectory CSV written to {csv_filename}")
    return end_yaw

if __name__ == "__main__":
    pass
