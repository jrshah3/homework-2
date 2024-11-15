import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define DH parameters for UR3
dh_params = {
    'd1': 151.9, 'd2': 0, 'd3': 0, 'd4': 112.35, 'd5': 85.35, 'd6': 81.9,
    'a1': 0, 'a2': 243.65, 'a3': 213.25, 'a4': 0, 'a5': 0, 'a6': 0,
    'alpha1': np.pi/2, 'alpha2': 0, 'alpha3': 0, 'alpha4': np.pi/2, 'alpha5': -np.pi/2, 'alpha6': 0
}

def dh_transform(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(q):
    T01 = dh_transform(q[0], dh_params['d1'], dh_params['a1'], dh_params['alpha1'])
    T12 = dh_transform(q[1], dh_params['d2'], dh_params['a2'], dh_params['alpha2'])
    T23 = dh_transform(q[2], dh_params['d3'], dh_params['a3'], dh_params['alpha3'])
    T34 = dh_transform(q[3], dh_params['d4'], dh_params['a4'], dh_params['alpha4'])
    T45 = dh_transform(q[4], dh_params['d5'], dh_params['a5'], dh_params['alpha5'])
    T56 = dh_transform(q[5], dh_params['d6'], dh_params['a6'], dh_params['alpha6'])
    
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    return T06

def jacobian(q):
    epsilon = 1e-6
    J = np.zeros((6, 6))
    T = forward_kinematics(q)
    p = T[:3, 3]
    R = T[:3, :3]
    
    for i in range(6):
        q_epsilon = q.copy()
        q_epsilon[i] += epsilon
        T_epsilon = forward_kinematics(q_epsilon)
        p_epsilon = T_epsilon[:3, 3]
        R_epsilon = T_epsilon[:3, :3]
        
        J[:3, i] = (p_epsilon - p) / epsilon
        J[3:, i] = np.dot(R.T, (R_epsilon - R)) / epsilon
    
    return J

def inverse_kinematics(q, target_pos, target_ori):
    max_iterations = 1000
    tolerance = 1e-3
    
    for _ in range(max_iterations):
        T = forward_kinematics(q)
        current_pos = T[:3, 3]
        current_ori = T[:3, :3]
        
        pos_error = target_pos - current_pos
        ori_error = np.dot(target_ori, current_ori.T) - np.eye(3)
        ori_error = np.array([ori_error[2, 1], ori_error[0, 2], ori_error[1, 0]])
        
        error = np.concatenate((pos_error, ori_error))
        
        if np.linalg.norm(error) < tolerance:
            break
        
        J = jacobian(q)
        q_dot = np.linalg.pinv(J) @ error
        q += q_dot
    
    return q

def generate_trajectory(center, radius, rect_width, rect_height):
    t = np.linspace(0, 20, 1000)
    trajectory = []
    
    # Semicircle
    for ti in t[:500]:
        x = center[0] + radius * np.cos(np.pi * ti / 10)
        y = center[1] + radius * np.sin(np.pi * ti / 10)
        z = center[2]
        trajectory.append([x, y, z])
    
    # Rectangle
    rect_points = [
        [center[0] - rect_width/2, center[1] + radius, center[2]],
        [center[0] + rect_width/2, center[1] + radius, center[2]],
        [center[0] + rect_width/2, center[1] + radius - rect_height, center[2]],
        [center[0] - rect_width/2, center[1] + radius - rect_height, center[2]],
        [center[0] - rect_width/2, center[1] + radius, center[2]]
    ]
    
    for i in range(4):
        start = rect_points[i]
        end = rect_points[i+1]
        for ti in np.linspace(0, 1, 125):
            x = start[0] + ti * (end[0] - start[0])
            y = start[1] + ti * (end[1] - start[1])
            z = start[2] + ti * (end[2] - start[2])
            trajectory.append([x, y, z])
    
    return np.array(trajectory)

def main():
    # Set up the problem
    center = [400, 0, 200]  # Example center point
    radius = 100
    rect_width = 200
    rect_height = 150
    
    # Generate trajectory
    trajectory = generate_trajectory(center, radius, rect_width, rect_height)
    
    # Initial configuration (home position)
    q_home = np.zeros(6)
    
    # Desired orientation (assuming constant orientation)
    target_ori = np.eye(3)
    
    # Solve inverse kinematics for each point in the trajectory
    q_trajectory = []
    q = q_home
    
    for target_pos in trajectory:
        q = inverse_kinematics(q, target_pos, target_ori)
        q_trajectory.append(q)
    
    q_trajectory = np.array(q_trajectory)
    
    # Plot the results
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('End-effector Trajectory')
    
    ax2 = fig.add_subplot(122)
    for i in range(6):
        ax2.plot(q_trajectory[:, i], label=f'Joint {i+1}')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Joint angle (rad)')
    ax2.set_title('Joint Trajectories')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
