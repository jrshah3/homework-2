import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify, pi, pprint, N

# Define symbolic variables for joint angles
theta1, theta2, theta3, theta4, theta5 = symbols('theta1 theta2 theta3 theta4 theta5')

# Define DH parameters (assuming these are correct for your robot)
dh_params = {
    'd1': 183.30, 'd2': -172.30, 'd3': 172.30, 'd4': -95.50, 'd5': 115.50,
    'a1': 0, 'a2': 731.31, 'a3': 387.80, 'a4': 0, 'a5': 0,
    'alpha1': -pi/2, 'alpha2': 0, 'alpha3': 0, 'alpha4': pi/2, 'alpha5': pi/2
}

def dh_transform(theta, d, a, alpha):
    """Create the D-H transformation matrix for a single link."""
    return Matrix([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Create symbolic transformation matrices
T01 = dh_transform(theta1, dh_params['d1'], dh_params['a1'], dh_params['alpha1'])
T12 = dh_transform(theta2, dh_params['d2'], dh_params['a2'], dh_params['alpha2'])
T23 = dh_transform(theta3, dh_params['d3'], dh_params['a3'], dh_params['alpha3'])
T34 = dh_transform(theta4, dh_params['d4'], dh_params['a4'], dh_params['alpha4'])
T45 = dh_transform(theta5, dh_params['d5'], dh_params['a5'], dh_params['alpha5'])

# Calculate the full transformation matrix symbolically
T05_sym = simplify(T01 * T12 * T23 * T34 * T45)

def forward_kinematics(joint_angles):
    """Calculate the end-effector position and orientation for given joint angles."""
    try:
        T05 = T05_sym.subs({theta1: joint_angles[0], theta2: joint_angles[1], 
                            theta3: joint_angles[2], theta4: joint_angles[3], 
                            theta5: joint_angles[4]}).evalf()
        position = np.array(T05[:3, 3].tolist()).astype(float).flatten()
        rotation = np.array(T05[:3, :3].tolist()).astype(float)
        return position, rotation
    except Exception as e:
        print(f"Error in forward_kinematics: {e}")
        return None, None

# Test configurations (joint angles in radians)
test_configs = [
    [0, 0, 0, 0, 0],           # Home position
    [pi/2, 0, 0, 0, 0],        # Rotate joint 1 by 90°
    [0, pi/2, 0, 0, 0],        # Rotate joint 2 by 90°
    [0, 0, pi/2, 0, 0],        # Rotate joint 3 by 90°
    [0, 0, 0, pi/2, 0],        # Rotate joint 4 by 90°
    [0, 0, 0, 0, pi/2]         # Rotate joint 5 by 90°
]

def to_degrees(config):
    """Convert configuration to degrees, handling both numeric and symbolic inputs."""
    return [float(N(angle * 180 / pi)) for angle in config]

# Validate kinematic equations
for i, config in enumerate(test_configs):
    position, rotation = forward_kinematics(config)
    if position is not None and rotation is not None:
        print(f"\nConfiguration {i+1}:")
        print(f"Joint angles: {np.round(to_degrees(config), 2)} degrees")
        print(f"End-effector position (x, y, z): {np.round(position, 2)} mm")
        print("End-effector orientation (rotation matrix):")
        print(np.round(rotation, 4))
    else:
        print(f"\nFailed to calculate for Configuration {i+1}")

# Print the symbolic transformation matrix
print("\nSymbolic Transformation Matrix T05:")
pprint(T05_sym)
pprint(T05_sym)
