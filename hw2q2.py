import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify, pi, pprint, diff

# Define symbolic variables for joint angles
theta1, theta2, theta3, theta4, theta5 = symbols('theta1 theta2 theta3 theta4 theta5')

# Define DH parameters
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

def derive_jacobian():
    # Extract the position vector from T05_sym
    p = T05_sym[:3, 3]
    
    # Calculate the linear velocity components
    J_v = Matrix([
        [diff(p[0], theta1), diff(p[0], theta2), diff(p[0], theta3), diff(p[0], theta4), diff(p[0], theta5)],
        [diff(p[1], theta1), diff(p[1], theta2), diff(p[1], theta3), diff(p[1], theta4), diff(p[1], theta5)],
        [diff(p[2], theta1), diff(p[2], theta2), diff(p[2], theta3), diff(p[2], theta4), diff(p[2], theta5)]
    ])
    
    # Calculate the angular velocity components
    z0 = Matrix([0, 0, 1])
    z1 = T01[:3, 2]
    z2 = (T01 * T12)[:3, 2]
    z3 = (T01 * T12 * T23)[:3, 2]
    z4 = (T01 * T12 * T23 * T34)[:3, 2]
    
    J_w = Matrix([z0, z1, z2, z3, z4]).T
    
    # Combine linear and angular components
    J = Matrix.vstack(J_v, J_w)
    
    return simplify(J)

# Calculate the symbolic Jacobian matrix
J_sym = derive_jacobian()

print("Symbolic Jacobian Matrix:")
pprint(J_sym)

def calculate_jacobian(joint_angles):
    """Calculate the numerical Jacobian for given joint angles."""
    try:
        J = J_sym.subs({theta1: joint_angles[0], theta2: joint_angles[1], 
                        theta3: joint_angles[2], theta4: joint_angles[3], 
                        theta5: joint_angles[4]}).evalf()
        return np.array(J.tolist()).astype(float)
    except Exception as e:
        print(f"Error in calculate_jacobian: {e}")
        return None

# Test configuration (e.g., home position)
home_config = [0, 0, 0, 0, 0]

# Calculate Jacobian for home configuration
J_home = calculate_jacobian(home_config)

print("\nJacobian at home configuration:")
print(np.round(J_home, 4))
