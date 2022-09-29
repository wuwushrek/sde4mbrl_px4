import tf_transformations as tft
import numpy as np

def w2x(q):
    """ Quaternion in w, x,y,z to x,y,z, w

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Quaternion [x, y, z, w].
    """
    return q[1], q[2], q[3], q[0]

def x2w(q):
    """ Quaternion in x, y, z, w to w, x,y,z

    Args:
        q (list): Quaternion [x, y, z, w].

    Returns:
        list: Quaternion [w, x, y, z].
    """
    return q[3], q[0], q[1], q[2]

def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        list: Quaternion [w, x, y, z].
    """
    return x2w(tft.quaternion_from_euler(roll, pitch, yaw))

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles.

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Euler angles [roll, pitch, yaw].
    """
    return tft.euler_from_quaternion(w2x(q))

def quaternion_get_yaw(q):
    """Get yaw angle from quaternion.

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        float: Yaw angle in radians.
    """
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

# ENU_NED_Q = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
# NED_ENU_Q = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
NED_ENU_Q = [0, np.sqrt(0.5), np.sqrt(0.5), 0]

# NED_ENU_Q = quaternion_from_euler(np.pi, 0, np.pi/2)

def ned_to_enu_orientation(q):
    """Convert NED orientation to ENU orientation.

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Quaternion [w, x, y, z].
    """
    return x2w(tft.quaternion_multiply(w2x(NED_ENU_Q), w2x(q)))

def ned_to_enu_position(p):
    """Convert ned to enu using transform matrix"""
    # Directly convert x,y, z to y, x, -z
    return p[1], p[0], -p[2]

def ned_to_enu_z(z):
    """Convert ned to enu position z"""
    return -z

def ned_euler_to_enu_euler(roll, pitch, yaw):
    """Convert NED euler angles to ENU euler angles.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        list: Euler angles [roll, pitch, yaw].
    """
    q = quaternion_from_euler(roll, pitch, yaw)
    q = ned_to_enu_orientation(q)
    return quaternion_to_euler(q)

enu_to_ned_position = ned_to_enu_position
enu_to_ned_z = ned_to_enu_z
enu_to_ned_orientation = ned_to_enu_orientation
enu_euler_to_ned_euler = ned_euler_to_enu_euler

def quatmult( a, b):
    """Multiply two quaternions.
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2]

# Main funcion for testing
if __name__ == '__main__':
    # Define NED directions with names
    NED = {
        'north': [1, 0, 0],
        'south': [-1, 0, 0],
        'east': [0, 1, 0],
        'west': [0, -1, 0],
        'up': [0, 0, -1],
        'down': [0, 0, 1],
    }
    # Define ENU directions with names
    ENU = {
        'north': [0, 1, 0],
        'south': [0, -1, 0],
        'east': [1, 0, 0],
        'west': [-1, 0, 0],
        'up': [0, 0, 1],
        'down': [0, 0, -1],
    }

    def l2norm(vec: np.ndarray):
        """Calculate the L2 norm, Euclidean distance, of a nx1 vector"""
        return np.sqrt(sum([_**2 for _ in vec]))
    
    for name, ned in NED.items():
        # Convert NED direction to ENU direction
        enu = ned_to_enu_position(ned)
        # Calculate the distance between the two directions
        dist = l2norm(np.array(enu) - np.array(ENU[name]))
        # Print the distance
        print(f'{name}: {dist}')
    
    print('---------------------------------')
    
    # A random quaternion
    # qrand = x2w(tft.random_quaternion())
    # print(ned_to_enu_orientation(qrand))
    # for name, ned in NED.items():
    #     # Convert NED direction to ENU direction
    #     ned_rot = np.matmul(tft.quaternion_matrix(w2x(qrand))[:3, :3], ned)
    #     qrand_enu = ned_to_enu_orientation(qrand)
    #     enu_rot = np.matmul(tft.quaternion_matrix(w2x(qrand_enu))[:3, :3], ENU[name])
    #     # Calculate the distance between the two directions
    #     dist = l2norm(np.array(ned_rot) - np.array(enu_rot))
    #     # Print the distance
    #     print(f'{name}: {dist}')
    # qrand = [0.6794157028198242, 0.010560745373368263, -0.009150485508143902, 0.7336205840110779]
    qrand =  quaternion_from_euler(0,0, np.pi)# [1, 0, 0, 0]
    print(qrand)
    tran = ned_to_enu_orientation(qrand)
    tran_in = ned_to_enu_orientation(tran)
    print(tran,quaternion_to_euler(tran), tran_in, qrand)
    print(ned_to_enu_orientation(NED_ENU_Q))
    print(quatmult(NED_ENU_Q, qrand))
    print(ned_to_enu_orientation(quaternion_from_euler(0,0,0)))
    print(ned_to_enu_orientation(qrand))
    print(quaternion_from_euler(0,0,0))
    # print(quaternion_from_euler(*quaternion_to_euler(qrand)))
    print(quaternion_to_euler(qrand))
    print(quaternion_to_euler(ned_to_enu_orientation(qrand)))

    q2 = quaternion_from_euler(0,0, np.pi/2)
    print(quaternion_to_euler(ned_to_enu_orientation(q2)))
    print(quaternion_to_euler(quatmult(q2, NED_ENU_Q)))
    
    print('---------------------------------')

    
    

    


