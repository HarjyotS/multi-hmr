import numpy as np


def find_plane_equation(p1, p2, p3):
    # Convert points to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    # Calculate vectors from p1 to p2 and from p1 to p3
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the normal vector by taking the cross product of v1 and v2
    normal_vector = np.cross(v1, v2)

    # Plane equation: ax + by + cz = d
    # We can find d by plugging in one of the points into the equation
    a, b, c = normal_vector
    d = np.dot(normal_vector, p1)

    return a, b, c, d


def find_midpoint(p1, p2, p3):
    # Calculate the midpoint of three points
    return (np.array(p1) + np.array(p2) + np.array(p3)) / 3


def find_normal_line_equation(point, normal_vector):
    # The parametric equation of the line passing through 'point' with direction 'normal_vector' is:
    # x = x0 + at
    # y = y0 + bt
    # z = z0 + ct
    # where (x0, y0, z0) is 'point' and (a, b, c) is 'normal_vector'
    return point, normal_vector


# Define the three points
p1 = [0.23908446729183197, -0.5658280253410339, 2.0103042125701904]
p2 = [0.17826293408870697, -0.5687577128410339, 2.0116469860076904]
p3 = [0.19733642041683197, -0.45205849409103394, 2.0077407360076904]

# Find the plane equation
a, b, c, d = find_plane_equation(p1, p2, p3)
print(f"The equation of the plane is: {a}x + {b}y + {c}z = {d}")

# Find the midpoint
midpoint = find_midpoint(p1, p2, p3)
print(f"The midpoint of the points is: {midpoint}")

# Find the normal line equation
normal_vector = np.array([a, b, c])
point_on_line, direction_vector = find_normal_line_equation(midpoint, normal_vector)
print(
    f"The normal line passing through the midpoint has the direction vector: {direction_vector} and passes through the point: {point_on_line}"
)
