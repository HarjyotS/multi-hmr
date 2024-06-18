import numpy as np

def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    # Create vectors from points
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1

    # Calculate the normal vector to the plane
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2

    # Calculate the plane constant D
    d = -(a * x1 + b * y1 + c * z1)

    return (a, b, c, d)


# function that gives the midpoint of three points in 3D space
def midpoint(p1, p2, p3):

    return [
        (p1[0] + p2[0] + p3[0]) / 3,
        (p1[1] + p2[1] + p3[1]) / 3,
        (p1[2] + p2[2] + p3[2]) / 3,
    ]


def normal_line_to_plane(A, B, C, D, x0, y0, z0):
    # Normal vector (A, B, C)

    # Parametric equations of the line:
    # x = x0 + A*t
    # y = y0 + B*t
    # z = z0 + C*t

    # Convert to the desired format:
    normal_line_equation = f"(x-{x0})/{A} = (y-{y0})/{B} = (z-{z0})/{C}"

    return normal_line_equation

def frustum_equation(a, b, c, d, P1, P2, P3, base_radius, lateral_slope):
    # Normalize the plane normal vector
    normal = np.array([a, b, c])
    norm_factor = np.linalg.norm(normal)
    normal = normal / norm_factor  # Normalize the normal vector

    # Calculate the frustum center (midpoint between P1 and P2)
    P1, P2, P3 = np.array(P1), np.array(P2), np.array(P3)
    C = (P1 + P2) / 2

    # Basis vectors on the plane
    B1 = P2 - P1
    B1 = B1 / np.linalg.norm(B1)
    B2 = np.cross(normal, B1)

    # Center coordinates
    Cx, Cy, Cz = C
    # Basis vectors components
    B1x, B1y, B1z = B1
    B2x, B2y, B2z = B2

    # Distance from point to plane
    h = f"({a}*x + {b}*y + {c}*z + {d}) / {norm_factor}"
    # Variable radius based on lateral slope
    radius = f"{base_radius} + {lateral_slope} * abs({h})"
    
    # Distances in plane basis
    d1 = f"({B1x}*(x - {Cx}) + {B1y}*(y - {Cy}) + {B1z}*(z - {Cz}))"
    d2 = f"({B2x}*(x - {Cx}) + {B2y}*(y - {Cy}) + {B2z}*(z - {Cz}))"
    
    # Frustum equation
    frustum_eq = f"(({d1})^2 + ({d2})^2) - ({radius})^2 = 0"
    
    return frustum_eq

def point_in_frustum(a, b, c, d, P1, P2, P3, radius, lateral_slope, Q):
    # Define the plane normal from the plane equation ax + by + cz + d = 0
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    # Calculate the frustum center (midpoint between P1 and P2)
    P1, P2, P3 = np.array(P1), np.array(P2), np.array(P3)
    C = (P1 + P2) / 2

    # Project point Q onto the plane
    Q = np.array(Q)
    Q_proj = Q - (np.dot(Q - C, normal)) * normal

    # Calculate vector from frustum center to projected point
    V = Q_proj - C

    # Basis vectors on the plane
    B1 = P2 - P1
    B1 = B1 / np.linalg.norm(B1)
    B2 = np.cross(normal, B1)

    # Transform vector V into the basis (B1, B2)
    V_b = np.array([np.dot(V, B1), np.dot(V, B2)])

    # Calculate the distance from the center in the plane's basis
    distance_in_plane = np.linalg.norm(V_b)

    # Check if the point is within the frustum radius
    return distance_in_plane <= radius
