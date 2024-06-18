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

    # Return the plane equation coefficients
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
