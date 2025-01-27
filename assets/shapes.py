import torch
import numpy as np
from .svg_extract_xy import svg_extract_xy

def square(num_points):
    num_points_per_edge = num_points // 4
    # Generate points for each edge, including one corner point per edge
    bottom = torch.stack([torch.linspace(-1, 1, num_points_per_edge + 1), -torch.ones(num_points_per_edge + 1)], dim=-1)[:-1]
    right = torch.stack([torch.ones(num_points_per_edge + 1), torch.linspace(-1, 1, num_points_per_edge + 1)], dim=-1)[:-1]
    top = torch.stack([torch.linspace(1, -1, num_points_per_edge + 1), torch.ones(num_points_per_edge + 1)], dim=-1)[:-1]
    left = torch.stack([-torch.ones(num_points_per_edge + 1), torch.linspace(1, -1, num_points_per_edge + 1)], dim=-1)[:-1]

    # Concatenate all edges in order
    X = torch.cat([bottom, right, top, left], dim=0)
    X = torch.vstack([X, X[0]])
    X = torch.cat((X[:1], X[2:]), dim=0)
    return X


def circle(num_points, radius = 1.0):
    theta = torch.linspace(0, 2 * torch.pi, num_points)
    
    # Compute x and y coordinates
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    
    # Stack x and y to form points
    circle_points = torch.stack([x, y], dim=-1)
    return circle_points


def stanford_bunny(num_points):
    X = svg_extract_xy('./assets/stanford_bunny.svg', num_points = num_points)
    X = torch.from_numpy(X).to(torch.float32)
    return X

def heart(num_points):
    X = svg_extract_xy('./assets/images/heart.svg', num_points = num_points)
    X = torch.from_numpy(X).to(torch.float32)
    return X

def hand(num_points):
    X = svg_extract_xy('./assets/images/hand.svg', num_points = num_points)
    X = torch.from_numpy(X).to(torch.float32)
    return X

def airplane(num_points):
    X = svg_extract_xy('./assets/images/airplane.svg', num_points = num_points)
    X = torch.from_numpy(X).to(torch.float32)
    return X

def puzzle_piece(num_points):
    X = svg_extract_xy('./assets/images/puzzle_piece.svg', num_points = num_points)
    X = torch.from_numpy(X).to(torch.float32)
    return X


def get_square_polar_radius(phi):
    """
    Compute the polar radius for a square given an angle phi.
    Equivalent to the C++ function provided.
    """
    # Normalize the angle to the range [-pi/4, pi/4]
    phi_in_pi_by_4_range = phi
    while phi_in_pi_by_4_range > torch.pi / 4:
        phi_in_pi_by_4_range -= torch.pi / 2
    while phi_in_pi_by_4_range < -torch.pi / 4:
        phi_in_pi_by_4_range += torch.pi / 2

    # Calculate the radius
    return 1 / torch.cos(phi_in_pi_by_4_range)


def generate_square_points_polar(num_points):
    """
    Generate points on a square in polar coordinates using the get_square_polar_radius function.
    """
    # Define theta values uniformly from 0 to 2*pi
    theta = torch.linspace(0, 2 * torch.pi, num_points)

    # Calculate radii for each theta
    r = torch.tensor([get_square_polar_radius(t) for t in theta])

    # Convert polar coordinates to Cartesian
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.column_stack((x, y))


def square_from_t(t):
    num_points_per_edge = t.shape[0] // 4
    # Generate points for each edge, including one corner point per edge
    bottom = torch.stack([torch.linspace(-1, 1, num_points_per_edge + 1), -torch.ones(num_points_per_edge + 1)], dim=-1)[:-1]
    right = torch.stack([torch.ones(num_points_per_edge + 1), torch.linspace(-1, 1, num_points_per_edge + 1)], dim=-1)[:-1]
    top = torch.stack([torch.linspace(1, -1, num_points_per_edge + 1), torch.ones(num_points_per_edge + 1)], dim=-1)[:-1]
    left = torch.stack([-torch.ones(num_points_per_edge + 1), torch.linspace(1, -1, num_points_per_edge + 1)], dim=-1)[:-1]

    # Concatenate all edges in order
    X = torch.cat([bottom, right, top, left], dim=0)
    X = torch.vstack([X, X[0]])
    X = torch.cat((X[:1], X[2:]), dim=0)
    return X