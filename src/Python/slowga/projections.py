import numpy as np
import matplotlib.pyplot as plt
from .helper import normalize_box

def convert_phi_theta(normals, top_half=True):
    phi_theta = np.zeros((normals.shape[0], 2))
    xy = normals[:, 0]**2 + normals[:, 1]**2
    phi_theta[:, 0] = np.arctan2(np.sqrt(xy), normals[:, 2])  # for elevation angle defined from Z-axis down
    phi_theta[:, 1] = np.arctan2(normals[:, 1], normals[:, 0])
    return phi_theta


def convert_lat_long(normals, degrees=True, return_mask=True):
    normals_new = np.copy(normals)
    phi_theta = np.zeros((normals_new.shape[0], 2))
    phi_theta[:, 0] = np.arccos(normals_new[:, 2])  # for elevation angle defined from Z-axis down
    phi_theta[:, 1] = np.arctan2(normals_new[:, 1], normals_new[:, 0])
    phi_theta[:, 0] = np.ones_like(phi_theta[:, 0]) * np.pi/2.0 - phi_theta[:, 0]
    if degrees:
        phi_theta[:, 0] = np.rad2deg(phi_theta[:, 0])
        phi_theta[:, 1] = np.rad2deg(phi_theta[:, 1])

    return phi_theta

def convert_stereographic(normals, top_half=True):
    normals_new = np.copy(normals)
    projection = np.zeros((normals_new.shape[0], 2))
    projection[:, 0] = normals_new[:, 0] / (1 - normals_new[:, 2])
    projection[:, 1] = normals_new[:, 1] / (1 - normals_new[:, 2])
    return projection


def convert_phi_theta_centered(normals, top_half=True):
    normals_new = np.copy(normals)
    projection = np.zeros((normals_new.shape[0], 2))
    xy = normals_new[:, 0]**2 + normals_new[:, 1]**2
    for i in range(normals_new.shape[0]):
        normal = normals_new[i, :]
        phi = np.arccos(normal[2])
        phi = -phi if normal[1] < 0 else phi
        theta = np.arctan2(normal[1], normal[0])
        theta = theta + np.pi / 2.0
        theta = theta - np.pi if theta > np.pi / 2.0 else theta
        projection[i, :] = [phi, theta]

    return projection


def down_proj(normals, top_half=True):
    normals_new = np.copy(normals)
    projection = np.zeros((normals_new.shape[0], 2))
    projection = normals_new[:, :2]

    return projection


def azimuth_equidistant(normals):
    normals_new = np.copy(normals)
    projection = np.zeros((normals_new.shape[0], 2))
    theta = np.arctan2(normals_new[:, 1], normals_new[:, 0])
    phi = np.zeros_like(theta)
    for i in range(normals_new.shape[0]):
        normal = normals_new[i, :]
        phi_ = np.arccos(normal[2])
        # phi_ = -phi_ if normal[1] < 0 else phi_
        phi[i] = phi_

    projection[:, 0] = phi * np.sin(theta)
    projection[:, 1] = - phi * np.cos(theta)
    projection = normalize_box(projection)
    # TODO For some reason not exactly the same
    # projection[:, 0] = phi * normals_new[:, 1]
    # projection[:, 1] =  -phi * normals_new[:, 0]

    return projection


def plot_projection(ga):

    projections = [("Spherical Coordinates", "phi", "theta", "convert_phi_theta"),
                   ("Spherical Coordinates Centered", "phi", "theta", "convert_phi_theta_centered"),
                   ("Geodetic", "lat", "lon", "convert_lat_long"),
                   ("Steographic Projection", "x*", "y*", "convert_stereographic"),
                   ("Project To Plane", "x", "y", "down_proj"),
                   ("Azimuth Equidistant", "x*", "y*", "azimuth_equidistant"),
                   ]
    fig, axs = plt.subplots(3, 2, figsize=(5, 6))
    axs = axs.reshape(-1)
    for i, (title_name, xlabel, ylabel, function_name) in enumerate(projections):
        ax = axs[i]
        proj = globals()[function_name](ga.gaussian_normals)
        ax.scatter(proj[:, 0], proj[:, 1], c=ga.colors)
        ax.set_title(title_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis('equal')
    fig.tight_layout()

    plt.show()