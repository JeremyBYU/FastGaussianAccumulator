import numpy as np

from fastga import convert_normals_to_hilbert, GaussianAccumulator

def main():
    normals = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ])
    # normals_flat = normals.reshape((9,))
    # proj, hilbert_values = convert_normals_to_hilbert(normals)
    # print(proj)
    # print(hilbert_values)

    ga = GaussianAccumulator(1)
    # print(ga)
    
    vertices = np.asarray(ga.mesh.vertices)
    triangles = np.asarray(ga.mesh.triangles)
    print(vertices)
    print(triangles)
    print(vertices.shape)
    print(triangles.shape)

if __name__ == "__main__":
    main()