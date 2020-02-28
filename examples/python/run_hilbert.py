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
    proj, hilbert_values = convert_normals_to_hilbert(normals)
    print(np.asarray(proj))
    print(hilbert_values)

    ga = GaussianAccumulator(4)
    print("Number of Buckets: {}".format(len(ga.buckets)))
    print(ga.buckets)
    
    vertices = np.asarray(ga.mesh.vertices)
    triangles = np.asarray(ga.mesh.triangles)
    # print(vertices)
    # print(triangles)
    # print(vertices.shape)
    # print(triangles.shape)

if __name__ == "__main__":
    main()