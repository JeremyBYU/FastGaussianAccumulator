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


if __name__ == "__main__":
    main()