""" Example showing two different forms of space filling curves
"""
import numpy as np

from fastgac import convert_normals_to_s2id, convert_normals_to_hilbert ,GaussianAccumulatorOpt, MatX3d

def main():
    normals = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ])
    ga = GaussianAccumulatorOpt(level=4)
    proj, hilbert_values = convert_normals_to_hilbert(MatX3d(normals), ga.projected_bbox)
    s2_values = convert_normals_to_s2id(MatX3d(normals))
    print("Normals: ")
    print(normals)
    print("Transformed to uint32 hilbert values (azimuth equal area projection at northpole):")
    print(hilbert_values)
    print("S2ID. Cubic Projection on sphere. Each face has its own hilbert value computed:")
    print(np.asarray(s2_values))



if __name__ == "__main__":
    main()