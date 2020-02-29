import numpy as np

from fastga import convert_normals_to_hilbert, GaussianAccumulatorKD, MatX3d

def main():
    normals = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ])

    ga = GaussianAccumulatorKD(0, 100)
    print("Number of Buckets: {}".format(len(ga.buckets)))
    print(ga.buckets)
    converted = MatX3d(normals)
    print(np.asarray(converted))
    bucket_indexes = ga.get_bucket_indexes(converted)
    print(np.asarray(bucket_indexes))
    

if __name__ == "__main__":
    main()