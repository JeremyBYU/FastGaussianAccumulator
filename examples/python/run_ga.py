"""This example is a simply demo with no visualization of integrating normals into level 0 GA
"""
import numpy as np
import time 

from fastgac import GaussianAccumulatorKD, MatX3d

np.set_printoptions(suppress=True, precision=2)

def main():
    normals = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ])
    t0 = time.perf_counter()
    ga = GaussianAccumulatorKD(level=0, max_phi=180)
    t1 = time.perf_counter()
    print("Number of Buckets: {}\n".format(len(ga.buckets)))
    print("Bucket representations:\n {}\n".format(ga.buckets))
    print("Bucket Cell Surface Normals: \n {}\n".format(np.asarray(ga.get_bucket_normals())))
    normals_mat = MatX3d(normals) # need to convert a format we understand
    t2 = time.perf_counter()
    bucket_indexes = ga.integrate(normals_mat)
    t3 = time.perf_counter()
    print("These normals: \n {} \n are most similar to these cell normlas: \n {} \n".format(normals, np.asarray(ga.get_bucket_normals())[bucket_indexes,:]))
    print(np.asarray(bucket_indexes))
    print("Building Index Took (ms): {}; Query Took (ms): {}".format((t1-t0) * 1000, (t3 - t2)* 1000))

    print("Change the level see a better approximation")
    

if __name__ == "__main__":
    main()