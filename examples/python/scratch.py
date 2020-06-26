import numpy as np
import math
from scipy import stats
from fastga import GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d
import matplotlib.pyplot as plt

def set_attributes(axes, xlabel='Array Index', ylabel='Unique ID (Hilbert Value)'):
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

def plot_indices(ga_opt, ga_s2):
    indices_opt = np.asarray(ga_opt.get_bucket_sfc_values())
    indices_opt = indices_opt - np.min(indices_opt)
    indices_s2 = np.asarray(ga_s2.get_bucket_sfc_values())
    indices_s2 = indices_s2 - np.min(indices_s2)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0,0].plot(indices_opt, '.',markersize=2, label='Azimuth Projection')
    ax[0,1].plot(indices_s2, '.', markersize=2, label='S2 Projection')
    bucket_positions_opt = np.arange(indices_opt.shape[0])
    bucket_positions_s2 = np.arange(indices_s2.shape[0])
    
    ax[1,0].plot(indices_opt, bucket_positions_opt, '.',markersize=2, label='Azimuth Projection')
    ax[1,1].plot(indices_s2, bucket_positions_s2 ,'.', markersize=2, label='S2 Projection')

    calc_linear_regression(bucket_positions_opt, indices_opt, ax[1, 0])
    calc_linear_regression(bucket_positions_s2, indices_s2, ax[1, 1])
        
    set_attributes([ax[0,0], ax[0, 1]], xlabel='Array Index', ylabel='Unique ID (S2 ID)')
    set_attributes([ax[1,0], ax[1, 1]], ylabel='Array Index', xlabel='Unique ID (S2 ID)')
    fig.tight_layout()
    plt.show()

def calc_linear_regression(bucket_positions, indices, ax):
    max_hv = np.max(indices)
    min_hv = np.min(indices)
    print("max, min", max_hv, min_hv)
    slope, intercept, r_value, p_value, std_err = stats.linregress(indices, bucket_positions)
    print("Slope and Intercept", slope, intercept)
    predicted = intercept + slope*indices
    ax.plot(indices, predicted, 'g', label=r'Fitted Line, $R^2$ = {:.1}'.format(r_value))
    # Plot Error
    error = predicted - bucket_positions
    ax2 = ax.twinx()
    ax2.set_ylabel(r"Error")
    ax2.plot(indices, error, '-r', label="Error")
    # Dummy line for legend
    ax.plot(np.nan, '-r', label='Error')
    min_val = math.floor(np.min(error))
    max_val = math.ceil(np.max(error))
    print("Min Index Error: {}; Max Index Error: {}".format(min_val, max_val))

def main():
    kwargs = dict(level=4, max_phi=180)
    ga_opt = GaussianAccumulatorOpt(**kwargs)
    ga_s2 = GaussianAccumulatorS2(**kwargs)

    plot_indices(ga_opt, ga_s2)


if __name__ == "__main__":
    main()