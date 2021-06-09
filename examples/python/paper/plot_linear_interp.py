import numpy as np
import math
from scipy import stats
from fastgac import GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
# import tikzplotlib


# sns.set()
sns.set(font_scale=1.1, style='white')  # crazy big

def set_attributes(axes, xlabel='Array Index', ylabel='Unique ID (Hilbert Value)'):
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left')

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def plot_indices(ga_s2):


    indices_s2 = np.asarray(ga_s2.get_bucket_sfc_values())
    indices_s2 = indices_s2 - np.min(indices_s2)
    fig0, ax0 = plt.subplots(1, 1, figsize=(4.5, 4))
    fig1, ax1 = plt.subplots(1, 1, figsize=(4.5, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 4))


    bucket_positions_s2 = np.arange(indices_s2.shape[0])
    ax0.plot(indices_s2, bucket_positions_s2 ,'.', markersize=2, label='Cells')


    
    ax1.plot(indices_s2, bucket_positions_s2 ,'.', markersize=2, label='Cells')
    predicted, r_value, error = calc_linear_regression(bucket_positions_s2, indices_s2, ax1)
    # ax[1].legend(loc='upper left')

    ax2.plot(indices_s2, bucket_positions_s2 ,'.', markersize=2, label='Cells')
    ax2.plot(indices_s2, predicted, 'g', label=r'Fitted Line, $R^2$ = {:.1}'.format(r_value))
    ax3 = ax2.twinx()
    ax3.set_ylabel(r"Error")
    ax3.plot(indices_s2, error, '-r', label="Error")
    # Dummy line for legend
    ax2.plot(np.nan, '-r', label='Error')
        

    set_attributes([ax0, ax1, ax2], ylabel='Cells Array Index', xlabel='S2ID')
    ax2.set_xlim([6.2e18, 7.2e18])
    ax2.set_ylim([2300, 2700])
    fig0.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()

    fig0.savefig('assets/imgs/ga_linear_interp_0.pdf', bbox_inches='tight')
    fig1.savefig('assets/imgs/ga_linear_interp_1.pdf', bbox_inches='tight')
    fig2.savefig('assets/imgs/ga_linear_interp_2.pdf', bbox_inches='tight')


    # tikzplotlib.save("assets/imgs/ga_linear_interp_0.tex", figure=fig0)

    # extent = full_extent(ax[0]).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('assets/imgs/ga_linear_interp_0.pdf', bbox_inches=extent.expanded(1.0, 1.0))

    # extent = full_extent(ax[1]).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('assets/imgs/ga_linear_interp_1.pdf', bbox_inches=extent.expanded(1.0, 1.0))

    # items = [ax2.get_xaxis().get_label().get_window_extent(), ax2.get_yaxis().get_label().get_window_extent()]
    # extent = Bbox.union([full_extent(ax[2]), *items]).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('assets/imgs/ga_linear_interp_2.pdf', bbox_inches=extent.expanded(1.0, 1.0))


    # plt.show()

def calc_linear_regression(bucket_positions, indices, ax):
    max_hv = np.max(indices)
    min_hv = np.min(indices)
    print("max, min", max_hv, min_hv)
    slope, intercept, r_value, p_value, std_err = stats.linregress(indices, bucket_positions)
    print("Slope and Intercept", slope, intercept)
    predicted = intercept + slope*indices
    ax.plot(indices, predicted, 'g', label=r'Fitted Line, $R^2$ = {:.1}'.format(r_value))
    error = predicted - bucket_positions

    min_val = math.floor(np.min(error))
    max_val = math.ceil(np.max(error))
    print("Min Index Error: {}; Max Index Error: {}".format(min_val, max_val))
    return predicted, r_value, error

def main():
    kwargs = dict(level=4, max_phi=180)
    ga_s2 = GaussianAccumulatorS2(**kwargs)

    plot_indices(ga_s2)


if __name__ == "__main__":
    main()