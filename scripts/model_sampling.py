import numpy as np
import os
from hbnm.io import Data
from hbnm.bnm import Bnm
from hbnm.model.utils import subdiag, fisher_z
from scipy.spatial.distance import squareform
from optimization import load_data

def matrix_plot(ax, x, cmap):
    ax.pcolormesh(x, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_aspect(1)
    # set axis limits to fit data
    ax.set_xlim([0, x.shape[1]])
    ax.set_ylim([0, x.shape[0]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def reg_plot(ax, x, y):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.scatter(x, y)
    ax.set_xlabel('model FC - z-transformed')
    ax.set_ylabel('empirical FC - z-transformed')
    text = 'r = ' + '{:3.2f}'.format(np.corrcoef(x, y)[0,1])
    ax.text(0.5, 0.95, text, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)

"""
Set directories
"""
current_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
input_dir = parent_path + '/data/'
output_dir = parent_path + '/outputs/'

"""
Load data
"""
data = Data(input_dir, output_dir)
sc, hmap, fc_obj = load_data(data)

fin = data.load('demirtas_neuron_2019.hdf5', from_output=False)
theta_heterogeneous = fin['apprx_posterior_heterogeneous'].value
theta_homogeneous = fin['apprx_posterior_homogeneous'].value
fin.close()

"""
Set model samples
"""
homogeneous = Bnm(sc)
homogeneous.set('w_EI', theta_homogeneous[0,0])
homogeneous.set('w_EE', theta_homogeneous[1,0])
homogeneous.set('G', theta_homogeneous[2,0])
homogeneous.moments_method()

heterogeneous = Bnm(sc, gradient=hmap)
heterogeneous.set('w_EI', (theta_heterogeneous[0,0], theta_heterogeneous[1,0]))
heterogeneous.set('w_EE', (theta_heterogeneous[2,0], theta_heterogeneous[3,0]))
heterogeneous.set('G', theta_heterogeneous[4,0])
heterogeneous.moments_method()

from scipy.stats import pearsonr
print pearsonr(subdiag(fc_obj), subdiag(heterogeneous.get('corr_bold')))

"""
Plot
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16.0
mpl.rcParams['axes.labelsize'] = 26.0
mpl.rcParams['axes.titlesize'] = 26.0
mpl.rcParams['xtick.labelsize'] = 20.0
mpl.rcParams['ytick.labelsize'] = 20.0

fig, axes = plt.subplots(2,3,figsize=(30, 15))

#import pdb; pdb.set_trace()
matrix_plot(axes[0][0], fc_obj, 'RdBu_r')
axes[0][0].set_title('Empirical FC')
matrix_plot(axes[0][1], homogeneous.get('corr_bold'), 'RdBu_r')
axes[0][1].set_title('Homogeneous model FC')
matrix_plot(axes[0][2], heterogeneous.get('corr_bold'), 'RdBu_r')
axes[0][2].set_title('Heterogeneous model FC')

reg_plot(axes[1][0], fisher_z(subdiag(homogeneous.get('corr_bold'))), fisher_z(subdiag(fc_obj)))
axes[1][0].set_title('Homogeneous model FC fit')
reg_plot(axes[1][1], fisher_z(subdiag(heterogeneous.get('corr_bold'))), fisher_z(subdiag(fc_obj)))
axes[1][1].set_title('Heterogeneous model FC fit')

axes[1][2].bar(np.arange(5), theta_heterogeneous.mean(1)[:5])
axes[1][2].set_title('Optimal model parameters')
axes[1][2].set_xticks(np.arange(5)+0.5)
axes[1][2].set_xticklabels(['wEI min', 'wEI scale', 'wEE min', 'wEE scale', 'g'], minor=False)
plt.tight_layout()

plt.show()
plt.savefig('Example_model_fit.png', dpi=100)
