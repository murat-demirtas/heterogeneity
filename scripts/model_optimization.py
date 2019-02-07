import os
import sys
from hbnm.io import Data
from hbnm.model.utils import subdiag, fisher_z
from scipy.stats import pearsonr
from optimization import load_data

if __name__ == '__main__':
    """
    arguments:
    1- model type: heterogeneous or homogeneous
    2- minimum number of samples / sampler
    3- number of samplers
    4- sampler id
    5- optimization task: sampler or wrapper
    6- append to output directory
    """

    model_type = sys.argv[1]
    n_samples = int(sys.argv[2])
    n_samplers = int(sys.argv[3])
    sampler_id = int(sys.argv[4])
    task = sys.argv[5]
    append_directory = sys.argv[6]

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

    fc_obj = fisher_z(subdiag(fc_obj))
    rejection_threshold = 1.0 - pearsonr(fc_obj, subdiag(sc))[0]

    if model_type == 'homogeneous':
        from optimization import Homogeneous

        pmc_opt = Homogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, gradient=None, n_particles=n_samples,
                           rejection_threshold=rejection_threshold, norm_sc=True)
    else:
        from optimization import Heterogeneous

        pmc_opt = Heterogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, gradient=hmap, n_particles=n_samples,
                           rejection_threshold=rejection_threshold, norm_sc=True)

    if task == 'sampler':
        pmc_opt.run(sampler_id)
    elif task == 'wrapper':
        pmc_opt.wrap(n_samplers)
    else:
        raise NotImplementedError("The task should be sampler or wrapper")
