from hbnm.pmc import Pmc
from scipy import stats
from hbnm.model.utils import subdiag, linearize_map, normalize_sc, fisher_z
from scipy.stats import pearsonr

class Homogeneous(Pmc):
    def get_appendices(self, run_id):
        return

    def set_prior(self):
        self.prior = [stats.uniform(0.001, 5.0),
                      stats.uniform(0.001, 15.0),
                      stats.uniform(0.001, 10.0)]

    def run_particle(self, theta):
        self.model.set('w_EI', theta[0])
        self.model.set('w_EE', theta[1])
        self.model.set('G', theta[2])

    def generate_data(self):
        self.model.moments_method(BOLD=True)
        return subdiag(self.model.get('corr_bold'))

    def distance_function(self, synthetic_data):
        fit = pearsonr(self.fc_objective, fisher_z(synthetic_data))[0]
        penalty = (self.fc_objective.mean() - synthetic_data.mean()) ** 2
        distance = 1.0 - (fit - penalty)
        return distance

class Heterogeneous(Pmc):
    def get_appendices(self, run_id):
        return

    def set_prior(self):
        self.prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5),
                      stats.uniform(0.001, 5.0), stats.uniform(0.0, 15.0),
                      stats.uniform(0.001, 5.0)]

    def run_particle(self, theta):
        self.model.set('w_EI', (theta[0], theta[1]))
        self.model.set('w_EE', (theta[2], theta[3]))
        self.model.set('G', theta[4])

    def generate_data(self):
        self.model.moments_method(BOLD=True)
        return subdiag(self.model.get('corr_bold'))

    def distance_function(self, synthetic_data):
        fit = pearsonr(self.fc_objective, fisher_z(synthetic_data))[0]
        penalty = (self.fc_objective.mean() - synthetic_data.mean()) ** 2
        distance = 1.0 - (fit - penalty)
        return distance

def load_data(data):
    fin = data.load('demirtas_neuron_2019.hdf5')
    sc = fin['sc'].value
    fc = fin['fc'].value
    t1t2 = fin['t1wt2w'].value
    fin.close()

    # For left hemisphere, use first 180 indices
    sc = normalize_sc(sc[:180,:180])
    fc_obj = fc[:180,:180]
    hmap = linearize_map(t1t2[:180])

    return sc, hmap, fc_obj
