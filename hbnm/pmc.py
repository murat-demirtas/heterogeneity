import numpy as np
from io import Data
from bnm import Bnm
from abc import ABCMeta, abstractmethod
from scipy import stats
from numpy.testing import assert_almost_equal
import os

class Pmc(object):
    """
    Class for particle monte carlo optimization
    
    This class is derived from the Python package SimpleABC:
        A Python package for Approximate Bayesian Computation
        Version 0.2.0
        
        Available in http://rcmorehead.github.io/SIMPLE-ABC/
        Sunnaker et al. - [Approximate Bayesian Computation](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3547661/)
    """
    __metaclass__ = ABCMeta
    def __init__(self, input_directory, output_directory, verbose=True):
        """
        Parameters
        ----------
        input_directory : str
            The input directory for the data
        output_directory : str
            The output directory for the results
        
        """
        self.data = Data(input_directory, output_directory)
        self.verbose = verbose
        self._check_iteration_n()
        self.set_prior()

    @abstractmethod
    def get_appendices(self, run_id):
        """
        An abstract method to provide additional parameters to save
        """

    @abstractmethod
    def set_prior(self):
        """
        Provide prior distributions for the parameters.
        This method requires defining prior such as: 
        e.g. self.prior = [stats.uniform(0.001, 2.), stats.uniform(0.0, 2.5)]...etc.
        """

    @abstractmethod
    def run_particle(self, theta):
        """
        An abstract method to draw a particle. An example use would be:
            self.model.set('w_EI',theta[0])
            self.model.set('w_EE',theta[1])
            ...etc.
         
        """

    @abstractmethod
    def generate_data(self):
        """
        An abstract method to execute the linearization and generate
        model FC. 
        For example:
        self.model.moments_method(BOLD=True)
        return self.model.get('corr_bold')
        
        Returns
        -------
        ndarray
            The model FC measure
        """

    @abstractmethod
    def distance_function(self, synthetic_data):
        """
        An abstract method to calculate distance. For example:
            model_fit = pearsonr(self.fc_objective, synthetic_data)[0]
            return 1.0 - model_fit
        
        Returns
        -------
        float
            Summary distance statistic 
        """

    def initialize(self, sc, fc=None, gradient=None,
            n_particles=10, rejection_threshold=None,
            *args, **kwargs):
        """
        Initialization for the optimization.
        
        Parameters
        ----------
        sc : ndarray or list
            Empirical structural connectivity matrix
        fc : ndarray
            Empirical functional connectivity
        gradient : ndarray or list
            Heterogeneity map to parametrize the model
        n_particles : int
            Maximum number of particles
        rejection_threshold : float
            Initial rejection threshold
        
        Notes
        -----
        This method requires a list for SC and heterogeneity map, if the model
        will be fitted for left and right hemispheres separately. The dimensions
        of the empirical functional connectivity should be (N_connextions x N_subjects),
        where N_connections is the number of connections, i.e. N x (N-1)/2, and N_subjects
        is the number of subjects.
        """
        self.model = Bnm(sc, gradient = gradient, *args, **kwargs)
        self.fc_objective = fc
        self.n_particles = n_particles
        self.rejection_threshold = rejection_threshold

    def run(self, run_id):
        """
        Run a single iteration for the particle Monte Carlo algorithm
        
        Parameters
        ----------
        run_id : int
            The labels for the run (required for paralelization)
        
        Notes
        -----
        This method executes a single iteration for the PMC. For paralelize the code, it samples
        multiple batches of particles independently and saves the output to a file called, 'samples_n.npy'.
        """

        self._check_iteration_n()
        if self.iteration > 1:
            file_prev = self.data.load('iteration_' + str(self.iteration - 1) + '.hdf5',
                                       from_path=self.data.output_dir)
            theta_prev = file_prev['theta'].value
            weights_prev = file_prev['weights'].value
            tau_squared_prev = file_prev['tau_squared'].value
            epsilon = file_prev['epsilon'].value
            file_prev.close()

            results = self._run_sampler(epsilon, run_id, theta_prev=theta_prev, weights=weights_prev,
                                        tau_squared=tau_squared_prev)
        else:
            results = self._run_sampler(self.rejection_threshold, run_id)

        if self.verbose:
            print "Completed sampler# " + str(run_id) + ", writing results..."
        self.data.save('samples_' + str(run_id + 1) + '.npy', results)

    def wrap(self, n_outputs):
        """
        Wrapper function to collect the samples.
        
        Parameters
        ----------
        n_outputs : int
            Total number of samplers that are run in parallel
        
        Notes
        -----
        This method collects all the results from previously saves files ('samples_n.npy'), and then
         dumps them into a single HDF file (iteration_n.hdf5).
        """
        self._check_iteration_n()
        if self.iteration > 1:
            file_prev = self.data.load('iteration_' + str(self.iteration - 1) + '.hdf5',
                                       from_path=self.data.output_dir)
            theta_prev = file_prev['theta'].value
            weights_prev = file_prev['weights'].value
            tau_squared_prev = file_prev['tau_squared'].value
            epsilon_prev = file_prev['epsilon'].value
            file_prev.close()

        p_theta = []
        distance = np.empty(1)
        n_accepted = np.empty(n_outputs)
        n_total = np.empty(n_outputs)
        for ii in range(n_outputs):
            results = self.data.load('samples_' + str(ii + 1) + '.npy', from_path=self.data.output_dir)
            p_theta += [results[0]]
            distance = np.hstack((distance, results[1]))
            n_accepted[ii] = results[2]
            n_total[ii] = results[3]

        theta = np.hstack([p_theta[p] for p in range(n_outputs)])
        n_accepted = n_accepted.sum()
        n_total = n_total.sum()
        distance = distance[1:]

        if self.iteration == 1:
            tau_squared = 2 * np.cov(theta)
            weights = np.ones(theta.shape[1]) * 1.0 / theta.shape[1]
        else:
            weights = self._calc_weights(theta_prev, theta, tau_squared_prev, weights_prev, prior=self.prior)
            tau_squared = 2 * self._weighted_covar(theta, weights)

        epsilon_new = stats.scoreatpercentile(distance, per=75)
        if self.iteration > 1:
            if epsilon_new < epsilon_prev:
                epsilon = epsilon_new
            else:
                epsilon = epsilon_prev
        else:
            epsilon = epsilon_new

        effective_sample = self._effective_sample_size(weights)

        if self.verbose:
            print "Collecting sampler results for iteration " + str(self.iteration) + "..."

        file_out = self.data.save('iteration_' + str(self.iteration) + '.hdf5')
        file_out.create_dataset('theta', data=theta)
        file_out.create_dataset('distance', data=distance)
        file_out.create_dataset('weights', data=weights)
        file_out.create_dataset('tau_squared', data=tau_squared)
        file_out.create_dataset('epsilon', data=epsilon)
        file_out.create_dataset('ess', data=effective_sample)
        file_out.create_dataset('n_accepted', data=n_accepted.sum())
        file_out.create_dataset('n_total', data=n_total.sum())
        file_out.close()

    def _draw_theta(self, theta_prev=None, weights=None, tau_squared=None):
        """
        Generates particles (a set of parameters) based on the prior (or proposal) distribution

        Parameters
        ----------
        theta_prev : ndarray
            The particles from the previous iteration (None for the first iteration)
        weights : ndarray
            Priority weights for each particle (None for the first iterations)
        tau_squared: ndarray
            2 x the covariance matrix of the particle distribution in the previous iteration  
        """

        theta = None
        unstable = True
        if theta_prev is None:
            while unstable:
                theta = []
                for p in self.prior:
                    theta.append(p.rvs())

                if not (np.array(theta) < 0).any():
                    self.run_particle(theta)
                    unstable = self.model.check_stability()
        else:
            theta_star = theta_prev[:, np.random.choice(xrange(0, theta_prev.shape[1]),
                                                        replace=True, p=weights / weights.sum())]
            while unstable:
                theta = stats.multivariate_normal.rvs(theta_star, tau_squared)
                if not (np.array(theta) < 0).any():
                    self.run_particle(theta)
                    unstable = self.model.check_stability()
        return theta

    def _run_sampler(self, epsilon, run_id, theta_prev=None, weights=None, tau_squared=None):
        """
        Samples particles from the proposal distrubution and perform rejection sampling

        Parameters
        ----------
        epsilon : float
            The rejection threshold
        run_id : int
            The label for the run (required for paralelization)
        theta_prev : ndarray
            The particles from the previous iteration (None for the first iterations)
        weights : ndarray
            Priority weights for each particle (None for the first iterations)
        tau_squared: ndarray
            2 x the covariance matrix of the particle distribution in the previous iteration   

        Returns
        -------
        Tuple
            A tuple containing the results for each particle: 
                (posterior, distances, accepted_count, trial_count)

        """
        posterior, distances = [], []
        trial_count, accepted_count = 0, 0
        while accepted_count < self.n_particles:
            theta = self._draw_theta(theta_prev=theta_prev, weights=weights, tau_squared=tau_squared)
            synthetic_data = self.generate_data()
            distance = self.distance_function(synthetic_data)
            trial_count += 1

            if distance < epsilon:
                if self.verbose:
                    print "Sampler #" + str(run_id+1)
                    print "Accepted sample " + str(accepted_count+1) + " of " + str(self.n_particles)
                    print "Model Fit (1 - distance) = " + str(1.0-distance)

                self.get_appendices(run_id)
                accepted_count += 1
                posterior.append(theta)
                distances.append(distance)
            else:
                pass

        posterior = np.asarray(posterior).T
        distances = np.asarray(distances)
        return (posterior, distances,
                accepted_count, trial_count)

    def _calc_weights(self, t_prev, t_curr, tau_2, w_old, prior="None"):
        """
        Calculates importance weights
        
        Parameters
        ----------
        t_prev : ndarray
            Particles from the previous iteration
        t_curr : ndarray
            Particles from the current iteration
        tau_2 : ndarray
            2 x covariance matrix of the particle distribution
        w_old : ndarray
            The importance weights from the previous iteration
        prior : list, optional
            The prior distributions (required for the first iteration only)
        """
        weights_new = np.zeros_like(w_old)
        if len(t_curr.shape) == 1:
            norm = np.zeros_like(t_curr)
            for i, T in enumerate(t_curr):
                for j in xrange(t_prev[0].size):
                    norm[j] = stats.norm.pdf(T, loc=t_prev[0][j],
                                         scale=tau_2)
                weights_new[i] = prior[0].pdf(T)/sum(w_old * norm)

            return weights_new/weights_new.sum()

        else:
            norm = np.zeros(t_prev.shape[1])
            for i in xrange(t_curr.shape[1]):
                prior_prob = np.zeros(t_curr[:, i].size)
                for j in xrange(t_curr[:, i].size):
                    prior_prob[j] = prior[j].pdf(t_curr[:, i][j])
                #assumes independent priors
                p = prior_prob.prod()

                for j in xrange(t_prev.shape[1]):
                    norm[j] = stats.multivariate_normal.pdf(t_curr[:, i], mean=t_prev[:, j], cov=tau_2)
                weights_new[i] = p/sum(w_old * norm)

            return weights_new/weights_new.sum()

    def _weighted_covar(self, x, w):
        """
        Calculates weighted covariance matrix
        
        Parameters
        ----------
        x : ndarray
            The particles sampled
        w : ndarray
            The importance weights for each particle
        
        Returns
        -------
        ndarray
            Weighted covariance of the particle distribution
        """
        sumw = w.sum()
        assert_almost_equal(sumw, 1.0)
        if len(x.shape) == 1:
            assert x.shape[0] == w.size
        else:
            assert x.shape[1] == w.size
        sum2 = np.sum(w**2)

        if len(x.shape) == 1:
            xbar = (w*x).sum()
            var = sum(w * (x - xbar)**2)
            return var * sumw/(sumw*sumw-sum2)
        else:
            xbar = [(w*x[i]).sum() for i in xrange(x.shape[0])]
            covar = np.zeros((x.shape[0], x.shape[0]))
            for k in xrange(x.shape[0]):
                for j in xrange(x.shape[0]):
                    for i in xrange(x.shape[1]):
                        covar[j,k] += (x[j,i]-xbar[j])*(x[k,i]-xbar[k]) * w[i]

            return covar * sumw/(sumw*sumw-sum2)

    def _effective_sample_size(self, w):
        """
        Calculates effective sample size
        
        Parameters
        ----------
        w : ndarray
            Importance weights
            
        Returns
        -------
        float
            Importance sampling weights
        """
        sumw = sum(w)
        sum2 = sum (w**2)
        return sumw*sumw/sum2

    def _check_iteration_n(self):
        """
        An auxiliary function to count the number of iterations
        """

        if os.path.exists(self.data.output_dir + 'iteration_1.hdf5'):
            self.iteration = int(os.popen("ls " + self.data.output_dir + "iteration*" + " | wc -l").read()) + 1
        else:
            self.iteration = 1

