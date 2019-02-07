#!/usr/bin/python

""" Dynamic mean field model base class."""
from utils import cov_to_corr
from utils import load_model_params
from hemo import Balloon
from sim import Sim

from scipy.optimize import fsolve
from scipy.linalg import solve_lyapunov, eig
import scipy.sparse as spr
import numpy as np
import sympy as sym

class Model(object):

    """
    Class for the large-scale computational model with optional heterogeneous parametrization of w^{EE} and w^{EI}, 
    based on provided heterogeneity map.
    """

    def __init__(self, sc, g=0, norm_sc=True, hmap = None,
                 wee=(0.15, 0.), wei=(0.15, 0.),
                 syn_params=None, bold_params='obata',
                 verbose=True):
        """
        
        Parameters
        ----------
        sc : ndarray
            Structural connectivity matrix
        g : float, optional
            Global coupling parameter to scale structural connectivity matrix
        norm_sc : bool, optional
            Normalize input strengths of the structural connectivity matrix (True by default)
        hmap : ndarray, optional
            Heterogeneity map to scale local model parameters. 
            If None, the model parameters are homogeneous (None by default)
        wee : tuple, optional
            Local recurrent excitatory connectivity weights (w^{EE}). 
            Requires a tuple with size 2 as (w_{min}, w_{scale}). 
            (w_{min}=0.15, w_{scale}=0.0 by default)
        wei : tuple, optional
            Local excitatory to inhibitory connectivity weights (w^{EI}). 
            Requires a tuple with size 2 as (w_{min}, w_{scale}). 
            (w_{min}=0.15, w_{scale}=0.0 by default)
        syn_params : list, optional
            Synaptic dynamical model parameters (None by default)
        bold_params : str, optional
            Hemodynamic model parameters. 'obata' or 'friston' ('obata' by default)
        verbose : bool, optional
            If True, prints diagnostics to console (True by default)  
        
        """

        # Structural connectivity / number of cortical areas
        self._SC = sc
        self._nc = sc.shape[0]

        # Global coupling
        self._G = g

        # Print diagnostics to console
        self._verbose = verbose

        # Initialize hemodynamic model (Balloon-Windkessel)
        self.hemo = Balloon(self._nc, parameters=bold_params)

        # Initialize simulation class
        self.sim = Sim()

        # If custom model parameters were provided, load model parameters with corresponding keys
        model_params = load_model_params()
        if syn_params is not None:
            for key in syn_params.keys():
                model_params[key] = syn_params[key]

        # Unstable if Jacobian has eval > 0
        self._unstable = False

        # Initialize model outputs to None
        self._jacobian = None
        self._cov = None
        self._corr = None
        self._cov_bold = None
        self._corr_bold = None
        self._full_cov = None

        # Initialize state members to None
        self._I_E = None
        self._I_I = None
        self._S_E = None
        self._S_I = None
        self._r_E = None
        self._r_I = None

        # Various model parameters
        self._w_II = np.repeat(model_params['w_II'], self._nc)
        self._w_IE = np.repeat(model_params['w_IE'], self._nc)
        self._w_EE = np.repeat(model_params['w_EE'], self._nc)
        self._w_EI = np.repeat(model_params['w_EI'], self._nc)

        self._I0 = np.repeat(model_params['I0'], self._nc)
        self._J_NMDA = np.repeat(model_params['J_NMDA'], self._nc)
        self._sigma = model_params['sigma']
        self._gamma = model_params['gamma']
        self._W_I = model_params['W_I']
        self._W_E = model_params['W_E']
        self._tau_I = model_params['tau_I']
        self._tau_E = model_params['tau_E']
        self._d_I = model_params['d_I']
        self._d_E = model_params['d_E']
        self._b_I = model_params['b_I']
        self._b_E = model_params['b_E']
        self._a_I = model_params['a_I']
        self._a_E = model_params['a_E']
        self._I_ext = np.repeat(model_params['I_ext'], self._nc)

        self._gamma_I = 1.0

        self._tau_E_reset = np.copy(self._tau_E)
        self._tau_I_reset = np.copy(self._tau_I)
        self._gamma_reset = np.copy(self._gamma)
        self._gamma_I_reset = np.copy(self._gamma_I)

        # Baseline input currents
        self._I0_E = self._W_E * self._I0
        self._I0_I = self._W_I * self._I0

        # Steady state values for isolated node
        self._I_E_ss = np.repeat(model_params['I_E_ss'], self._nc)
        self._I_I_ss = np.repeat(model_params['I_I_ss'], self._nc)
        self._S_E_ss = np.repeat(model_params['S_E_ss'], self._nc)
        self._S_I_ss = np.repeat(model_params['S_I_ss'], self._nc)
        self._r_E_ss = np.repeat(model_params['r_E_ss'], self._nc)
        self._r_I_ss = np.repeat(model_params['r_I_ss'], self._nc)

        # Noise covariance matrix
        self._Q = np.identity(2 * self._nc) * self._sigma * self._sigma

        # Add lookup tables for transfer function and its derivatives
        self._phi()

        # Heterogeneity map values for each area
        self._raw_hmap = hmap
        self._hamp = 0.0
        self._hmap_rev = 0.0

        # Set heterogeneity gradients
        if self._raw_hmap is not None:
            hmap_range = np.ptp(self._raw_hmap)
            self._hmap = (-(self._raw_hmap - np.max(self._raw_hmap)) / hmap_range)

            hmap_norm = self._raw_hmap - np.min(self._raw_hmap)
            self._hmap_rev = hmap_norm / np.max(hmap_norm)

            self._w_EE = self._apply_hierarchy(wee[0], wee[1])
            self._w_EI = self._apply_hierarchy(wei[0], wei[1])

        # Set SC normalization
        self._sc_norm = 1.0
        if norm_sc:
            sc_norm = 1. / self._SC.sum(1)
            self._sc_norm = np.tile(sc_norm, (self._nc, 1)).T

        return

    def __repr__(self):
        return "dynamic mean field model class"

    def __str__(self):
        msg = ""
        msg += '%-17s %s' % ("\nUnstable:", self._unstable)
        msg += '\n%-16s %s' % ("Coupling (G):", self._G)
        msg += '\n%-16s %s' % ("N areas:", self._nc)
        return msg

    def set_jacobian(self, compute_fic=True):
        """ 
        Set Jacobian matrix given the model parameters. 
         
        Parameters
        ----------
        compute_fic : boolean, optional
            if True, local feedback inhibition parameters (w^{IE}) are adjusted to set the firing rates of
            excitatory populations to ~3Hz
        
        Returns
        -------
        boolean
            If True, the system is stable and linearized covariance can be successfully calculated, otherwise
            moments method fails because the solution is not stable for the given parameters.
        
        Notes
        -----
        This method should be executed before calculating the linearized covariance or performing numerical integration,
        each time the model parameters are modified.
        """

        eye = np.identity(self._nc)

        # Excitatory and inhibitory connection weights

        self._K_EE = (self._w_EE * eye) + (self._G * self._J_NMDA * self._sc_norm * self._SC)
        self._K_EI = (self._w_EI * eye)

        # Local feedback inhbition
        if compute_fic:
            self._w_IE = self._analytic_FIC()

        self._K_IE = -self._w_IE * eye
        self._K_II = -self._w_II * eye

        if np.any(self._w_IE < 0):
            self._unstable = True
            raise ValueError("Warning: FIC calculation led to negative J values!")
        else:
            # Derivatives of transfer function for each cell type
            # at steady state value of current
            dr_E = self.dphi_E(self._I_E_ss) * eye
            dr_I = self.dphi_I(self._I_I_ss) * eye

            # A_{mn} = dS_i^m/dS_j^n
            A_EE = (-1. / self._tau_E - (self._gamma * self._r_E_ss)) * eye + \
                   ((-self._gamma * (self._S_E_ss - 1.)) * eye).dot(dr_E.dot(self._K_EE))

            A_IE = ((self._gamma * (1. - self._S_E_ss)) * eye).dot(dr_E.dot(self._K_IE))
            A_EI = self._gamma_I * dr_I.dot(self._K_EI)
            A_II = (-1. / self._tau_I) * eye + self._gamma_I * dr_I.dot(self._K_II)

            # Stack blocks to form full Jacobian
            col1 = np.vstack((A_EE, A_EI))
            col2 = np.vstack((A_IE, A_II))
            self._jacobian = np.hstack((col1, col2))

            # Eigenvalues of Jacobian matrix
            self._evals, self._evects = eig(self._jacobian)
            self._max_eval = np.real(self._evals.max())

            # Check stability using eigenvalues
            self._unstable = self._max_eval >= 0.0

        return not self._unstable


    def moments_method(self, bold=False, use_lyapunov=False):
        """
        Computes the linearized covariance and the correlation matrices between model variables.
        
        Parameters
        ----------
        bold : boolean, optional
            if True, the covariance and correlation are computed for the extended hemodynamic system (BOLD),
            otherwise it computes only for the synaptic system of equations. (False by default)
        use_lyapunov : boolean, optional
            if True, the builtin function scipy.linalg.solve_lyapunov is used to solve Lyapunov equations.
            (False by default)(not recommended, only for debugging)
        
        Notes
        -----
        The covariance and correlation matrices can be called only after performing this method. 
        """

        if self._jacobian is None: self.set_jacobian()
        self._linearized_cov(use_lyapunov=use_lyapunov, bold=bold)
        self._reset_state()

        return

    def csd(self, freqs, pop='E'):
        """
        Computes cross-spectral density of synaptic variables.

        Parameters
        ----------
        freqs : ndarray or list
            An array or list containing the frequency bins for which the CSD will be computed
        pop : str, optional
            If 'E', the CSD of excitatory populations will return (default). if 'I', the CSD of inhibitory 
            populations will return. Any other string will be ignored and the full CSD will be returned 

        Returns
        -------
        csd : ndarray
            CSD of the synaptic system in shape NxNxf, where N is the number of regions and f is the number
            of frequency bins
        """
        if self._jacobian is None: self.set_jacobian()

        Id = np.identity(self._nc * 2)
        power = np.empty((2 * self._nc, 2 * self._nc, len(freqs)), dtype=complex)
        sig = complex(self._sigma ** 2)
        for i, f in enumerate(freqs):
            w = 2. * np.pi * f
            M1 = np.linalg.inv(self.jacobian + 1.j * w * Id)
            M2 = np.linalg.inv(self.jacobian.T - 1.j * w * Id)
            M3 = np.dot(M1, M2)
            power[:, :, i] = M3 * sig
        if pop == 'E':
            return power[:self._nc, :self._nc, :]
        elif pop == 'I':
            return power[self._nc:, self._nc:, :]
        else:
            return power

    def csd_bold(self, freqs):
        """Computes cross-spectral density of hemodynamic variables.

        Parameters
        ----------
        freqs : ndarray or list
            An array or list containing the frequency bins for which the CSD will be computed

        Returns
        -------
        ndarray
            CSD of the BOLD transformed hemodynamic system in shape NxNxf, where N is the number 
            of regions and f is the number of frequency bins.
            
        Notes
        -----
        The computation is performed for the hemodynamic system, the results are returned for
        BOLD signals. 
        """
        if self._jacobian_bold is None: self.moments_method(bold=True)
        N = self._jacobian_bold.shape[0]

        Id = np.identity(N)
        power = np.empty((self._nc, self._nc, len(freqs)), dtype=complex)
        sig = complex(self._sigma ** 2)
        for i, f in enumerate(freqs):
            w = 2. * np.pi * f
            M1 = np.linalg.inv(self._jacobian_bold + 1.j * w * Id)
            M2 = np.linalg.inv(self._jacobian_bold.T - 1.j * w * Id)
            M3 = np.dot(M1, M2)
            hemo_power = M3 * sig
            power[:,:,i] = (np.dot(np.dot((self.hemo.B), hemo_power), (self.hemo.B.conj().T)))
        return power


    def integrate(self, t,
                  dt=1e-4, n_save=10, stimulation=0.0,
                  delays=False, distance=None, velocity=None,
                  include_BOLD=True, from_fixed=True,
                  sim_seed=None, save_mem=False):
        """Computes cross-spectral density of hemodynamic variables.

        Parameters
        ----------
        t : int
            Total simulation time in seconds.
        dt : float, optional
            Integration time step in seconds. By default dt is 0.1 msec.
        n_save : int, optional
            Sampling rate (time points). By default n_save is 10, therefore in dt is 0.1 msec, all the 
            variables will be sampled at 1 msec.
        stimulation : ndarray or float, optional
            An array or matrix containing external currents if required. The size of array should match
            to the number time points (i.e. int(t / dt + 1)) (0.0 by default)
        delays : bool, optional
            If True, delays are included during the integration (False by default)
        distance : ndarray, optional
            The distance matrix, If delays will be taken into account. The distance matrix should contain
            the euclidean or geodesic distance between regions in mm.
        velocity : float, optional
            The conduction velocity in m/sec, if conduction delays are not ignored.
        include_BOLD : boolean, optional
            If True, the simulation will also include hemodynamic model and BOLD signals (True by default)
        from_fixed : boolean, optional
            If True, the simulation will begin using steady state values of the parameters,
            otherwise the last available values will be used (i.e. from previous simulations...etc.)
        sim_seed : int, optional
            The seed for random number generator.
        
        
        Returns
        -------
        None
            
        Notes
        -----
            This method simulates the system for the given simulation time and the parameter values are stored.
            After successfull simulation, The excitatory synaptic variables can be obtained by .sim.S_E or 
            BOLD signals can be obtained by .sim.y
        """

        sim_seed = np.random.randint(0, 4294967295) if sim_seed is None else sim_seed
        np.random.seed(sim_seed)

        # Initialize to fixed point
        if from_fixed:
            self._reset_state()
            self.hemo.reset_state()

        # Simulation parameters
        dt_save = dt * n_save
        n_sim_steps = int(t / dt + 1)
        n_save_steps = int(t / dt_save + 1)

        # Synaptic state record
        if not save_mem:
            synaptic_state = np.zeros((6, self.nc, n_save_steps))
            synaptic_state[:, :, 0] = self.state
        # Synaptic state record
        #synaptic_state = np.zeros((6, self.nc, n_save_steps))
        #synaptic_state[:, :, 0] = self.state

        if self._verbose:
            print "Beginning simulation."

        self.delays = delays
        if self.delays:
            if distance is None or velocity is None:
                self.delays = False
                msg = "Distance matrix and transmission velocity to implement delays"
                raise NotImplementedError(msg)
            else:
                self.distance = distance
                self.velocity = velocity
                self.steps_Delay = np.round(self.distance / (self.velocity * 1e-4 * 1e3)).astype(int)
                self._S_E_mem = np.tile(self._S_E, (self.steps_Delay.max() + 1, 1)).T
                self._S_E_vect = self._S_E_mem[range(self._nc), self.steps_Delay]


        # Initialize BOLD variables if required
        if include_BOLD:
            if save_mem:
                # Hemodynamic state record
                hemo_state = np.zeros((self._nc, n_save_steps))
                #hemo_state[:3, :, 0] = 1.  # ICs
            else:
                # Hemodynamic state record
                hemo_state = np.zeros((5, self._nc, n_save_steps))
                hemo_state[:3, :, 0] = 1.  # ICs
            # Hemodynamic state record
            #hemo_state = np.zeros((5, self._nc, n_save_steps))
            #hemo_state[:3, :, 0] = 1.  # ICs

        # Main for-loop
        for i in range(1, n_sim_steps):
            if self.delays:
                self._S_E_mem[:, 0] = self._S_E
                self._S_E_vect = self._S_E_mem[range(self._nc), self.steps_Delay]

            self._step(dt)

            if self.delays:
                self._S_E_mem[:, 1:] = self._S_E_mem[:, :-1]

            # Update state variables
            if not (i % n_save):
                i_save = i / n_save
                if not save_mem:
                    synaptic_state[:, :, i_save] = self.state

                if include_BOLD:
                    self.hemo.step(dt*10., self._S_E - self._S_E_ss)
                    if save_mem:
                        hemo_state[:, i_save] = self.hemo._y
                    else:
                        hemo_state[:, :, i_save] = self.hemo.state
                    #hemo_state[:, :, i_save] = self.hemo.state

                if self._verbose:
                    if not (i_save % 1000):
                        print i_save

        if self._verbose:
            print "Simulation complete."

        self.sim.t = t
        self.sim.dt = dt_save
        self.sim.n_save = n_save
        self.sim.t_points = np.linspace(0, t, n_save_steps)
        self.sim.seed = sim_seed

        if not save_mem:
            self.sim.I_I, self.sim.I_E, self.sim.r_I, self.sim.r_E, \
            self.sim.S_I, self.sim.S_E = synaptic_state

        if include_BOLD:
            if save_mem:
                self.sim.y = hemo_state
            else:
                self.sim.x, self.sim.f, self.sim.v, self.sim.q, self.sim.y = hemo_state

        #self.sim.I_I, self.sim.I_E, self.sim.r_I, self.sim.r_E, \
        #self.sim.S_I, self.sim.S_E = synaptic_state

        #if include_BOLD:
        #    self.sim.x, self.sim.f, self.sim.v, self.sim.q, self.sim.y = hemo_state

        return

    # Auxiliary Methods
    def _reset_state(self):
        """
        Reset state members to steady-state values.
        """
        self._I_I = np.copy(self._I_I_ss)
        self._I_E = np.copy(self._I_E_ss)
        self._r_I = np.copy(self._r_I_ss)
        self._r_E = np.copy(self._r_E_ss)
        self._S_I = np.copy(self._S_I_ss)
        self._S_E = np.copy(self._S_E_ss)

        return

    def _phi(self):
        """
        Generate transfer function and derivatives for Excitatory and Inhibitory populations.
        """
        IE = sym.symbols('IE')
        II = sym.symbols('II')
        phi_E = (self._a_E * IE - self._b_E) / (1. - sym.exp(-self._d_E * (self._a_E * IE - self._b_E)))
        phi_I = (self._a_I * II - self._b_I) / (1. - sym.exp(-self._d_I * (self._a_I * II - self._b_I)))
        dphi_E = sym.diff(phi_E, IE)
        dphi_I = sym.diff(phi_I, II)

        self.phi_E = sym.lambdify(IE, phi_E, "numpy")
        self.phi_I = sym.lambdify(II, phi_I,"numpy")
        self.dphi_E = sym.lambdify(IE, dphi_E, "numpy")
        self.dphi_I = sym.lambdify(II, dphi_I, "numpy")

        return

    def _inh_curr_fixed_pts(self, I):
        """ 
        Auxiliary function to find steady state inhibitory currents when FFI is enabled.

        Parameters
        ----------
        I : Inhibitory current

        Returns
        -------
        ndarray
            The fixed points for inhibitory currents 
        """
        return self._I0_I + self._K_EI.dot(self._S_E_ss) - \
               self._w_II * self._gamma_I * self._tau_I * self.phi_I(I) - I

    def _analytic_FIC(self):
        """ 
        Analytically solves for the strength of feedback inhibition for each cortical area. 

        Returns
        -------
        J : ndarray
            Local feedback inhibition providing excitatory firing rates ~3Hz
        """

        if self._SC is None:
            raise Exception("You must supply a connectivity matrix.")

        # Numerically solve for inhibitory currents first
        I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts,
                                             x0=self._I_I_ss, full_output=True)

        if ier:  # successful exit from fsolve
            self._I_I = np.copy(I_I_ss)  # needed for self._update_rate_I()
            self._I_I_ss = np.copy(I_I_ss)  # update stored steady state value
            self._r_I = self.phi_I(self._I_I)  # compute new steady state rate
            self._r_I_ss = np.copy(self._r_I)  # update stored steady state value
            self._S_I_ss = np.copy(self._r_I_ss) * self._tau_I * self._gamma_I # update stored val.
        else:
            err_msg = "Failed to find new steady-state currents." + \
                      " Cause of failure: %s" % mesg
            raise Exception(err_msg)

        # Solve for J using the steady state values (fixed points)
        J = (-1. / self._S_I_ss) * \
            (self._I_E_ss -
             self._I_ext - self._I0_E -
             self._K_EE.dot(self._S_E_ss))

        return J

    def _solve_lyapunov(self, jacobian, evals, L, Q, bold=False, builtin=False):
        """
        Solves Lyapunov equation
        
        Parameters
        ----------
        jacobian : ndarray
            The Jacobian of the system
        evals : ndarray
            Eigenvalues of the Jacobian matrix
        L : ndarray
            Eigenvectors of the Jacobian matrix
        Q : ndarray
            Input covariance matrix
        bold : boolean, optional
            If True, the Lyapunov equation is solved for hemodynamic system.
        builtin : boolean, optional
            If True, the builtin function will be used (not recommended)

        Notes
        -----
        This method requires the Jacobian matrix and the eigendecomposition of the Jacobian matrix
        as input for computational efficiency.

        Returns
        -------
        cov : ndarray
            The covariance matrix of the system around stable fix point
        """

        if builtin:
            return solve_lyapunov(jacobian, -Q)
        else:
            n = jacobian.shape[0] / self._nc
            evals_cc = np.conj(evals)
            Q = spr.csc_matrix(Q)

            L_inv = np.linalg.inv(L)

            inv_L_dagger = np.conj(L_inv).T

            QQ = Q.dot(inv_L_dagger)

            Q_tilde = L_inv.dot(QQ)

            denom_lambda_i = np.tile(evals.reshape((1, n * self._nc)).T,
                                     (1, n * self._nc))
            denom_lambda_conj_j = np.tile(evals_cc, (n * self._nc, 1))
            total_denom = denom_lambda_i + denom_lambda_conj_j
            M = -Q_tilde / total_denom

            if bold:
                B = spr.csc_matrix(self.hemo.B)
                X = B.dot(L)
                cov = np.dot(X.dot(M), X.conj().T).real
            else:
                L_dagger = np.conj(L).T
                cov = L.dot(M.dot(L_dagger)).real
            return cov

    def _linearized_cov(self, use_lyapunov=False, bold=False):
        """ 
        Solves for the linearized covariance matrix, using either
        the Lyapunov equation or eigen-decomposition.

        Parameters
        ----------
        use_lyapunov : boolean, optional
            If True, the builtin function will be used (not recommended)
        bold : boolean, optional
            If True, the Lyapunov equation is solved for hemodynamic system.
        """

        if self._unstable:
            if self._verbose: print "System unstable - no solution to Lyapunov equation - exiting"
            self._cov, self._cov_bold, self._corr_bold, self._corr = None, None, None, None
            return
        else:
            if bold:
                self.hemo.linearize_BOLD(self._S_E_ss, self._jacobian, self._Q)
                self._jacobian_bold = self.hemo.full_A
                self._Q_bold = self.hemo.full_Q

                evals, evects = eig(self._jacobian_bold)
                self.evals_bold = evals
                self.evects_bold = evects

                self._cov_bold = self._solve_lyapunov(self._jacobian_bold, evals, evects, self._Q_bold, bold=True,
                                                      builtin=use_lyapunov)
                self._corr_bold = cov_to_corr(self._cov_bold, full_matrix=False)
            else:
                self._cov = self._solve_lyapunov(self._jacobian, self._evals, self._evects, self._Q,
                                                 builtin=use_lyapunov)
                self._corr = cov_to_corr(self._cov)


    def _exc_current(self):
        """
        Excitatory current for each cortical region.
        
        
        Parameters
        ----------
        I_ext : float
            External stimulation at each time step
        
        Returns
        -------
        ndarray
            Excitatory currents at each time step 
        """
        if self.delays:
            return self._I0_E + self._I_ext + (self._K_EE * self._S_E_vect).sum(1) + self._K_IE.dot(self._S_I)
        else:
            return self._I0_E + self._I_ext + self._K_EE.dot(self._S_E) + self._K_IE.dot(self._S_I)

    def _inh_current(self):
        """ 
        Inhibitory current for each cortical region
        
        Returns
        -------
        ndarray
            Inhibitory currents at each time step
        """
        return self._I0_I + self._K_EI.dot(self._S_E) + self._K_II.dot(self._S_I)

    def _step(self, dt):
        """ 
        Advance system synaptic state by time evolving for time dt. 
        
        Parameters
        ----------
        dt : float
            Integration time step in seconds. By default dt is 0.1 msec.
        I_ext : float
            External stimulation at time step
        """
        self._I_E = self._exc_current()
        self._I_I = self._inh_current()

        self._r_E = self.phi_E(self._I_E)
        self._r_I = self.phi_I(self._I_I)

        # Compute change in synaptic gating variables
        dS_E = self._dSEdt() * dt + np.sqrt(dt) * self._sigma * \
               np.random.normal(size=self._nc)

        dS_I = self._dSIdt() * dt + np.sqrt(dt) * self._sigma * \
               np.random.normal(size=self._nc)

        # Update class members S_E, S_I
        self._S_E += dS_E
        self._S_I += dS_I

        # Clip synaptic gating fractions
        self._S_E = np.clip(self._S_E, 0., 1.)
        self._S_I = np.clip(self._S_I, 0., 1.)

        return

    def _dSEdt(self):
        """
        Returns time derivative of excitatory synaptic gating variables in absense of noise.
        
        Returns
        -------
        ndarray
            Derivatives of excitatory synaptic gating variables
        """
        return -(self._S_E / self._tau_E) + (self._gamma * self._r_E) * (1. - self._S_E)

    def _dSIdt(self):
        """Returns time derivative of inhibitory synaptic gating variables in absense of noise.
        
        Returns
        -------
        ndarray
            Derivatives of excitatory synaptic gating variables
        """
        return -(self._S_I / self._tau_I) + self._gamma_I * self._r_I

    def _apply_hierarchy(self, a, b):
        """
        Parametrize model parameters based on the heterogeneity map. 
                
        Parameters
        ----------
        a : float
            The interceot term
        b : float
            The scaling factor 

        Returns
        -------
        ndarray
            Parameter values varying along heterogeneity gradient given the intercept and scaling
            factor.
            
        Notes
        -----
        If b is negative the heterogeneity gradient will be calculated in opposite direction.
        """
        if b < 0.0:
            return a + np.abs(b) * self._hmap_rev
        else:
            return a + b * self._hmap

    # Properties
    @property
    def Q(self):
        """
        Returns
        -------
        ndarray
            Input covariance matrix
        """
        return self._Q

    @property
    def cov(self, full=False):
        """
        Parameters
        ----------
        full : bool, optional
            If True, returns the full covariance matrix.
        
        Returns
        -------
        ndarray
            Covariance matrix of linearized fluctuations about fixed point.
        """
        if full:
            return self._cov
        else:
            return self._cov[:self._nc, :self._nc]

    @property
    def cov_bold(self):
        """
        Returns
        -------
        ndarray
            Covariance matrix of linearized fluctuations for BOLD
        """
        return self._cov_bold

    @property
    def var(self, full=False):
        """
        Parameters
        ----------
        full : bool, optional
            If True, returns the full variance.

        Returns
        -------
        ndarray
            Variances of linearized fluctuations about fixed point.
        """
        if full:
            return np.diag(self._cov)
        else:
            return np.diag(self._cov[:self._nc, :self._nc])

    @property
    def var_bold(self):
        """
        Returns
        -------
        ndarray
            Variances of linearized fluctuations for BOLD.
        """
        return np.diag(self._cov_bold)

    @property
    def corr(self):
        """
        Returns
        -------
        ndarray
            Correlation matrix (model FC) for synaptic system 
        """
        return self._corr

    @property
    def corr_bold(self):
        """
        Returns
        -------
        ndarray
            Correlation matrix (model FC) for BOLD 
        """
        return self._corr_bold

    @property
    def jacobian(self):
        """
        Returns
        -------
        ndarray
            Jacobian of linearized fluctuations about fixed point. 
        """
        return self._jacobian

    @property
    def evals(self):
        """
        Returns
        -------
        ndarray
            Eigenvalues of Jacobian matrix. 
        """
        return eig(self._jacobian)[0] if self.jacobian is not None else None

    @property
    def evecs(self):
        """
        Returns
        -------
        ndarray
            Left Eigenvextors of Jacobian matrix. 
        """
        return eig(self._jacobian)[1] if self.jacobian is not None else None

    @property
    def nc(self):
        """
        Returns
        -------
        ndarray
            Number of cortical areas. 
        """
        return self._nc

    @property
    def SC(self):
        """
        Returns
        -------
        ndarray
            Empirical structural connectivity. 
        """
        return self._SC

    @property
    def sigma(self):
        """
        Returns
        -------
        ndarray
            Input noise to each area 
        """
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        """
        Parameters
        -----------
        sigma : ndarray or float
            Input noise to each area
        """
        self._sigma = sigma
        self._Q = np.identity(2 * self._nc) * self._sigma * self._sigma

    @SC.setter
    def SC(self, SC):
        """
        Parameters
        -----------
        SC : ndarray
            Empirical structural connectivity
        """
        assert(SC.shape[0] == self._nc)
        self._SC = SC
        return

    @property
    def state(self):
        """
        Returns
        -------
        ndarray
            All state variables, 6 rows by len(nodes) columns.
            Rows are I_I, I_E, r_I, r_E, S_I, S_E.  
        """
        return np.vstack((self._I_I, self._I_E, self._r_I,
                          self._r_E, self._S_I, self._S_E))

    @property
    def steady_state(self):
        """
        Returns
        -------
        ndarray
            All steady state variables, shape 6 x nc.
            Rows are, respectively, I_I, I_E, r_I, r_E, S_I, S_E.  
        """
        return np.vstack((self._I_I_ss, self._I_E_ss, self._r_I_ss,
                          self._r_E_ss, self._S_I_ss, self._S_E_ss))

    @property
    def w_EE(self):
        """
        Returns
        -------
        ndarray
            Local recurrenm excitatory strengths.  
        """
        return self._w_EE

    @w_EE.setter
    def w_EE(self, w):
        """
        Parameters
        -------
        w : ndarray
            Local recurrent excitatory strengths.
              
        Notes
        -----
        If w is float, sets all strengths to w;
        if w has N elements (number of regions), sets all strengths to w;
        if w has size 2, sets w according to heterogeneity map
        """
        if isinstance(w, float):
            self._w_EE = w
        else:
            if len(w) == self._nc:
                self._w_EE = np.array(w)
            else:
                self._w_EE = self._apply_hierarchy(w[0], w[1])

    @property
    def w_EI(self):
        """
        Returns
        -------
        ndarray
            Local excitatory to inhibitory strengths.  
        """
        return self._w_EI

    @w_EI.setter
    def w_EI(self, w):
        """
        Parameters
        -------
        w : ndarray
            Local excitatory to inhibitory strengths.

        Notes
        -----
        If w is float, sets all strengths to w;
        if w has N elements (number of regions), sets all strengths to w;
        if w has size 2, sets w according to heterogeneity map
        """
        if isinstance(w, float):
            self._w_EI = w
        else:
            if len(w) == self._nc:
                self._w_EI = np.array(w)
            else:
                self._w_EI = self._apply_hierarchy(w[0], w[1])

    @property
    def G(self):
        """
        Returns
        -------
        ndarray
            Global coupling strength.  
        """
        return self._G

    @G.setter
    def G(self, g):
        """
        Parameters
        ----------
        g : ndarray
            Global coupling strength.  
        """
        self._G = g
        return

    @property
    def w_IE(self):
        """
        Returns
        -------
        ndarray
            Feedback inhibition weights.  
        """
        return self._w_IE

    @w_IE.setter
    def w_IE(self, J):
        """
        Parameters
        ----------
        J : ndarray
            Feedback inhibition weights.  
        """
        self._w_IE = J
        return

    @property
    def hmap(self):
        """
        Returns
        -------
        ndarray
            Heterogeneity map values
        """
        return self._hmap

    @hmap.setter
    def hmap(self, h):
        """
        Parameters
        ----------
        h : ndarray
            Heterogeneity map values
            
        Notes
        -----
        The heterogeneity map values should be normalized between 0 and 1
        """
        self._hmap = h

    @property
    def I_ext(self):
        """
        Returns
        -------
        ndarray
            External current  
        """
        return self._I_ext

    @I_ext.setter
    def I_ext(self, I):
        """
        Parameters
        ----------
        I : ndarray
            External current  
        """
        self._I_ext = I

    @property
    def J_NMDA(self):
        """
        Returns
        -------
        float
            Effective NMDA conductance  
        """
        return self._J_NMDA

    @J_NMDA.setter
    def J_NMDA(self, J):
        """
        Parameters
        ----------
        float
            Effective NMDA conductance  
        """
        self._J_NMDA = J
