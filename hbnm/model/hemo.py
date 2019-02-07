#! /usr/bin/python

""" Hemodynamic transfer function class for input-state-output. """
from params.synaptic import S_E_ss as z0
from utils import clean_builtins
import numpy as np
from scipy.linalg import solve_lyapunov, eig

class Balloon(object):
    """
    The class containing hemodynamic response function    
    """

    def __init__(self, nc, linearize=False, parameters='obata'):
        """
        
        Parameters
        ----------
        nc : int
            Number of brain areas
        linearize : boolean, optional
            if True, the calculations (including the simulations) will be performed on the linearized system
            of equations, otherwise the nonlinear system of equations will be used
        parameters : str, optional
            The parameter set for hemodynamic response function. By default, the parameters in Obata et al., 2014
             will be used. The BOLD parameters k1, k2, k2 are updated for 3T following Heinzle et al., 2016. 
        """

        linearize = linearize
        self._nc = nc
        if isinstance(parameters, dict):
            hemo_param_dict = parameters
            for checkkeys in ['V0', 'kappa', 'gamma', 'tau', 'alpha', 'rho', 'k1', 'k2', 'k3']:
                if not checkkeys in hemo_param_dict.keys():
                    from params import obata04
                    param_dict = clean_builtins(vars(obata04))
                    hemo_param_dict[checkkeys] = param_dict[checkkeys]
        else:
            if parameters == 'obata':
                # Clean dictionary by removing builtins
                from params import obata04
                hemo_param_dict = clean_builtins(vars(obata04))
            elif parameters == 'friston':
                # Clean dictionary by removing builtins
                from params import friston03
                hemo_param_dict = clean_builtins(vars(friston03))
            else:
                raise NotImplementedError("invalid hemodynamic response function parameter set")

        # Hemodynamic model parameters
        self._V0 = hemo_param_dict['V0']
        self._kappa = hemo_param_dict['kappa']
        self._gamma = hemo_param_dict['gamma']
        self._tau = hemo_param_dict['tau']
        self._alpha = hemo_param_dict['alpha']
        self._rho = hemo_param_dict['rho']
        self._k1 = hemo_param_dict['k1']
        self._k2 = hemo_param_dict['k2']
        self._k3 = hemo_param_dict['k3']

        # Steady-state values
        self._z0 = 0.0#z0
        self._x0 = 0.0
        self._f0 = 1.0
        self._v0 = 1.0
        self._q0 = 1.0
        
        # Obata 2004
        self._y0 = 0.0

        # Hemodynamic state variables
        self._x = np.repeat(self._x0, self._nc) # vasodilatory signal
        self._f = np.repeat(self._f0, self._nc) # normalized inflow rate
        self._v = np.repeat(self._v0, self._nc) # normalized blood volume
        self._q = np.repeat(self._q0, self._nc) # norm. deoxyhemoglobin content
        self._y = np.repeat(self._y0, self._nc) # BOLD signal (%)

        if linearize:
            self.step = self.linear_step
        else:
            self.step = self.nonlinear_step

        return

    def BOLD_tf(self, freqs):
        """
        The analytic solution to the transfer function
        of the BOLD signal y as a function of the input
        synaptic signal z, at a given frequency f, for the
        Balloon-Windkessel hemodynamic model. For derivation
        details see Robinson et al., 2006, BOLD responses to
        stimuli. 
        
        Parameters
        ----------
        freqs : float
        
        Returns
        -------
        ndarray
            Transfer function
        """
        w = 2 * np.pi * freqs
        beta = (self._rho + (1.-self._rho)*np.log(1.-self._rho)) / self._rho
        T_yz = (self._V0 * (self._alpha * (self._k2 + self._k3) * (w*self._tau*1.j - 1) +
               (self._k1 + self._k2) * (self._alpha + beta - 1. - w *
                self._alpha*beta*self._tau*1.j))) / ((1. - w * self._tau*1.j) *
               (1. - self._alpha * w * self._tau*1.j) * (w**2 + w * self._kappa*1.j -
                                                     self._gamma))
        return T_yz * np.conj(T_yz)

    def reset_state(self):
        """
        Reset hemodynamic state variables to steady state values.
         
        Returns
        -------
        none
        """

        self._x = np.repeat(self._x0, self._nc) # vasodilatory signal
        self._f = np.repeat(self._f0, self._nc) # normalized inflow rate
        self._v = np.repeat(self._v0, self._nc) # normalized blood volume
        self._q = np.repeat(self._q0, self._nc) # norm. deoxyhemoglobin content
        self._y = np.repeat(self._y0, self._nc) # BOLD signal (%)

        return

    def nonlinear_step(self, dt, z):
        """
        Evolve hemodynamic equations by time dt and update state variables.
        System evolved according to Balloon - Windkessel hemodynamic model.
        
        Parameters
        ----------
        dt : float
            Time step of the integration
        z : ndarray
            Synaptic activity in each brain region
            
        Returns
        -------
        none
        """

        # Hemodynamic response equations
        dx = dt * (z - self._kappa * self._x - self._gamma * (self._f - 1.))
        df = dt * self._x
        dv = (dt / self._tau) * ((self._f - 1.) - (self._v - 1.) / self._alpha)
        x1 = (1. + (1. / self._rho) * (1. - self._rho) * np.log(1. - self._rho)) * (self._f - 1.)
        x2 = -(self._q - 1.)
        x3 = (self._alpha - 1) * (self._v - 1.) / self._alpha
        dq = dt * (x1 + x2 + x3) / self._tau

        # Add differences to initial values
        self._x += dx
        self._f += df
        self._v += dv
        self._q += dq

        self._y = self._V0 * (self._k1 * (1. - self._q) + self._k2 * (
            1. - self._q / self._v) + self._k3 * (1. - self._v))

        return

    def linear_step(self, dt, z):
        """ Evolve linearized hemodynamic equations by time dt, 
        and update state variables. System evolved according to linearized 
        Balloon - Windkessel hemodynamic model.
        
        Parameters
        ----------
        dt : float
            Time step of the integration
        z : ndarray
            Synaptic activity in each brain region
            
        Returns
        -------
        none
        """
        dx = dt * (z - self._kappa * self._x - self._gamma * (self._f - 1.))
        df = dt * self._x

        num = self._f - np.power(self._v, 1. / self._alpha)
        if isinstance(num, complex):
            raise ValueError("Complex value encountered.")
        dv = dt * num / self._tau

        dq = dt * (self._f * (1. - np.power(1. - self._rho, 1. / self._f)) / self._rho - self._q * (
            np.power(self._v, 1. / self._alpha - 1.))) / self._tau

        # Update state
        self._x += dx
        self._f += df
        self._v += dv
        self._q += dq

        # Obata 2004
        self._y = self._V0 * ((self._k1 + self._k2) * (
                1. - self._q) + (self._k3 - self._k2) * (1. - self._v))

        return

    def linearize_BOLD(self, z, A_syn, Q):
        """
        Calculates the Jacobian of the full system.
        Such that three numpy arrays are produced:
        (2n x 2n) S covariance matrix
        (4n x 4n) hemodynamic covariance matrix
        (n x n)   BOLD covariance matrix
                 
        Parameters
        ----------
        z : ndarray
            The steady state values synaptic gating parameters
        A_syn : ndarray
            The Jabobian matrix for the synaptic system of equations
        Q : ndarray
            The input noise matrix for the synaptic system of equations
            
        Returns
        -------
        none
        """

        # Define variables for some recurring matrices
        idmat = np.eye(self._nc)
        zeromat = np.zeros(shape=(self._nc, self._nc))

        # Partials of x w.r.t. hemodynamic quantities x, f, v, q
        A_xx = -self._kappa * idmat
        A_xf = -self._gamma * idmat
        A_xv = zeromat
        A_xq = zeromat

        # Partials of f w.r.t. hemodynamic quantities x, f, v, q
        A_fx = idmat
        A_ff = zeromat
        A_fv = zeromat
        A_fq = zeromat

        # Partials of v w.r.t. hemodynamic quantities x, f, v, q
        A_vx = zeromat
        A_vf = idmat / self._tau
        A_vv = -idmat / (self._alpha * self._tau)
        A_vq = zeromat

        # Partials of q w.r.t. hemodynamic quantities x, f, v, q
        A_qx = zeromat
        A_qf = idmat / self._tau * (
                1. + (1 - self._rho) * np.log(1 - self._rho) / self._rho)
        A_qv = idmat * (self._alpha - 1.) / (self._tau * self._alpha)
        A_qq = idmat * -1. / self._tau

        # Construct hemodynamic sub-block
        hemo_row1 = np.hstack((A_xx, A_xf, A_xv, A_xq))
        hemo_row2 = np.hstack((A_fx, A_ff, A_fv, A_fq))
        hemo_row3 = np.hstack((A_vx, A_vf, A_vv, A_vq))
        hemo_row4 = np.hstack((A_qx, A_qf, A_qv, A_qq))
        hemo_block = np.vstack((hemo_row1, hemo_row2, hemo_row3, hemo_row4))

        # Dependence of hemodynamic state on vasodilatory signal z only enters in
        # equation for dxdt (assumed through S_E only)
        dState_dSE = np.vstack((idmat, np.zeros((3 * self._nc, self._nc))))
        dState_dS = np.hstack((dState_dSE, np.zeros((4 * self._nc, self._nc))))

        # Construct full 6x6 Jacobian with (S_E, S_I, x, f, v, q)
        input_row = np.hstack((A_syn, np.zeros((2 * self._nc, 4 * self._nc))))
        state_row = np.hstack((dState_dS, hemo_block))

        # Use Lyapunov equation on full Jacobian to solve for full covariance
        self.full_A = np.vstack((input_row, state_row))
        self.full_Q = np.pad(Q, pad_width=(0, 4 * self._nc), mode='constant', constant_values=0)

        # Dependence of BOLD output signal on state variables
        dydv = idmat * (self._k2 - self._k3) * self._V0
        dydq = -idmat * (self._k1 + self._k2) * self._V0
        self.B = np.hstack((np.zeros((self._nc, self._nc * 4)), dydv, dydq))


    @property
    def state(self):
        """
        All hemodynamic state variables, 5 rows by len(nodes) columns.
            Rows are x, v, f, q, y. 
        
        Returns
        -------
        ndarray
            The collection of hemodynamic state variables.
            
        """
        return np.vstack((self._x, self._f, self._v, self._q, self._y))
