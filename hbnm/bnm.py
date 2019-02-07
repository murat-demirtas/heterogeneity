import numpy as np
from model import dmf as dmf_model

class Bnm():
    """
    Wrapper class for the large-scale computational model.
    """

    def __init__(self, sc, gradient = None, *args, **kwargs):
        """

        Parameters
        ----------
        sc : ndarray
            Structural connectivity matrix
        gradient : ndarray, optional
            Heterogeneity map to scale local model parameters. 
            If None, the model parameters are homogeneous (None by default)
        
        Notes
        -----
        The optional arguments and keyword arguments pass to the Model class.
        If the model is separated for left and right hemispheres, the SC and heterogeneity map
         should be given as a list as [left, right]. In this case, the wrapper generates 2 models
         as a list in the same order that is [left, right].

        """
        self.sc = sc
        self.gradient = gradient

        if isinstance(self.sc, list):
            if not isinstance(gradient, list):
                self.gradient = [self.gradient, self.gradient]
            self.dmf = [dmf_model.Model(self.sc[ii], g=1.0, hmap=self.gradient[ii],
                                        verbose=False, *args, **kwargs) for ii in range(2)]
        else:
            self.dmf = dmf_model.Model(self.sc, g=1.0, hmap=self.gradient,
                                       verbose=False, *args, **kwargs)

    def set(self, parameter, values, separate = False):
        """
        Set method for the model
        
        Parameters
        ----------
        parameter : str
            The name of the parameter, such as 'w_EE' or 'w_EI'
        values : float or tuple
            The parameter values to set. The values should be tuple for heterogeneous model.
        separate : bool, optional
            If True, the value passes to left and right hemisphere separately. This flag is 
             required only if the global coupling parameter is separate for each hemisphere. 
        """

        if isinstance(self.dmf, list):
            if separate:
                [setattr(self.dmf[ii], parameter, values[ii]) for ii in range(2)]
            else:
                [setattr(self.dmf[ii], parameter, values) for ii in range(2)]
        else:
            setattr(self.dmf, parameter, values)

    def get(self, parameter):
        """
        Get method for the model
        
        Parameters
        ----------
        parameter : str
            The parameter to get, such as 'w_EE', 'w_EI', 'corr_bold'...etc. 
        """

        if isinstance(self.dmf, list):
            return [self.dmf[ii].__getattribute__(parameter) for ii in range(2)]
        else:
            return self.dmf.__getattribute__(parameter)

    def check_stability(self, compute_FIC=True):
        """
        Check stability of the system
        
        Parameters
        ----------
        compute_FIC : bool, optional
            If True, the feedback inhibition is recomputed.
        
        Returns
        -------
        bool
            The truth value for the largest eigenvalue of the system is smaller than 0
        
        """

        if isinstance(self.dmf, list):
            [self.dmf[ii].set_jacobian(compute_fic=compute_FIC) for ii in range(2)]
            return np.array([self.dmf[ii]._unstable for ii in range(2)]).any()
        else:
            self.dmf.set_jacobian(compute_fic=compute_FIC)
            return self.dmf._unstable

    def moments_method(self, BOLD = True, *args, **kwargs):
        """
        Calls moments_method method in the Model class
        
        Parameters
        ----------
        BOLD : bool, optional
            If True, the linearized covariances are computed for hemodynamic system (BOLD) 
        """

        if isinstance(self.dmf, list):
            [self.dmf[ii].moments_method(bold = BOLD, *args, **kwargs) for ii in range(2)]
        else:
            self.dmf.moments_method(bold = BOLD, *args, **kwargs)

    def psd_syn(self, freqs, pop='E'):
        """
        Calculates the power spectral density for the synaptic gating variables
        
        Parameters
        ----------
        freqs : ndarray
            The frequency bins for which the power spectral density is computed
        pop : str, optional
            If 'E', the power spectral density is computed for excitataory populations,
            if 'I', the power spectral density is computed for inhibitory populations.
        
        Returns
        -------
        ndarray
            Power spectral density
        """

        if isinstance(self.dmf, list):
            csd_1 = self.dmf[0].csd(freqs, pop=pop)
            csd_2 = self.dmf[1].csd(freqs, pop=pop)
            psd_E_1 = np.abs(np.array([np.diag(csd_1[:, :, ii]) for ii in xrange(len(freqs))]).T)
            psd_E_2 = np.abs(np.array([np.diag(csd_2[:, :, ii]) for ii in xrange(len(freqs))]).T)
            return [psd_E_1, psd_E_2]
        else:
            csd = self.dmf.csd(freqs, pop=pop)
            return np.abs(np.array([np.diag(csd[:, :, ii]) for ii in xrange(len(freqs))]).T)

    def psd_bold(self, freqs):
        """
        Calculates the power spectral density for the BOLD activity

        Parameters
        ----------
        freqs : ndarray
            The frequency bins for which the power spectral density is computed
            
        Returns
        -------
        ndarray
            Power spectral density
        """
        if isinstance(self.dmf, list):
            csd_1 = self.dmf[0].csd_bold(freqs)
            csd_2 = self.dmf[1].csd_bold(freqs)
            psd_E_1 = np.abs(np.array([np.diag(csd_1[:, :, ii]) for ii in xrange(len(freqs))]).T)
            psd_E_2 = np.abs(np.array([np.diag(csd_2[:, :, ii]) for ii in xrange(len(freqs))]).T)
            return [psd_E_1, psd_E_2]
        else:
            csd = self.dmf.csd_bold(freqs)
            return np.abs(np.array([np.diag(csd[:, :, ii]) for ii in xrange(len(freqs))]).T)

    def disconnect(self):
        """
        Notes
        -----
            This method creates another Model object named 'dmf_disconnected', the total input to each region
             is set to long-range input level and global coupling parameter is set to 0.
        """
        from copy import deepcopy
        self.dmf_disconnect = deepcopy(self.dmf)
        if isinstance(self.dmf_disconnect, list):
            for ii in range(2):
                self.dmf_disconnect[ii].I_ext = (self.dmf[ii].G * self.dmf[ii]._sc_norm * self.dmf[ii].w_EE) \
                                                 * ((self.dmf[ii].SC.T).dot(self.dmf[ii]._S_E_ss))
                self.dmf_disconnect[ii].G = 0.0
                self.dmf_disconnect[ii].moments_method(bold=False)
        else:
            self.dmf_disconnect.I_ext = (self.dmf.G * self.dmf._sc_norm * self.dmf.w_EE) \
                                             * ((self.dmf.SC.T).dot(self.dmf._S_E_ss))
            self.dmf_disconnect.G = 0.0
            self.dmf_disconnect.moments_method(bold=False)
