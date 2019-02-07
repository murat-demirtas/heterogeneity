"""
Hemodynamic parameters for Balloon-Windkessel model according to Obata 2004.
"""


V0 = 0.02     # resting blood volume fraction
kappa = 0.65  # [s^-1] rate of signal decay
gamma = 0.41  # [s^-1] rate of flow-dependent elimination
tau = 0.98    # [s] hemodynamic transit time
alpha = 0.32  # Grubb's exponent
rho = 0.34    # resting oxygen extraction fraction

k1 = 3.72
k2 = 0.527
k3 = 0.53

'''
V0 = 0.02     # resting blood volume fraction
kappa = 0.65  # [s^-1] rate of signal decay
gamma = 0.41  # [s^-1] rate of flow-dependent elimination
tau = 0.98    # [s] hemodynamic transit time
alpha = 0.32  # Grubb's exponent
rho = 0.34    # resting oxygen extraction fraction

k1 = 7 * rho
k2 = 1.43 * rho
k3 = 0.43
'''