"""
Hemodynamic parameters for Balloon-Windkessel model according to Friston 2003.
"""

V0 = 0.02     # resting blood volume fraction
kappa = 0.65  # [s^-1] rate of signal decay
gamma = 0.41  # [s^-1] rate of flow-dependent elimination
tau = 0.98    # [s] hemodynamic transit time
alpha = 0.32  # Grubb's exponent
rho = 0.34    # resting oxygen extraction fraction

# Friston 2003 equations
k1 = 7.*rho
k2 = 2.
k3 = 2.*rho - 0.2
