# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:31:49 2025

Random data transit plot

@author: grt
"""

import numpy as np
import exoplanet as xo
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import pymc_ext as pmx
from astropy.timeseries import LombScargle
import arviz as az

random = np.random
random.seed(17)


num_transits = 178
period = 2.5 * np.pi  
total_days = 1400  
t = np.arange(0, total_days, 0.08) 
yerr = 5e-4

with pm.Model():
    mean = pm.Normal("mean", mu=0.0, sigma=0.2)

    # Define transits
    t0 = pm.Normal("t0", mu=1 * period - 2, sigma=0.05)
    t1 = pm.Normal("t1", mu=2 * period - 2, sigma=0.05)
    t2 = pm.Normal("t2", mu=3 * period - 2, sigma=0.05)
    t3 = pm.Normal("t3", mu=4 * period - 2, sigma=0.05)

    # The Kipping (2013) parameterization for quadratic limb darkening parameters
    u = xo.distributions.quad_limb_dark("u", initval=np.array([0.3, 0.2]))

    # The radius ratio and impact parameter
    log_r = pm.Normal("log_r", mu=np.log(0.04), sigma=1.0)
    r = pm.Deterministic("r", pt.exp(log_r))
    b = xo.distributions.impact_parameter("b", r, initval=0.35)

    # Set up a Keplerian orbit for the planets
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

    # Compute the model light curve
    light_curve = (
        xo.LimbDarkLightCurve(u[0], u[1]).get_light_curve(
            orbit=orbit, r=r, t=t
        )[:, 0]
        + mean
    )

    # ==================SIMULATED DATA================== #
    y = pmx.eval_in_model(light_curve)
    y += yerr * random.normal(size=len(y))
    # ==================================================#


t_folded = t % period


plt.figure(1)
plt.plot(t_folded, y, ".k", ms=2)
plt.xlabel("Time (t % period)")
plt.ylabel("Photon Flux")
plt.title("Photon Flux over Time (Folded on T = {})".format(period))
plt.xlim(0, period)  
plt.legend()
plt.grid()
plt.show()



# frequency = np.linspace(1/190, 1/150, 1000)
# power = LombScargle(t, y, yerr).power(frequency)
# plt.figure(2)
# plt.plot(frequency,power)
# best_frequency = frequency[np.argmax(power)]
# T = 1 / best_frequency
# print("lombscargle says the period is", T)
