# PL models for semiconductor emission
# Author  : Stephen Church <stephen.church@manchester.ac.uk>
# Version : 0.2

import numpy as np
from scipy import optimize
from scipy import signal
import matplotlib.pyplot as plt

# Physical Constants
boltzmann_kb = 8.6173324 * 1e-5
planck_constant = 1.054571 * 1e-34
electron_mass = 9.10938356 * 1e-31

# GaAs constants
# TODO: Implement for different materials
E_gaas = 1.424
mqe = 0.063 * electron_mass
mbe = 0.097 * electron_mass
me = 0.063 * electron_mass
mh = 0.51 * electron_mass


########################################
# Building blocks for models
########################################

def gauss(x, par):
    output = np.exp(-(x - par[0]) ** 2 / abs(par[1]) ** 2)
    return output


def dos_3d_boltz(x, par):
    output = np.power((x - par[0]), 0.5) * np.exp(-(x - par[0]) / (boltzmann_kb * par[1]))
    output = np.nan_to_num(output)
    return output


def dos_2d_boltz(x, par):
    output = np.exp(-(x - par[0]) / (boltzmann_kb * par[1]))
    output = np.where(output > 1, 0, output)
    return output


def urbach(x, par):
    output = np.exp((x - par[0]) / (par[1]))
    output = np.where(output > 1, 0, output)
    return output


# E0, T, EF
def lsw_2d(x, par):
    num = 0.5 * np.multiply(np.power((x - par[0]), 2),
                            (1 - np.divide(2, (np.exp((x - par[2]) / (boltzmann_kb * par[1])) ** 0.5 + 1))))
    den = (np.exp((x - par[2]) / (boltzmann_kb * par[1])) - 1)
    output = np.divide(num, den)
    output = np.where(x < par[0], 0, output)
    return np.array(output)


# E0, T, EF
def lsw_3d(x, par):
    num = np.multiply((1 - np.exp(-(x - par[0]))), np.multiply(np.power((x - par[0]), 2), (
                1 - np.divide(2, (np.exp((x - par[2]) / (boltzmann_kb * par[1])) ** 0.5 + 1)))))
    den = (np.exp((x - par[2]) / (boltzmann_kb * par[1])) - 1)
    output = np.divide(num, den)
    output = np.where(x < par[0], 0, output)
    return output


########################################
# PL models
########################################

def pl_3d_boltz(x, par):
    a = gauss(x, [(x[0] + x[-1]) / 2, par[0]])  # 5.95 for full wavelength range
    b = dos_3d_boltz(x, [par[1], par[2]])
    return par[3] * np.convolve(a, b, 'same')


def pl_2d_boltz(x, par):
    a = gauss(x, [(x[0] + x[-1]) / 2, par[0]])
    b = dos_2d_boltz(x, [par[1], par[2]])
    return par[3] * np.convolve(a, b, 'same')


def pl_2d_urbach(x, par):
    a = gauss(x, [(x[0] + x[-1]) / 2, par[0]])
    urb = urbach(x, [par[1], par[4]])
    b = dos_2d_boltz(x, [par[1], par[2]])
    c = urb / max(urb) + b / max(b)
    return par[3] * np.convolve(a, c, 'same')


# TODO: ISSUE WITH JOINING OF TWO SIDES OF MODEL
def pl_3d_urbach(x, par):
    a = gauss(x, [(x[0] + x[-1]) / 2, par[0]])
    urb = urbach(x, [par[1], par[4]])
    b = dos_3d_boltz(x, [par[1], par[2]])
    c = urb / max(urb) + b / max(b)
    return par[3] * np.convolve(a, c, 'same')


def pl_2d_lsw(x, par):
    a = gauss(x, [(x[0] + x[-1]) / 2, par[0]])
    b = lsw_2d(x, [par[1], par[2], par[3]])
    return par[4] * np.convolve(a, b, 'same')


def pl_3d_lsw(x, par):
    a = gauss(x, [(x[0] + x[-1]) / 2, par[0]])
    b = lsw_3d(x, [par[1], par[2], par[3]])
    return par[4] * np.convolve(a, b, 'same')


# class to fit spectra
# works for example: fitter = PLfit('pl_2d_boltz')
#                    output = fitter(s)
#                    bounds and initial params must be set, others optional
#                    note that number of parameters needed depends on the model selected
class PLfit():
    def __init__(self, model):
        self.model = model  # model string from definitions above
        self.bounds = []
        self.par0 = []

        # defines the x axis of the spectrum (energy)
        wavelength = []
        for i in range(1044):
            wavelength.append(346 + 0.796 * i - 3.79e-5 * i ** 2)
        self.eV = 1239.842 / np.array(wavelength)

        # options for fitting
        self.baseline_rem = 0  # remove of constant baseline
        self.spectral_correct = 0  # correction with spectral response
        self.clip_spectra = 0  # choose a section of the spectrum to analyse
        self.plot_output = 1  # plot the data and fit spectrum

        # limits and thresholds
        self.width_thresh = 10  # will remove spectra with width less than this (cosmic rays) (in spectrum increments)
        self.A_lim = 3  # Amplitude limit, below which ignore the spectrum
        self.spec_lims = [1.3, 1.7]  # spectral limits for clipping (eV)
        self.response = np.array(1044 * [1])  # spectral response field, default unity unless imported

        # results
        self.output_par = []
        self.output_PL = []
        self.output_res = []  # squared residuals

    # calling the class fits the spectrum
    def __call__(self, spec):

        # function to calculate the squared residual to minimise in fit
        def residuals(par):
            return np.sum((eval(self.model)(self.eV, par) - data) ** 2)

            # slects region of interest

        if self.clip_spectra == 1:
            ind = np.where(np.logical_and(self.eV >= self.spec_lims[0], self.eV <= self.spec_lims[1]))
            self.eV = self.eV[ind]
            data = spec[ind]
        else:
            data = spec

        # calculate and subtract constant baseline
        if self.baseline_rem == 1:
            # data = data - baseline_als(data,baseline_par[0],baseline_par[1])
            data = data - (data[-1] + data[-2] + data[-3]) / 3

        # check to see if spectrum is bright
        if np.max(data) < self.A_lim:
            print('very dim sample - skipped')
            return

            # correct for spectral response
        if self.spectral_correct == 1:
            if self.clip_spectra == 1:
                clipped_response = self.response[ind]
                data = data * clipped_response
            else:
                data = data * self.response

        # normalise spectrum
        data = data / max(data)

        # redefines upper bound to the peak of the PL
        j = np.argmax(data)
        upbound = self.eV[j]
        self.bounds[1] = (1, upbound)

        # check to see if cosmic ray based on width
        s = signal.peak_widths(data, [j], rel_height=0.5)
        if s[0][0] < self.width_thresh:
            print('Peak too narrow - skipped')
            return

        # do the fitting
        output = optimize.minimize(residuals, self.par0, method='Powell', tol=1e-7, bounds=self.bounds)

        # output the parameters
        self.init_PL = eval(self.model)(self.eV, self.par0)
        self.output_par = output.x
        self.output_PL = eval(self.model)(self.eV, output.x)
        self.output_res = output.fun

        if self.plot_output == 1:
            plt.plot(self.eV, data)
            # plt.plot(self.eV,self.init_PL)
            plt.plot(self.eV, self.output_PL)
