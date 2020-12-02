# PL models for semiconductor emission
# Author  : Stephen Church <stephen.church@manchester.ac.uk>
# Version : 0.2

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Physical Constants
boltzmann_kb = 8.6173324 * 1e-5
planck_constant = 1.054571 * 1e-34
electron_mass = 9.10938356 * 1e-31

# GaAs constants
material_constants = {'GaAs': {}}
material_constants['GaAs']['E'] = 1.424
material_constants['GaAs']['mqe'] = 0.063 * electron_mass
material_constants['GaAs']['mbe'] = 0.097 * electron_mass
material_constants['GaAs']['me'] = 0.063 * electron_mass
material_constants['GaAs']['mh'] = 0.51 * electron_mass


########################################
# Building blocks for models
########################################

def gauss(x, par):
    output = np.exp(-(x - np.take(x,x.size/2)) ** 2 / abs(par[1]) ** 2)
    return output


def dos_3d_boltz(x, par):
    output = np.power(np.fmax((x - par[0]), 0), 0.5) * np.exp(-(x - par[0]) / (boltzmann_kb * par[1]))
    output = np.where(x>par[0], output, 0)
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
# TODO: Gaussian only requires a single parameter - width. Need to change throughout
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


#######################################################
# class to fit spectra
# works for example: fitter = PLfit(pl_2d_boltz)
#                    output = fitter(s)
#                    bounds and initial params must be set, others optional
#                    note that number of parameters needed depends on the model selected
#######################################################
class PLfit():

    def __repr__(self):
        return "PL fitting class, using ({}) model with [{}] starting parameters".format(
            self.model.__name__, self.par0)

    def __init__(self, model):
        self.model = model  # model string from definitions above
        self.bounds = []
        self.par0 = []

        # defines the x axis of the spectrum (energy)
        i = np.arange(1, 1045)
        self.wavelength = (346 + 0.796 * i - 3.79e-5 * i ** 2)
        self.eV = 1239.842 / np.array(self.wavelength)

        # options for fitting
        self.remove_baseline = False  # remove of constant baseline
        self.spectral_correct = False  # correction with spectral response
        self.clip_spectra = False  # choose a section of the spectrum to analyse
        self.normalise_spectrum = True
        self.plot_output = True  # plot the data and fit spectrum
        self.ftol = 1e-4  # Settings for optimizer
        self.tol = 1e-9  # Settings for optimizer

        # limits and thresholds
        self.width_thresh = 10  # will remove spectra with width less than this (cosmic rays) (in spectrum increments)
        self.A_lim = 3  # Amplitude limit, below which ignore the spectrum
        self.spec_lims = [np.min(self.eV), np.max(self.eV)]  # spectral limits for clipping (eV)
        self.response = np.array(1044 * [1])  # spectral response field, default unity unless imported

        # results
        self.output_par = []
        self.output_PL = []
        self.output_res = []  # squared residuals

    # calling the class fits the spectrum
    def __call__(self, spec):

        if self.clip_spectra:
            ind = np.where(np.logical_and(self.eV >= self.spec_lims[0], self.eV <= self.spec_lims[1]))
            eV = self.eV[ind]
            data = spec[ind]
        else:
            ind = np.ones(self.eV.shape)
            data = spec
            eV = self.eV

        # calculate and subtract constant baseline
        if self.remove_baseline:
            data = data - (data[-1] + data[-2] + data[-3]) / 3

        # check to see if spectrum is bright
        if np.max(data) < self.A_lim:
            print('very dim sample - skipped')
            return

        # correct for spectral response
        if self.spectral_correct:
            if self.clip_spectra:
                clipped_response = self.response[ind]
                data = data * clipped_response
            else:
                data = data * self.response

        # normalise spectrum
        if self.normalise_spectrum:
            data = data / max(data)

        # function to calculate the squared residual to minimise in fit
        def residuals(par):
            return np.sum((self.model(eV, par) - data) ** 2)

        # do the fitting
            # do the fitting
        output = optimize.minimize(residuals, np.array(self.par0),
                                   method='Powell', tol=self.tol,
                                   options={'ftol': self.ftol}, bounds=self.bounds)
        # output the parameters
        self.init_PL = self.model(eV, self.par0)
        self.output = output
        self.output_PL = self.model(self.eV, output.x)

        # Show the fit if required
        if self.plot_output == 1:
            plt.plot(eV, data)
            plt.plot(self.eV, self.output_PL)
            plt.xlabel('Energy (eV)')
            plt.ylabel('PL intensity')
            plt.show()

        return output.x