# PL models for semiconductor emission
# Author  : Stephen Church <stephen.church@manchester.ac.uk>
# Version : 0.1

import math
import numpy as np

# Physical Constants
k = 8.6173324 * 1e-5
planck = 1.054571 * 1e-34
m0 = 9.10938356 * 1e-31

# GaAs constants
# TODO: Implement for different materials
E_gaas = 1.424
mqe = 0.063 * m0
mbe = 0.097 * m0
me = 0.063 * m0
mh = 0.51 * m0


########################################
# Building blocks for models
########################################
def gauss(x, par):
    output = []
    for element in x:
        output.append(math.exp(-(element - par[0]) ** 2 / abs(par[1]) ** 2))
    return np.array(output)


def dos_3d_boltz(x, par):
    output = []
    for element in x:
        if element < par[0]:
            output.append(0)
        else:
            output.append((element - par[0]) ** 0.5 * math.exp(-(element - par[0]) / (k * par[1])))
    return np.array(output)


def dos_2d_boltz(x, par):
    output = []
    for element in x:
        if element < par[0]:
            output.append(0)
        else:
            output.append(math.exp(-(element - par[0]) / (k * par[1])))
    return np.array(output)


def urbach(x, par):
    output = []
    for element in x:
        if element > par[0]:
            output.append(0)
        else:
            output.append(math.exp((element - par[0]) / (par[1])))
    return np.array(output)


# E0, T, EF
def lsw_2d(x, par):
    output = []
    for element in x:
        if element > par[0]:
            num = (element - par[0]) ** 2 * (1 - 0.5) * (
                        1 - 2 / (math.exp((element - par[2]) / (k * par[1])) ** 0.5 + 1))
        else:
            num = 0

        den = 1 / (math.exp((element - par[2]) / (k * par[1])) - 1)
        output.append(num * den)
    return np.array(output)


# E0, T, EF
def lsw_3d(x, par):
    output = []
    for element in x:
        if element > par[0]:
            num = (element - par[0]) ** 2 * (1 - math.exp(-(element - par[0]))) * (
                        1 - 2 / (math.exp((element - par[2]) / (k * par[1])) ** 0.5 + 1))
        else:
            num = 0

        den = 1 / (math.exp((element - par[2]) / (k * par[1])) - 1)
        output.append(num * den)
    return np.array(output)

########################################
# PL models
########################################
# Arbitrary center point for gaussian, chosen to not disturb peak of PL
def pl_3d_boltz(x, par):
    a = gauss(x, [x[0] / 9.9 + x[-1], par[0]])  # 5.95 for full wavelength range
    # plt.plot(x,a/max(a))
    b = dos_3d_boltz(x, [par[1], par[2]])
    # plt.plot(x,b/max(b))
    return par[3] * np.convolve(a, b, 'same')


# Arbitrary center point for gaussian, chosen to not disturb peak of PL
def pl_2d_boltz(x, par):
    a = gauss(x, [x[0] / 9.9 + x[-1], par[0]])
    # plt.plot(x,a/max(a))
    b = dos_2d_boltz(x, [par[1], par[2]])
    # plt.plot(x,b/max(b))
    return par[3] * np.convolve(a, b, 'same')


def pl_2d_urbach(x, par):
    a = gauss(x, [x[0] / 9.9 + x[-1], par[0]])
    urb = urbach(x, [par[1], par[4]])
    b = dos_2d_boltz(x, [par[1], par[2]])

    c = urb / max(urb) + b / max(b)
    # plt.plot(x,urb)
    # plt.plot(x,b)
    return par[3] * np.convolve(a, c, 'same')


# TODO: ISSUE WITH JOINING OF TWO SIDES OF MODEL
def pl_3d_urbach(x, par):
    a = gauss(x, [x[0] / 9.9 + x[-1], par[0]])
    urb = urbach(x, [par[1], par[4]])
    b = dos_3d_boltz(x, [par[1], par[2]])
    c = urb / max(urb) + b / max(b)
    # plt.plot(x,urb)
    # plt.plot(x,b)
    return par[3] * np.convolve(a, c, 'same')


def pl_2d_lsw(x, par):
    a = gauss(x, [x[0] / 9.9 + x[-1], par[0]])
    b = lsw_2d(x, [par[1], par[2], par[3]])

    return par[4] * np.convolve(a, b, 'same')


def pl_3d_lsw(x, par):
    a = gauss(x, [x[0] / 9.9 + x[-1], par[0]])
    b = lsw_3d(x, [par[1], par[2], par[3]])

    return par[4] * np.convolve(a, b, 'same')
