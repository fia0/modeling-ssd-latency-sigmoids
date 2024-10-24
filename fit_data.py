#!/bin/env python

from scipy.optimize import least_squares, curve_fit
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt
import pynverse

def sigmoid(x, alpha, beta, gamma):
    return alpha / (1 + math.e**(beta * (x - gamma)))

def sigmoids(xs, alpha, beta, gamma):
    return numpy.array([sigmoid(x, alpha, beta, gamma) for x in xs])

#def inverse_sigmoid(xs, alpha, beta, gamma):
#    return numpy.array([(math.log(a/x - 1) + beta * gamma)/beta for x in xs])

data = pd.read_csv("output.csv", skipinitialspace=True)
data = data[['nsec', 'read_clat_ns_count', 'read_clat_ns_cumulative', 'read_clat_ns_percentile','write_clat_ns_count', 'write_clat_ns_cumulative', 'write_clat_ns_percentile',]]

def sigmoid_error(s):
    return data['read_clat_ns_cumulative'] - sigmoid(data['nsec'], *s)

# TODO: Initial seeding is hand-tunded and might not work for all cases.
res_curve, cov = curve_fit(
    sigmoids,
    numpy.log(data['nsec'].to_numpy()),
    data['read_clat_ns_cumulative'].to_numpy(),
    [0, 0, math.log(data['nsec'].min())],
    bounds = (
        [0, -numpy.inf, math.log(data['nsec'].min())],
        [data['read_clat_ns_count'].cumsum().max(), numpy.inf, math.log(data['nsec'].max())],
    ),
)

# we got f(x) now
#
# let's get g(x) next

# approx_res = sigmoids(numpy.log(data['nsec']), *res_curve)
# lower_bound = approx_res[0]
# upper_bound = approx_res[len(approx_res) - 1]
# 
# def normalized_sigmoid(x):
#     return sigmoid(math.log(x), *res_curve) / (upper_bound - lower_bound)
# 
# # bounded_res = approx_res / (upper_bound - lower_bound)
# 
# inverse_normalized_sigmoid = pynverse.inversefunc(normalized_sigmoid)

# plot the function compared to the real data
fig, axs = plt.subplots(1,3, figsize=(12,4))
axs[0].scatter(data['nsec'], data['read_clat_ns_cumulative'], label="Reference")
axs[0].plot(data['nsec'], sigmoid(numpy.log(data['nsec']), *res_curve), label="CF")
axs[0].set_xlabel("Time [nsec]")
axs[0].set_ylabel("Requests [#]")
axs[0].set_title("Cumulative Number of Requests")

#axs[1].plot(data['nsec'], [normalized_sigmoid(x) for x in data['nsec']])
#axs[1].set_title("Normalized Cumulative Distribution")
#
#axs[2].plot(numpy.arange(0, 1, 0.01), [inverse_normalized_sigmoid(x) for x in numpy.arange(0, 1, 0.01)])
#axs[2].set_title("Inverse of Normalized\n Cumulative Distribution")

fig.legend()
fig.savefig("output.svg")
