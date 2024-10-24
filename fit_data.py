#!/bin/env python

from scipy.optimize import least_squares, curve_fit
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt

def sigmoid(xs, alpha, beta, gamma):
    return numpy.array([alpha / (1 + math.e**(beta * (x - gamma))) for x in xs])

data = pd.read_csv("output.csv", skipinitialspace=True)
data = data[['nsec', 'read_clat_ns_count', 'read_clat_ns_cumulative', 'read_clat_ns_percentile','write_clat_ns_count', 'write_clat_ns_cumulative', 'write_clat_ns_percentile',]]

def sigmoid_error(s):
    return data['read_clat_ns_cumulative'] - sigmoid(data['nsec'], *s)

# TODO: Initial seeding is hand-tunded and might not work for all cases.
res_curve, cov = curve_fit(
    sigmoid,
    data['nsec'].to_numpy(),
    data['read_clat_ns_cumulative'].to_numpy(),
    [0, 0, 2_000_000],
    bounds = (
        [0, -numpy.inf, data['nsec'].min()],
        [data['read_clat_ns_count'].cumsum().max(), numpy.inf, data['nsec'].max()],
    ),
)

# plot the function compared to the real data
fig, ax = plt.subplots(1,1)
ax.scatter(data['nsec'], data['read_clat_ns_cumulative'], label="Reference")
ax.plot(data['nsec'], sigmoid(data['nsec'], *res_curve), label="CF")
fig.legend()
fig.show()
