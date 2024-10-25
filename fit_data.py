#!/bin/env python
from scipy.optimize import curve_fit
import pandas as pd
import math
import numpy
import matplotlib.pyplot as plt

"""
This scripts does:
    - Find an approximated function for IO latency count described by
      output.csv
    - Inverse a normalized version of this approximation to find the relation
      percentile -> latency with another approximation

The input is a csv file created by fio accompanying
`tools/fio_jsonplus_clat2csv` script.

The inverse function is a single logit given with 5 constants which are
determined in the script:

h(x) = a * log((x*c+d)/(1-x+e)) + b

The result of this function is the time of the given percentile in nanoseconds.

The results is given as another csv with the following format:

blocksize, rwratio, a, b, c, d, e
"""

def sigmoid(x, alpha, beta, gamma):
    return alpha / (1 + math.e**(beta * (math.log(x) - gamma)))

def sigmoids(xs, alpha, beta, gamma):
    return numpy.array([sigmoid(x, alpha, beta, gamma) for x in xs])

def double_sigmoids(xs, a0, b0, c0, a1, b1, c1):
    return numpy.array([sigmoid(x, a0, b0, c0) for x in xs]) + numpy.array([sigmoid(x, a1, b1, c1) for x in xs])

data = pd.read_csv("input.csv", skipinitialspace=True)
data = data[['nsec', 'read_clat_ns_count', 'read_clat_ns_cumulative', 'read_clat_ns_percentile','write_clat_ns_count', 'write_clat_ns_cumulative', 'write_clat_ns_percentile',]]

def sigmoid_error(s):
    return data['read_clat_ns_cumulative'] - sigmoid(data['nsec'], *s)

def lower_bound():
    return [0, -numpy.inf, math.log(data['nsec'].min())]

def upper_bound():
    return [data['read_clat_ns_count'].cumsum().max(), numpy.inf, math.log(data['nsec'].max())]

# logistic regression over measured data
res_curve, cov = curve_fit(
    sigmoids,
    data['nsec'].to_numpy(),
    data['read_clat_ns_cumulative'].to_numpy(),
    [0, 0, math.log(data['nsec'].min())],
    bounds = (
        lower_bound(),
        upper_bound(),
    ),
)

double_res_curve, _ = curve_fit(
    double_sigmoids,
    data['nsec'].to_numpy(),
    data['read_clat_ns_cumulative'].to_numpy(),
    [*res_curve, 0, 0, math.log(data['nsec'].min())],
    bounds = (
        lower_bound() + lower_bound(),
        upper_bound() + upper_bound(),
    ),
)

# we got f(x) now
#
# let's get g(x) next

approx_res = sigmoids(data['nsec'], *res_curve)
lower_bound = approx_res[0]
upper_bound = approx_res[len(approx_res) - 1]

def normalized_sigmoid(x):
    return sigmoid(x, *res_curve) / (upper_bound - lower_bound)

# bounded_res = approx_res / (upper_bound - lower_bound)

# with pynverse
# import pynverse
#
#inverse_normalized_sigmoid = pynverse.inversefunc(normalized_sigmoid, domain=[data['nsec'].min(), data['nsec'].max()])

def inverse_sigmoid(x, alpha, beta, gamma, delta, epsilon):
    if x <= 0:
        return 0
    elif x >= 1.0:
        return data['nsec'].max()
    else:
        if 0 > (x*gamma+delta) / (1-x+epsilon):
            return numpy.nan
        return (alpha * numpy.log((x*gamma+delta) / (1-x+epsilon)) + beta)

def inverse_sigmoids(xs, alpha, beta, gamma, delta, epsilon):
    return numpy.array([inverse_sigmoid(x, alpha, beta, gamma, delta, epsilon) for x in xs])

inv_curve, _ = curve_fit(
    inverse_sigmoids,
    [normalized_sigmoid(x) for x in data['nsec']],
    data['nsec'],
    [0, 0, data['nsec'].min(), 1, 1],
    bounds = (
        [-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf],
        [numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf],
    )
)

# plot the function compared to the real data
fig, axs = plt.subplots(2,3, figsize=(12,10))
axs[0][0].scatter(data['nsec'], data['read_clat_ns_cumulative'], label="Reference")
axs[0][0].plot(data['nsec'], sigmoids(data['nsec'], *res_curve), label="CF")
axs[0][0].set_xlabel("Time [nsec]")
axs[0][0].set_ylabel("Requests [#]")
axs[0][0].set_title("Cumulative\nNumber of Requests")

axs[0][1].plot(data['nsec'], [normalized_sigmoid(x) for x in data['nsec']])
axs[0][1].set_title("Normalized\nCumulative Distribution")
axs[0][1].set_xlabel("Time [nsec]")
axs[0][1].set_ylabel("Percentiles")

percentages =[normalized_sigmoid(x) for x in data['nsec']] 

#axs[2].plot(percentages, [inverse_normalized_sigmoid(x) for x in percentages])
#axs[2].scatter(
#    percentages,
#    data['nsec'],
#)
#axs[2].set_title("Inverse of Normalized\n Cumulative Distribution (pynverse)")

axs[0][2].plot(percentages, [inverse_sigmoid(x, *inv_curve) for x in percentages])
axs[0][2].scatter(
    percentages,
    data['nsec'],
)
axs[0][2].set_title("Inverse of Normalized\n Cumulative Distribution (CF)")
axs[0][2].set_xlabel("Percentiles")
axs[0][2].set_ylabel("Time [nsec]")

axs[1][0].scatter(data['nsec'], data['read_clat_ns_cumulative'], label="Reference")
axs[1][0].plot(data['nsec'], double_sigmoids(data['nsec'], *double_res_curve), label="CF")

fig.legend()
fig.savefig("output.svg")


with open("output.csv", 'w', encoding="utf-8") as file:
    file.write("blocksize,rwratio,a,b,c,d,e\n")
    file.write("131072,1.0")
    for x in inv_curve:
        file.write(f",{x}")
    file.write("\n")


print("f(x):")
print(f"\t alpha: {res_curve[0]}")
print(f"\t beta: {res_curve[1]}")
print(f"\t gamma: {res_curve[2]}")

print("h(x):")
print(f"\t alpha: {inv_curve[0]}")
print(f"\t beta: {inv_curve[1]}")
print(f"\t gamma: {inv_curve[2]}")
print(f"\t delta: {inv_curve[3]}")
print(f"\t epsilon: {inv_curve[4]}")

