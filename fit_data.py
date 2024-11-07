#!/bin/env python3
import math

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from scipy.optimize import curve_fit
import sys

"""
This scripts does:
    - Find an approximated function for IO latency count described by the input
    - Inverse a normalized version of this approximation to find the relation
      percentile -> latency with another approximation
    - The output will be created next to the input prepended with "param"

The input is a csv file created by fio accompanying
`tools/fio_jsonplus_clat2csv` script.

The name of the input is structured:

    <name>_<blocksize in B>_<rw ratio>_<queue_depth>

The inverse function is a single logit given with 5 constants which are
determined in the script:
      
    h(x) = euler^c * (a / (x*gap - 1))^(1/b)

The result of this function is the time of the given percentile in nanoseconds.

The results is given as another csv file with the following format:

    blocksize, rwratio, a, b, c, d, e


"""

if len(sys.argv) < 2:
    print("Usage:")
    print(f"{sys.argv[0]} <PATH_TO_INPUT_CSV>")

input = sys.argv[1]

block_size = input.split("_")[1]
rw_ratio = input.split("_")[2]
queue_depth = input.split("_")[3]

def sigmoid(x, alpha, beta, gamma):
    return alpha / (1 + math.e**(beta * (math.log(x) - gamma)))

def sigmoids(xs, alpha, beta, gamma):
    return numpy.array([sigmoid(x, alpha, beta, gamma) for x in xs])

def double_sigmoids(xs, a0, b0, c0, a1, b1, c1):
    return numpy.array([sigmoid(x, a0, b0, c0) for x in xs]) + numpy.array([sigmoid(x, a1, b1, c1) for x in xs])

data = pd.read_csv(input, skipinitialspace=True)
data = data[['nsec', 'read_clat_ns_count', 'read_clat_ns_cumulative', 'read_clat_ns_percentile','write_clat_ns_count', 'write_clat_ns_cumulative', 'write_clat_ns_percentile',]]

def inverse_sigmoid(x, alpha, beta, gamma):
    #return numpy.e**((numpy.log(alpha / x - 1) + beta * gamma) / beta)
    return numpy.e**gamma * (alpha / x - 1)**(1/beta)

def inverse_sigmoids(xs, alpha, beta, gamma):
    return numpy.array([inverse_sigmoid(x, alpha, beta, gamma) for x in xs])

def single_sigmoid_approx(column):
    """
    Perform an approximation using a single sigmoid function.
    """

    if not data[column].cumsum().max() > 10:
        print(f"Column {column} contains not enough requests.")
        return

    def sigmoid_error(s):
        return data[column] - sigmoid(data['nsec'], *s)

    def lower_bound():
        return [0, -numpy.inf, math.log(data['nsec'].min())]

    def upper_bound():
        return [data[column].cumsum().max(), numpy.inf, math.log(data['nsec'].max())]

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

    # plot the function compared to the real data
    fig, axs = plt.subplots(1,4, figsize=(16,5))

    axs[0].scatter(data['nsec'], data[column], label="Reference")
    axs[0].plot(data['nsec'], sigmoids(data['nsec'], *res_curve), label="CF")
    axs[0].set_xlabel("Time [nsec]")
    axs[0].set_ylabel("Requests [#]")
    axs[0].set_title("Cumulative\nNumber of Requests")

    axs[1].plot(data['nsec'], (data[column] - sigmoids(data['nsec'], *res_curve)) / data['nsec'] * 100 )
    axs[1].set_title("Error of\nCumulative Distribution")
    axs[1].set_xlabel("Time [nsec]")
    axs[1].set_ylabel("Error [%]")

    percentages = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    gap = data[column].max() - data[column].min()

    axs[2].plot(percentages, [inverse_sigmoid(x * gap, *res_curve) for x in percentages])
    axs[2].scatter(
        percentages,
        data['nsec'],
    )
    axs[2].set_title("Inverse of Normalized\n Cumulative Distribution (CF)")
    axs[2].set_xlabel("Percentiles")
    axs[2].set_ylabel("Time [nsec]")

    axs[3].plot(
        percentages,
        ([inverse_sigmoid(x * gap, *res_curve) for x in percentages] - data['nsec'])/data['nsec']*100,
    )
    axs[3].set_title("Error of INCD")
    axs[3].set_xlabel("Percentiles")
    axs[3].set_ylabel("Error [%]")


    fig.legend()
    fig.tight_layout()
    fig.savefig(f"output_{column}.svg")

    with open(f"param_{input}.csv", 'a', encoding="utf-8") as file:
        op = column.split("_")[0]
        file.write(f"{block_size},{op},{rw_ratio},{gap}")
        for x in res_curve:
            file.write(f",{x}")
        file.write("\n")

    print(column)
    print("f(x):")
    print(f"\t alpha: {res_curve[0]}")
    print(f"\t beta: {res_curve[1]}")
    print(f"\t gamma: {res_curve[2]}")


with open(f"param_{input}.csv", 'w', encoding="utf-8") as file:
        file.write("blocksize,op,rw,gap,a,b,c\n")

single_sigmoid_approx("read_clat_ns_cumulative")
single_sigmoid_approx("write_clat_ns_cumulative")
