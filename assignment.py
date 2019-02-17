## Matrix Decompositions in Data Analysis
## Winter 2019
## Assignment file
## FILL IN the following information
## Name: Viet Ta
## Student ID: 299954

##
## This file contains stubs of code to help you to do your 
## assignment. You can fill your parts at the indicated positions
## and return this file as a part of your solution. 
##
## Remember to fill your name and student ID number above.
##
## This file is meant to be run with Python3

import numpy as np
from numpy import linalg
from numpy.linalg import svd, norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.stats import zscore
from scipy import stats
from collections import deque
from time import time

# The percentage of construction error change that is considered no longer significant
CONVERGENCE_THRESHOLD = 0.01


## In python, we can just write the code and it'll get called when the file is
## run with python3 assignment.py

## Task 1
##########

# Function that updates W and H using ALS
def nmf_als(A, w, h):
    # ADD YOUR code to update W and H
    h = linalg.pinv(w) @ A
    h[h < 0] = 0
    w = A @ linalg.pinv(h)
    w[w < 0] = 0
    return w, h


# Function that updates W and H using Lee and Seung multiplicative updates
def nmf_lns(A, w, h):
    h_denominator = w.T @ w @ h + np.finfo(float).eps
    h = h * (w.T @ A) / h_denominator
    w_denominator = w @ h @ h.T + np.finfo(float).eps
    w = w * (A @ h.T) / w_denominator
    w = w / np.sum(w)
    return w, h


# Function that updates W and H using OPL gradient descent
def nmf_opl(A, w, h):
    iter = 20
    WtW = w.T @ w
    h_eta = np.diag(1 / np.sum(WtW, 1))
    for i in range(iter):
        G = WtW @ h - w.T @ A
        h = h - h_eta @ G
        h[h < 0] = 0

    HHt = h @ h.T
    w_eta = np.diag(1 / np.sum(HHt, 1))
    for i in range(iter):
        G = w @ HHt - A @ h.T
        w = w - G @ w_eta
        w[w < 0] = 0
    return w, h


## Boilerplate for NMF
def nmf(A, k, optFunc=nmf_als, maxiter=300, repetitions=1):
    (n, m) = A.shape
    bestErr = np.Inf
    convergence_rep = [-1] * repetitions
    convergence_time = [0] * repetitions
    errors = [np.finfo(float).max] * repetitions
    for rep in range(repetitions):
        # Init W and H
        start_time = int(time() * 1000)
        W = np.random.rand(n, k)
        H = np.random.rand(k, m)
        errs = [np.nan] * maxiter
        recent_errors = deque([np.finfo(float).max] * 5)
        i = 0
        while convergence_rep[rep] < 0 and i < maxiter:
            (W, H) = optFunc(A, W, H)
            currErr = norm(A - np.matmul(W, H), 'fro') ** 2
            if abs(currErr - np.mean(recent_errors)) / currErr < CONVERGENCE_THRESHOLD \
                    and convergence_rep[rep] < 0:
                convergence_rep[rep] = i - 3
                errors[rep] = currErr
            recent_errors.append(currErr)
            recent_errors.popleft()
            errs[i] = currErr
            i += 1
        if convergence_rep[rep] < 0:
            convergence_rep[rep] = maxiter
            errors[rep] = errs.pop()
        if currErr < bestErr:
            bestErr = currErr
            bestW = W
            bestH = H
            bestErrs = errs
            best_index = rep
        end_time = int(time() * 1000)
        convergence_time[rep] = end_time - start_time
    return (bestW, bestH, errors, convergence_rep, convergence_time, bestErrs, best_index)


def test_optimize_func(A, k=20, opt_func=nmf_als, repetitions=100, name='NMF'):
    ## Sample use of nmf_als with A
    (W, H, errors, convergence, convergence_time, best_errs, best) = nmf(A, k, optFunc=opt_func, maxiter=100,
                                                                         repetitions=repetitions)

    ## To show the per-iteration error
    plt.plot(best_errs, label='Construction error')
    ax = plt.gca()
    plt.xlabel('Iterations')
    plt.ylabel('Squared Frobenius')
    plt.title('Convergence of {}'.format(name))
    l = mlines.Line2D([convergence[best], convergence[best]], [0, 100000], color='red', linestyle='--',
                      label='Convergence')
    ax.add_line(l)
    ax.legend(loc=9)
    plt.show()

    errors_fig, errors_ax = plt.subplots()
    plt.title('{}\'s reconstruction errors'.format(name))
    plt.xlabel('Error')
    errors_ax.hist(errors, normed=True, alpha=0.5)
    print(errors)
    kde = stats.gaussian_kde(errors)
    xx = np.linspace(np.min(errors), np.max(errors), 1000)
    errors_ax.plot(xx, kde(xx))
    plt.show()

    conv_fig, conv_ax = plt.subplots()
    plt.title('{}\'s Convergence Speed (number of iteration needed to converge)'.format(name))
    plt.xlabel('Iterations')
    conv_ax.hist(convergence, normed=True, alpha=0.5)
    kde = stats.gaussian_kde(convergence)
    xx = np.linspace(0, np.max(convergence), 1000)
    conv_ax.plot(xx, kde(xx))
    plt.show()

    time_fig, time_ax = plt.subplots()
    plt.title('{}\'s Convergence Time (time needed to converge)'.format(name))
    plt.xlabel('Time (ms)')
    time_ax.hist(convergence_time, density=True, alpha=0.5)
    kde = stats.gaussian_kde(convergence_time)
    xx = np.linspace(0, np.max(convergence_time), 1000)
    time_ax.plot(xx, kde(xx))

    print('Finished NMF with {} repetitions of {} optimization function'.format(repetitions, name))
    print('Average reconstruction errors: {}'.format(np.mean(errors)))
    print('Average convergence speed: {}'.format(np.mean(convergence)))
    print('Average convergence time (ms): {}'.format(np.mean(convergence_time)))
    plt.show()


## Load the news data
A = np.genfromtxt('news_sample.csv', delimiter=',', skip_header=1)
## To read the terms, just read the first line of news_sample.csv
with open('news_sample.csv') as f:
    header = f.readline()
    terms = [x.strip('"\n') for x in header.split(',')]

# Comparing performance of different algorithms
test_optimize_func(A, opt_func=nmf_als, name='NMF ALS')
test_optimize_func(A, opt_func=nmf_lns, name='NMF Lee and Seung')
test_optimize_func(A, opt_func=nmf_opl, name='NMF OPL')

'''
## Task 2
#########

## Normalise the data before applying the NMF algorithms
B = A/sum(sum(A)) # We're assuming Python3 here

## To print the top-10 terms of the first row of H, we can do the following
h = H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))

## USE NMF to analyse the data
## REPEAT the analysis with GKL-optimizing NMF

## Task 3
#########

## In Python, we can compute a slightly different normalized mutual information using scikit-learn's normalized_mutual_info_score (imported as nmi)
def nmi_news(x):
    gd = np.loadtxt('news_ground_truth.txt')
    return 1 - nmi(x, gd)

## We can compute Karhunen-Loeve 'manually'
Z = zscore(A)
U, S, V = svd(Z, full_matrices=False)
V.transpose()
V = V[0:20,:]
KL = np.matmul(Z, V.transpose())

## COMPUTE pLSA with the matrix B from the previous task

# Clustering the KL matrix
clustering = KMeans(n_clusters=20, n_init=20).fit(KL)
idx = clustering.labels_
## How good is this?
print("NMI for KL = {}".format(nmi_news(idx)))
'''
