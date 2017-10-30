#encoding: utf-8

import numpy as np
import random
#import pylab
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt.base import matrix
import math

class SVM:
    def __init__(self, kernel, slack):
        self.kernel = kernel
        self.slack = slack 
        self.supportVectors = []

    def train(self, data):
        n = len(data)
        
        # Generate P matrix
        P = []
        for i in range(n):
            P.append([])
            xi, yi, ti = data[i]
            xiVec = (xi, yi)
            for j in range(n):
                xj, yj, tj = data[j]
                xjVec = (xj, yj)
                P[i].append(ti * tj * self.kernel(xiVec, xjVec))

        self.P = P

        # Build q vector
        q = np.ones((n,1)) * -1

        # Build h vector
        h = np.ones((n,1)) * self.slack

        # Build G matrix
        G = np.eye(n) * -1

        # Minimize

        r = qp(matrix(P), matrix(q), matrix(G), matrix(h))

        if r['status'] != 'unknown':
            # Get alpha values
            alpha = list(r['x'])

            for i in range(len(alpha)):
                if alpha[i] > 1e-5:
                    x, y, t = data[i]
                    self.supportVectors.append((x, y, t, alpha[i]))
                    

    def indicator(self, xStar):
        sum = 0
        
        for a in self.supportVectors:
            x, y, t, alpha = a
            sum += alpha * t * self.kernel(xStar, (x,y))

        return sum
        #return 1 * (sum > 0) - 1 * (sum < 0)

    
    def plotMargin(self):

        xrange = np.arange(-4, 4, 0.05)
        yrange = np.arange(-4, 4, 0.05)

        grid = matrix([[self.indicator((x, y))
                        for y in yrange]
                       for x in xrange])
        
        CS = plt.contour(xrange, yrange,-grid,
                      levels = np.linspace(-4, 4, len(xrange)*4),
                      ls = '-',
                      cmap = plt.cm.seismic,
                      origin='lower')

        plt.contour(CS,
                      levels = np.linspace(-4, 4, len(xrange)*4),
                      ls = '-',
                      cmap = plt.cm.seismic,
                      origin='lower')

        plt.contour(xrange, yrange, grid,
                      (-1.0, 0.0, 1.0),
                      colors = ('red', 'black', 'blue'),
                      linewidths=(1, 3, 1))
        


def polKernel(x, y, p):
    return (np.dot(x,y) + 1)**p

def sigmoidKernel(x, y, k , delta):
    return math.tanh(k * np.dot(x, y) - delta)

def radialKernel(x, y, sigma):
    diff = np.array(x) - np.array(y)
    return math.exp(-np.dot(diff, diff) / (2*sigma**2))




