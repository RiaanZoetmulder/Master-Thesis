#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
File where you can select which optimization procedure to use

Author Riaan
"""

import tensorflow as tf


def create_optimization_updates(cost, params = None, method = 'sgd',
                                max_norm = 5,
                                updates = None, gradients = None,
                                lr = 0.01, eps = None, rho = 0.99,
                                gamma = 0.999, beta1 = 0.9,
                                beta2= 0.999, momentum = 0.0):
    
    if method == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)
        gnorms = None
    elif method == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(learning_rate = lr, rho = rho).mimimze(cost)
        gnorms = None
    elif method == 'adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate = lr).minimize(cost)
        gnorms = None
    elif method =='adam':
        opt_uncl = tf.train.AdamOptimizer(learning_rate = lr,
                                         beta1 = beta1,
                                         beta2= beta2).minimize(cost)
        
        opt = opt_uncl
        gnorms = None

        
    
    return opt, gnorms