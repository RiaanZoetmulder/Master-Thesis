#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:48:29 2017

@author: riaan
"""

import tensorflow as tf
import numpy as np
np.random.seed(0)
gen = np.random.RandomState(0)

if 'sess' in globals() and sess:
    sess.close()
    tf.reset_default_graph()
    
with tf.Session() as sess:
    #tf.set_random_seed(0)
    test = tf.get_variable('checktf', [10], 
                         initializer = tf.random_uniform_initializer(-0.1, 0.1, seed = 0))

    init = tf.initialize_all_variables()

            
            
    sess.run(init)
    
    feed_dict = {}
    
    test  = sess.run(test, feed_dict = feed_dict)
    
    print test

print '\n'
print np.random.uniform(-0.1, 0.1, 10)
print '\n'
print gen.uniform(-0.1, 0.1, 10)
    
    