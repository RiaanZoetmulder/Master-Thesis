# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:48:53 2017

@author: riaan
"""

import tensorflow as tf

def Layer(x, n_classes,  hasbias = True, scope = None, act = tf.nn.sigmoid):
    
    with tf.variable_scope(scope or 'output_layer') as scope:
        W = tf.get_variable('W_out', 
                            [x.get_shape()[1], n_classes],
                            initializer = tf.random_uniform_initializer(-0.05, 0.05),
                            dtype = tf.float32)
        
        tf.histogram_summary('output_enc_weights', W)
        
        temp = tf.matmul(x, W)
        if hasbias:
            
            B = tf.get_variable('B_out', 
                                [1, n_classes],
                                initializer = tf.constant_initializer(0.0),
                                dtype = tf.float32)
            
            tf.histogram_summary('output_enc_bias', B)
            
            temp += B

        logits = act(temp)
        
        
    
    return logits