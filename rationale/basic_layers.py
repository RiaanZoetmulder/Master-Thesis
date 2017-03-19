# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:48:53 2017

@author: riaan
"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell 
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
tf.set_random_seed(2345)


_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"

def Layer(x, n_classes,  hasbias = True, scope = None, act = tf.nn.sigmoid,
          initializer =  tf.random_uniform_initializer(-0.05, 0.05, seed = 2345)):
    
    with tf.variable_scope(scope or 'output_layer') as scope:
        W = tf.get_variable('W_out', 
                            [x.get_shape()[1], n_classes],
                            initializer = initializer,
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
    
    

class BasicRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = self._activation(_linear([inputs, state], self._num_units, True))
    return output, output
    

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype, 
        initializer=  tf.random_uniform_initializer(-0.05, 0.05, seed = 2345))
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term