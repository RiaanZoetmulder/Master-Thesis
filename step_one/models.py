#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
models.py
Creates the generator, the encoder and the entire model class
"""

import tensorflow as tf
from advanced_layers import Z_Layer, RCNNCell, ExtRCNNCell
from basic_layers import Layer
import time
from optimization_updates import create_optimization_updates
from IO import create_batches
import numpy as np

###############################
#######    Generator  #########
###############################

class Generator(object):
    
    
    def __init__(self, args, nclasses, embs):
        self.args = args
        self.nclasses = nclasses
        self.embs = embs
        self.vocab_size, self.embedding_dim = embs.params[0].shape
        print 'Received dictionary of vocab size %s and embedding dim %s.' % \
                    (self.vocab_size, self.embedding_dim)
            
        self.ACTIVATION_DICT = {'tanh':lambda : tf.nn.tanh,
                                 'sigmoid':lambda:  tf.nn.sigmoid}
        
        
    def ready(self):
        
        # inputs for feed dict.
        # x should be a matrix of word Id's, integer valued
        # embedding placeholder is 
        self.x = x = tf.placeholder(tf.int32, [None, self.args.max_len], name='input_placeholder')  # None, 256
        self.embedding_placeholder = embedding_placeholder = tf.placeholder(tf.float32, 
                                                                            [self.vocab_size,
                                                                             self.embedding_dim]) # about 140000, 200
        self.dropout = dropout = tf.placeholder(tf.float32, name='dropout_rate')                                 # scalar dropout 
        self.training = training = tf.placeholder(tf.bool, name = 'training')
        
        with tf.variable_scope("Generator"):
            # create variable for embeddings
            W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]),
                    trainable=False, name="W")
            
            # assign embeddings, doing this should ensure it is not trainable.
            embedding_init = W.assign(embedding_placeholder)
            
            # embedding lookup
            rnn_inputs = self.rnn_inputs =  tf.nn.embedding_lookup(embedding_init, x) # (batch, max_len, embedding_size)
            
            
            # set the padding id
            padding_id = self.padding_id = self.embs.vocab_map["<padding>"]
            
            # params for network
            n_d = self.args.hidden_dimension # hidden size for RCNN layer
            n_e = self.embs.n_d              # Input size for rcnn layer
            
            # get activation
            activation = self.ACTIVATION_DICT[self.args.activation]
            
            # TODO: implement layer type for experiments.. Perhaps not necessary
            
            with tf.name_scope('First_RCNN_Layers'): 
                # layer list
                self.layers = []
                self.zero_states = []
                for i in xrange(2):
                    
                    # create RCNN Cell
                    self.layers.append(
                        RCNNCell(n_d, idx = i)
                    )
                    
                    # Create zero states for cells
                    self.zero_states.append(self.layers[i].zero_state(x.get_shape()[1],
                                                               tf.float32))
                
                # len * batch --> len is simply the number of similarly shaped objects, batch is maximum words
                masks = tf.cast(tf.not_equal(x, padding_id), tf.float32, name = 'masks_generator')
                
                # apply dropout depending on whether training or not
                inputs = tf.cond(training,
                                 lambda: tf.nn.dropout(rnn_inputs, dropout), 
                                 lambda: rnn_inputs, name='dropout_input')
                
                # reverse sentences
                inputs_reversed = tf.cond(training,
                                          lambda: tf.nn.dropout(rnn_inputs[::-1], dropout),
                                          lambda: rnn_inputs[::-1], 
                                          name='dropout_input_reversed')
                with tf.name_scope('dynamic_rnn_encoder'):
                
                    # collect outputs
                    h1, _=  tf.nn.dynamic_rnn(self.layers[0],
                                              inputs,
                                              initial_state= self.zero_states[0], time_major = True)
                    
                    h2, _=  tf.nn.dynamic_rnn(self.layers[1],
                                              inputs_reversed,
                                              initial_state= self.zero_states[1], time_major = True)
                    
                # concatenate outputs
                h_concat = tf.concat(2,[h1, h2[::-1]])
                
                
                # apply dropout to output first layer
                h_final = tf.cond(training,
                                  lambda: tf.nn.dropout(h_concat, dropout), 
                                  lambda: h_concat, 
                                  name='dropout_firstlayer')
                                                           
                                                                                                        
            with tf.name_scope('Zlayer') as ns:
                
                # creating the output layer
                self.output_layer = output_layer = Z_Layer(h_final.get_shape()[2])
                
                # sample a which words should be kept
                zpred = output_layer.sample_all(h_final)
                
                # z itself should not be updated
                zpred = tf.stop_gradient(zpred)
        
                # get the probabilities and log loss
                with tf.name_scope('zlayer_forward_pass'):
                    probs = self.probs= output_layer.forward_all(h_concat, zpred)
                
                with tf.name_scope('sigmoid_cross_entropy'):
                    logpz = self.logpz = - tf.nn.sigmoid_cross_entropy_with_logits(probs, zpred) * masks
                
                logpz = self.logpz = tf.reshape(logpz,tf.shape(x), name = 'reshape_logpz')
                probs = self.probs = tf.reshape(probs, tf.shape(x), name = 'probs_reshape')
                
                # assign z
                z = self.zpred = zpred
            
                # sum z
                
                with tf.name_scope('operations_on_z'):
                    self.zsum = tf.reduce_sum(z, 0, name = 'zsum')
                    self.zdiff = tf.reduce_sum(tf.abs(z[1:]-z[:-1]),  0, name = 'zdiff')
                
            
            # collect number of trainable params
            total_parameters = 0
            for variable in tf.trainable_variables():
                sh = variable.get_shape()
                variable_parametes = 1
                for dim in sh:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print 'total #  Generator parameters:', total_parameters
            
            # get l2 cost for all parameters
            varls = tf.trainable_variables() 
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in varls
                        if 'bias' not in v.name ]) * self.args.l2_reg
            
            self.L2_loss = lossL2


###############################
#######    Encoder    #########
###############################
class Encoder(object):

    def __init__(self, args, embedding_layer, nclasses, generator):
        self.args = args
        self.emb_layer = embedding_layer
        self.nclasses = nclasses
        self.gen = generator

    def ready(self):
        with tf.variable_scope("Encoder"):
            
            gen = self.gen
            emb_layer = self.emb_layer
            args = self.args
            padding_id = emb_layer.vocab_map["<padding>"]
            training = gen.training

            # variables from the generator
            dropout = gen.dropout
            x = gen.x
            # removed z here. can add back if you want to
            z = tf.expand_dims(gen.zpred, 2)

            # input placeholder
            y = self.y = tf.placeholder(tf.float32, [None, self.nclasses], name= 'target_values')

            n_d = args.hidden_dimension
            n_e = emb_layer.n_d

            layers = self.layers = [ ]
            zero_states = self.zero_states = [ ]

            depth = args.depth
            use_all = args.use_all
            layer_type = args.layer.lower()
            
            with tf.name_scope('ExtCells_Encoder'):
                # create layers
                for i in xrange(depth):
    
                    # TODO: Include ExtLSTMCell here
                    layers.append(
                                    ExtRCNNCell(n_d,
                                                idx = 'ExtRCNNCell_%i'%i)
                                 )
                    zero_states.append(
                                        layers[i].zero_state(x.get_shape()[1])
                                      )
    
    
                # TODO: Some stuff missing here!
    
                # create layers
    
                h_prev = gen.rnn_inputs
                lst_states = []
                # print 'z outside shape: ', z.get_shape()
                # print 'zero_state outside shape: ', zero_state.get_shape()
                # print 'embs: ', embs.get_shape()
                layers_enc = []
                for idx, layer in enumerate(layers):
    
                    # a bug might occur here because you are using the same names for hnext t and t+1 
                    layers_enc.append(
                                      tf.nn.dynamic_rnn(layer,
                                              (h_prev, z),
                                              initial_state= zero_states[idx], 
                                              time_major = True)[0]
                                      )
                    print 'layer ' + str(idx)+ ' ', layers_enc[idx].get_shape()
    
                    ############################
                    # TODO: if pooling do stuff#
                    ############################
                    if args.pooling:
                        # do something
                        print 'implement the pooling'
                        raise NotImplemented
    
                    else:
                        lst_states.append(layers_enc[idx][-1])
    
                    # update next state, apply dropout
                    h_prev = tf.cond(training,
                                 lambda: tf.nn.dropout(layers_enc[idx], dropout), 
                                 lambda: layers_enc[idx], name='dropout_h_next')
    
                # select whether to use all of them or not.
                if args.use_all:
                    size = depth * n_d
    
                    # batch * size (i.e. n_d*depth)
                    h_final = tf.concat(1, lst_states)
                else:
                    size = n_d
                    h_final = lst_states[-1]
    
                # apply dropout to final state
                h_final = tf.cond(training,
                                 lambda: tf.nn.dropout(h_final, dropout), 
                                 lambda: h_final, name='dropout_h_next')


            print h_final.get_shape()
            # implement final layer
            
            with tf.name_scope('output_layer'):
                preds = self.preds = Layer(h_final, self.nclasses)
                
                tf.histogram_summary('encoder_output', h_final)

            print 'preds: ', preds.get_shape()
            
            with tf.name_scope('error_functions_encoder'):
                
                
                loss_mat = self.loss_mat = (preds-y)**2 # batch
    
                # difference in predicitons
                pred_diff = self.pred_diff = tf.reduce_mean(tf.reduce_max(preds, 1) - tf.reduce_min(preds, 1))
    
                # get the loss for each class
                if args.aspect < 0:
                    loss_vec = tf.reduce_mean(loss_mat, 1)
                else:
                    assert args.aspect < self.nclasses
    
                    loss_vec = loss_mat[:,args.aspect]
    
                self.loss_vec = loss_vec
    
                # get values from the generator
                zsum = gen.zsum
                zdiff = gen.zdiff
                logpz = gen.logpz
    
    
                coherent_factor = args.sparsity * args.coherent
                # total loss
                loss = self.loss = tf.reduce_mean(loss_vec)
    
                # calculate the sparsity cost
                sparsity_cost = self.sparsity_cost = tf.reduce_mean(zsum) * args.sparsity + \
                                                     tf.reduce_mean(zdiff) * coherent_factor
    
                # loss function as mentioned in the paper
                cost_vec = loss_vec + zsum * args.sparsity + zdiff * coherent_factor
    
                cost_logpz = tf.reduce_mean(cost_vec * tf.reduce_sum(logpz, 0))
                self.obj = tf.reduce_mean(cost_vec)
    
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
                
                total_parameters = 0
                for variable in variables:
                    sh = variable.get_shape()
                    variable_parametes = 1
                    for dim in sh:
                        variable_parametes *= dim.value
                    total_parameters += variable_parametes
                print 'total # Encoder parameters:', total_parameters
                
                
                # theano code
                
                lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables
                            if 'bias' not in v.name ]) * self.args.l2_reg
                
                # generator and encoder loss
                self.cost_g = cost_logpz * 10 + gen.L2_loss
                self.cost_e = loss * 10 + lossL2
                
                print 'initialized!'
# TODO: Finish the model

###############################
#######     Model     #########
###############################
class Model(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses


    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
        self.generator = Generator(args, nclasses, embedding_layer)
        self.encoder = Encoder(args, embedding_layer, nclasses, self.generator)
        
        self.generator.ready()
        self.encoder.ready()
        
        self.x = self.generator.x
        self.y = self.encoder.y
        self.z = self.generator.zpred
        
    
    def train(self, train, dev, test, rationale_data, sess):
        '''
        Function to do training procedure, use args to set which optimizer
        and what parameters.
        train:
            data at index 0 targets at index 1
        dev:
            development data
        test:
            test data
        rationale_data:
            data with rationales
        '''
        
        args = self.args
        dropout = args.dropout
        padding_id= self.generator.padding_id
        
        if dev is not None:
            # TODO: Implement development set
            raise NotImplemented 
            
        if test is not None:
            # TODO: Implement testing later
            raise NotImplemented 
            
        if rationale_data is not None:
            # TODO: implement rationale data
            raise NotImplemented 
            
        start_time = time.time()
        
        train_batches_x, train_batches_y = create_batches(
                                train[0], train[1], args.batch, padding_id
                            )
        
        print 'Time to create batches: %f.2' % (time.time()-start_time)
        
        #args.learning = 'sgd'
        train_step_enc = create_optimization_updates(self.encoder.cost_e,
                                                    method= args.learning,
                                                    beta1 = args.beta1,
                                                    beta2 = args.beta2,
                                                    lr = args.learning_rate)
        
        train_step_gen = create_optimization_updates(self.encoder.cost_g,
                                                    method= args.learning,
                                                    beta1 = args.beta1,
                                                    beta2 = args.beta2,
                                                    lr = args.learning_rate)
        
        
        merged = tf.merge_all_summaries()
        
        init = tf.initialize_all_variables()
        
        train_writer = tf.train.SummaryWriter( 'train', sess.graph)
        
        sess.run(init)
        
        
        # TODO: initialize training loop here...
        eval_period = args.eval_period
        unchanged = 0
        best_dev = 1e+2
        best_dev_e = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.10 + 1e-3
        for epoch in xrange(args.max_epochs):
            
            unchanged += 1
            if unchanged > 20: return
            
            # Create new batches
            train_batches_x, train_batches_y = create_batches(
                            train[0], train[1], args.batch, padding_id
                        )

            more = True
            
            # TODO: implement learning rate decay
            
            
            while more:
                
                processed = 0
                train_cost = 0.0
                train_loss = 0.0
                train_sparsity_cost = 0.0
                p1 = 0.0
                start_time = time.time() # record the time for starting the loop
                
                N = len(train_batches_x)
                for i in xrange(N):
                    
                    # notify user for elapsed time
                    if (i+1)%10 ==0:
                        print "\r{}/{} {:.2f}       ".format(i+1,N,p1/(i+1))
                        print 'cost: ', cost
                        print 'loss: ', loss
                        print 'sparsity cost: ', sparsity_cost
                    
                    # training batches for this round
                    bx, by = train_batches_x[i], train_batches_y[i]
                    
                    mask = (bx != padding_id)
                    
                    
                    # TODO: figure out why i get a shape mismatch!
                    if bx.shape[1] != 256:
                        print 'shape of x: ', bx.shape
                        continue 
                    
                    
                    feed_dict = {self.x: bx,
                                 self.y : by, 
                                 self.generator.embedding_placeholder: self.embedding_layer.params[0], 
                                 self.generator.dropout: args.dropout, 
                                 self.generator.training: True}

                    _,_, cost, loss, sparsity_cost, bz, summary  = sess.run([train_step_enc,train_step_gen,
                                                self.encoder.obj,
                                                self.encoder.loss,
                                                self.encoder.sparsity_cost,
                                                self.z, merged], 
                                                feed_dict)
                    
                    train_writer.add_summary(summary, i)
                    
                    
                    
                    k = len(by)
                    processed += k
                    train_cost += cost
                    train_loss += loss
                    train_sparsity_cost += sparsity_cost
                    p1 += np.sum(bz*mask) / (np.sum(mask)+1e-8)
                    
                curr_train_avg_cost = train_cost/N
                
                
                print 'train Average Cost:', curr_train_avg_cost
                more = False
                    
                
        #TODO: contemplate whether you want to get the norms of the matrices
        
        
        
