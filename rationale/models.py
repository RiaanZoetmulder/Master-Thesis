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
import json
import matplotlib 
import matplotlib.pyplot as plt

###############################
#######    Generator  #########
###############################
tf.set_random_seed(2345)

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
        args = self.args
        
        # inputs for feed dict.
        # x should be a matrix of word Id's, integer valued self.args.max_len
        # embedding placeholder is 
        self.x = x = tf.placeholder(tf.int64, [None, self.args.max_len], 
                                    name='input_placeholder')  
        self.embedding_placeholder = embedding_placeholder = tf.placeholder(tf.float32, 
                                                                           [self.vocab_size,
                                                                             self.embedding_dim]) # about 140000, 200
        self.dropout = dropout = tf.placeholder(tf.float32, name='dropout_rate')                                 # scalar dropout 
        self.training = training = tf.placeholder(tf.bool, name = 'training')
        self.lr = tf.placeholder(tf.float32, name = 'lr')
        
        with tf.variable_scope("Generator"):
            
            # create variable for embeddings
            W = tf.Variable(tf.constant(0.0, 
                            shape=[self.vocab_size, self.embedding_dim],
                            dtype = tf.float32),
                            trainable=False, name="W")
            
            # assign embeddings, doing this should ensure it is not trainable.
            embedding_init = W.assign(embedding_placeholder)
            
            # embedding lookup
            rnn_inputs = self.rnn_inputs =  tf.nn.embedding_lookup(embedding_init,
                                                                   x) 
            # set the padding id
            padding_id = self.padding_id = self.embs.vocab_map["<padding>"]
            
            # dimensions RCNN
            n_d = self.args.hidden_dimension 
            n_e = self.embs.n_d              
            
            # get activation
            activation = self.ACTIVATION_DICT[self.args.activation]
            
            with tf.name_scope('First_RCNN_Layers'): 
                
                # layer list
                self.layers = []
                self.zero_states = []
                for i in xrange(2):
                    
                    # create RCNN Cell
                    self.layers.append(
                        RCNNCell(n_d,
                                 idx = i,
                                 initializer = tf.contrib.layers.xavier_initializer()
                                 )
                    )
                    
                    # Create zero states for cells
                    self.zero_states.append(self.layers[i].zero_state(x.get_shape()[1],
                                                               tf.float32))
                
                # masks for removing padding
                masks = tf.cast(tf.not_equal(x, padding_id),
                                tf.float32,
                                name = 'masks_generator')
                
                self.masks = masks
                
                # dropout
                inputs = tf.nn.dropout(rnn_inputs, dropout)
                
                # reverse sentences
                inputs_reversed = inputs[::-1]
                
                
                with tf.name_scope('forward_pass_first_layers_generator'):

                    h1tp = tf.scan(self.layers[0], inputs,
                                   initializer = self.zero_states[0])
                    
                    h2tp = tf.scan(self.layers[1], inputs_reversed,
                                   initializer = self.zero_states[1])
                    
                    if len(h1tp.get_shape())>1:
                        h1 = h1tp[:,:, n_d * args.order:]
                        h2 = h2tp[:,:, n_d * args.order:]
                    else:
                        h1 = h1tp[:, n_d * args.order:]
                        h2 = h2tp[:, n_d * args.order:]
                        
                    
                # concatenate outputs
                h_concat = tf.concat(2,[h1, h2[::-1]])
                
                self.avg = h_concat[0, :20, 0]
                
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
                
                
                self.zpredsum = tf.reduce_sum(zpred)
                
                # z itself should not be updated
                zpred = tf.stop_gradient(zpred)
        
                # get the probabilities and log loss
                with tf.name_scope('zlayer_forward_pass'):
                    probs, logits = output_layer.forward_all(h_concat, zpred)
                
                with tf.name_scope('sigmoid_cross_entropy'):
                    
                    # this error function is the binary cross entropy rewritten
                    # in terms of softplus, should be more numerically stable
                    
                    logpz = (-logits*(1-zpred)-tf.nn.softplus(-logits)) * masks
                    
                logpz = self.logpz = tf.reshape(logpz,tf.shape(x),
                                                name = 'reshape_logpz')
                probs = self.probs = tf.reshape(probs, tf.shape(x),
                                                name = 'probs_reshape')
                
                # assign z
                z = self.zpred = zpred
                self.ztotsum = tf.reduce_sum(zpred)
            
                # sum z
                with tf.name_scope('operations_on_z'):
                    self.zsum = tf.reduce_sum(z, 0, name = 'zsum')
                    self.zdiff = tf.reduce_sum(tf.abs(z[1:]-z[:-1]),  0,
                                               name = 'zdiff')
                
            
            # collect number of trainable params
            total_parameters = 0
            for variable in tf.trainable_variables():
                sh = variable.get_shape()
                variable_parametes = 1
                for dim in sh:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            
            print 'total #  Generator parameters:', total_parameters
            
            string= 'Generator/RCNN_Feed_Forward_Layer_0_0/weights0:0'
            self.weights = [v for v in tf.trainable_variables() if v.name == string][0]
            
            
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
            y = self.y = tf.placeholder(tf.float32, [None, self.nclasses],
                                        name= 'target_values')

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
    
                    # Layers
                    layers.append(
                                    ExtRCNNCell(n_d,
                                                idx = 'ExtRCNNCell_%i'%i, 
                                                initializer = tf.contrib.layers.xavier_initializer())
                                 )
                    zero_states.append(
                                        layers[i].zero_state(x.get_shape()[1])
                                      )
    
    
                # create layers
                h_prev = gen.rnn_inputs
                lst_states = []
                layers_enc = []
                for idx, layer in enumerate(layers):
                    
                    h_temp = tf.scan(layer, (h_prev, z), initializer = zero_states[idx])
                    
                    if len(h_temp.get_shape())>1:
                        layers_enc.append(h_temp[:,:,layer._order*layer._num_units:])
                    else:
                        layers_enc.append(h_temp[:,layer._order*layer._num_units:])
    
                        
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
                if use_all:
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

                
            # output layer encoder
            with tf.name_scope('output_layer'):
                preds = self.preds = Layer(h_final, self.nclasses,
                                           initializer = tf.contrib.layers.xavier_initializer())
                
            
            with tf.name_scope('error_functions_encoder'):
                
                loss_mat = self.loss_mat = (preds-y)**2 
                
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
                self.zsum = zsum = gen.zsum
                self.zdiff = zdiff = gen.zdiff
                self.logpz = logpz = gen.logpz
    
                coherent_factor = args.sparsity * args.coherent
                loss = self.loss = tf.reduce_mean(loss_vec)
    
                # calculate the sparsity cost
                sparsity_cost = self.sparsity_cost = tf.reduce_mean(zsum) * args.sparsity + \
                                                     tf.reduce_mean(zdiff) * coherent_factor
    
                # loss function as mentioned in the paper
                cost_vec = loss_vec + zsum * args.sparsity + zdiff * coherent_factor
                self.cost_logpz = cost_logpz = tf.reduce_mean(cost_vec * tf.reduce_sum(logpz, 0))
                self.obj = tf.reduce_mean(cost_vec)
    
                
                # count the parameters in the encoder
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
                total_parameters = 0
                for variable in variables:
                    sh = variable.get_shape()
                    variable_parametes = 1
                    for dim in sh:
                        variable_parametes *= dim.value
                    total_parameters += variable_parametes
                print 'total # Encoder parameters:', total_parameters
                
                

                # regularization                
                lossL2 =self.loss_l2_e= tf.add_n([ tf.nn.l2_loss(v) for v in variables
                            if 'bias' not in v.name ]) * self.args.l2_reg
                self.loss_l2_g = gen.L2_loss
                # generator and encoder loss
                
                with tf.name_scope('total_cost_functions'):
                    self.cost_g = cost_logpz * 10 + gen.L2_loss
                    self.cost_e = loss * 10 + lossL2
                    self.tlg = tf.scalar_summary('total_cost_generator', self.cost_g)
                    self.tle = tf.scalar_summary('total_cost_encoder', self.cost_e)
                
                
                    # scalar summaries
                
                    self.l2g = tf.scalar_summary('L2_loss_generator', gen.L2_loss)
                
                    self.l2e = tf.scalar_summary('L2_loss_encoder', lossL2)
                    self.lg = tf.scalar_summary('cost_logpz_x10', cost_logpz * 10)
                    self.le = tf.scalar_summary('loss_encoder', loss * 10)
                    self.obj_loss = tf.scalar_summary('Objective', self.obj)
                
                
                print 'Fully Initialized!'

###############################
#######     Model     #########
###############################
class Model(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        
        self.obj_array = []
        self.prec_array = []


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
            
            dev_batches_x, dev_batches_y = create_batches(
                            dev[0], dev[1], args.batch, padding_id
                        )
            
        if test is not None:
            
            test_batches_x, test_batches_y = create_batches(
                            test[0], test[1], args.batch, padding_id
                        ) 
            
        if rationale_data is not None:
            
            valid_batches_x, valid_batches_y = create_batches(
                    [ u["xids"] for u in rationale_data ],
                    [ u["y"] for u in rationale_data ],
                    args.batch,
                    padding_id,
                    sort = False
                )

            
        
        start_time = time.time()
        
        train_batches_x, train_batches_y = create_batches(
                                train[0], train[1], args.batch, padding_id
                            )
        
        print 'Time to create batches: %f.2' % (time.time()-start_time)
        
        train_step_enc, enorm = create_optimization_updates(self.encoder.cost_e,
                                                    method= args.learning,
                                                    beta1 = args.beta1,
                                                    beta2 = args.beta2,
                                                    lr = self.generator.lr)
        
        train_step_gen, gnorm = create_optimization_updates(self.encoder.cost_g,
                                                    method= args.learning,
                                                    beta1 = args.beta1,
                                                    beta2 = args.beta2,
                                                    lr = self.generator.lr)
        le, lg, tle, tlg, l2e, l2g, obj_loss = self.encoder.le, self.encoder.lg, \
                                        self.encoder.tle, self.encoder.tlg, \
                                        self.encoder.l2e, self.encoder.l2g, \
                                        self.encoder.obj_loss
        
        self.merged = merged = tf.merge_summary([le, lg, tle, tlg, l2e, l2g, obj_loss])
        
        init = tf.initialize_all_variables()
        
        train_writer = tf.train.SummaryWriter( 'train', sess.graph)
        eval_writer = tf.train.SummaryWriter( 'eval', sess.graph)
        
        saver = tf.train.Saver()
        
        sess.run(init)
        
        
        # Training Loop
        unchanged = 0
        best_dev = 1e+2
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
            
            
            while more:
                
                processed = 0
                train_cost = 0.0
                train_loss = 0.0
                train_sparsity_cost = 0.0
                p1 = 0.0
                start_time = time.time()
                
                N = len(train_batches_x)
                for i in xrange(N):
                    
                    # notify user for elapsed time
                    if (i+1)%50 == 0:
                        print "\r{}/{} {:.5f}       ".format(i+1,N,p1/(i+1))
                        
                    # training batches for this round
                    bx, by = train_batches_x[i], train_batches_y[i]
                    
                    mask = (bx != padding_id)
                    
                    if bx.shape[1] != 256:
                        bx = bx.T
                        continue
                    
    
                    
                    feed_dict = {self.x: bx,
                                 self.y : by, 
                                 self.generator.embedding_placeholder: self.embedding_layer.params[0], 
                                 self.generator.dropout:1.0 -  args.dropout, 
                                 self.generator.training: True,
                                 self.generator.lr: self.args.learning_rate}
                                 
                    # training forward pass
                    _,_,  cost, loss, sparsity_cost, bz, summary, ztotsum  = sess.run([train_step_enc, train_step_gen,
                                                self.encoder.obj,
                                                self.encoder.loss,
                                                self.encoder.sparsity_cost,
                                                self.z, merged, self.generator.ztotsum], 
                                                feed_dict)
                    
                    
                    k = len(by)
                    processed += k
                    train_cost += cost
                    train_loss += loss
                    train_sparsity_cost += sparsity_cost
                    temp = np.sum(bz*mask) / (np.sum(mask)+1e-8)
                    p1 += temp
                    
                    if temp ==0 or p1 ==0:
                        print 'p1 zero in training loop'
                    
                    
                    
                    
                train_writer.add_summary(summary, epoch)
                    
             
                cur_train_avg_cost = train_cost/N
                
                
                more = False
                if dev:
                    # development set stuff
                    dev_obj, dev_loss, dev_diff, dev_p1 = self.evaluate_data(
                            dev_batches_x, dev_batches_y, sess, epoch, merged,
                            eval_writer)
                    
                    cur_dev_avg_cost = dev_obj
                    
                if args.decay_lr and last_train_avg_cost is not None:
                    
                    # report on training cost and development cost
                    if cur_train_avg_cost > last_train_avg_cost*(1+tolerance):
                        more = True
                        print "\nTrain cost {} --> {}\n".format(
                                last_train_avg_cost, cur_train_avg_cost
                            )
                    if dev and cur_dev_avg_cost > last_dev_avg_cost*(1+tolerance):
                        more = True
                        print "\nDev cost {} --> {}\n".format(
                                last_dev_avg_cost, cur_dev_avg_cost
                            )
                if more:
                    self.args.learning_rate = self.args.learning_rate*0.5
                    print ("Decrease learning rate to {}\n".format(float(self.args.learning_rate)))
                    
                    continue
                    
                last_train_avg_cost = cur_train_avg_cost
                if dev: last_dev_avg_cost = cur_dev_avg_cost
                
                print '\n'
                print  ("Generator Epoch {:.2f}  costg={:.4f}  scost={:.4f}  lossg={:.4f}  " +
                    "p[1]={:.2f} \t[{:.2f}m / {:.2f}m]\n").format(
                        epoch+(i+1.0)/N,
                        train_cost / N,
                        train_sparsity_cost / N,
                        train_loss / N,
                        p1 / N,
                        (time.time()-start_time)/60.0,
                        (time.time()-start_time)/60.0/(i+1)*N
                    )

                self.obj_array.append(train_loss / N)
                    
                if dev:
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.dump and rationale_data:
                            
                            # implement dump rationales
                            self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
                                        sess)

                        if args.save_model:
                            
                            print 'Saving Model: dev_obj @ {:.4f}'.format(dev_obj)
                            saver.save(sess, args.save_model.format(dev_obj))

                    print ("\tsampling devg={:.4f}  mseg={:.4f}  avg_diffg={:.4f}" +
                                "  p[1]g={:.2f}  best_dev={:.4f}\n").format(
                        dev_obj,
                        dev_loss,
                        dev_diff,
                        dev_p1,
                        best_dev
                    )

                    if rationale_data is not None:
                        r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
                                rationale_data, valid_batches_x,
                                valid_batches_y, sess)
                        
                        print ("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
                                    "  prec2={:.4f}\n").format(
                                r_mse,
                                r_p1,
                                r_prec1,
                                r_prec2
                        )
                        self.prec_array.append((r_prec1, r_prec2))
        self.plot()
        
    def plot(self):
        
        epochs = range(len(self.obj_array))
        plt.figure(1)
        
        # loss of objective function
        plt.subplot(211)
        plt.plot(epochs, self.obj_array)
        plt.yscale('loss')
        plt.xscale('epochs')
        plt.title('loss objective')
        plt.grid(True)
        
        # precision
        temp = zip(*self.prec_array)
        prec1, prec2 = list(temp[0]), list(temp[0])
        plt.subplot(212)
        plt.plot(epochs, prec1, color = 'g')
        plt.plot(epochs, prec2, color = 'r')
        plt.yscale('precision')
        plt.xscale('epochs')
        plt.title('precision')
        plt.grid(True)
        
        plt.savefig('precision+loss.png')
        
    

                    
    def evaluate_data(self, batches_x, batches_y, sess, epoch, merged, eval_writer):
            
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        tot_obj, tot_mse, tot_diff, p1 = 0.0, 0.0, 0.0, 0.0
        for bx, by in zip(batches_x, batches_y):
            
            if bx.shape[1] != 256:
                print 'shape of eval x: ', bx.shape
                continue 
            
            feed_dict = {
                         self.x: bx,
                         self.y: by, 
                         self.generator.embedding_placeholder: self.embedding_layer.params[0], 
                         self.generator.dropout:1.0, 
                         self.generator.training:False, 
                         self.generator.lr:self.args.learning_rate
                         }
               
            mask = (bx != padding_id)
            
            
            bz, o, e, d, summary = sess.run([self.z, self.encoder.obj,
                                    self.encoder.loss, self.encoder.pred_diff,
                                    merged],
                                    feed_dict = feed_dict)
            
            p1 += np.sum(bz*mask) / (np.sum(mask) + 1e-8)
            tot_obj += o
            tot_mse += e
            tot_diff += d
            
        eval_writer.add_summary(summary, epoch)

        
        n = len(batches_x)
        return tot_obj/n, tot_mse/n, tot_diff/n, p1/n
        
    
    def dump_rationales(self, path, batches_x, batches_y, sess):
        
        # get sampling function and eval function
        embedding_layer = self.embedding_layer

        lst = [ ]
        for bx, by in zip(batches_x, batches_y):
            if bx.shape[1] != 256:
                print 'dev shape of x: ', bx.shape
                continue 
            
            feed_dict = {
                         self.x: bx,
                         self.y : by, 
                         self.generator.embedding_placeholder: self.embedding_layer.params[0], 
                         self.generator.dropout:1.0,
                         self.generator.training: False,
                         self.generator.lr: self.args.learning_rate
                         }
                
            loss_vec_r, preds_r, bz = sess.run([ self.encoder.loss_vec, 
                                                self.encoder.preds,
                                                self.z ], 
                                                feed_dict = feed_dict)
            
            
            assert len(loss_vec_r) == bx.shape[1]
            
            for loss_r, p_r, x,y,z in zip(loss_vec_r, preds_r, bx.T, by, bz.T):
                loss_r = float(loss_r)
                p_r, x, y, z = p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
                w = embedding_layer.map_to_words(x)
                r = [ u if v == 1 else "__" for u,v in zip(w,z) ]
                diff = max(y)-min(y)
                lst.append((diff, loss_r, r, w, x, y, z, p_r))

        #lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
        with open(path,"w") as fout:
            for diff, loss_r, r, w, x, y, z, p_r in lst:
                fout.write( json.dumps( { "diff": diff,
                                          "loss_r": loss_r,
                                          "rationale": " ".join(r),
                                          "text": " ".join(w),
                                          "x": x,
                                          "z": z,
                                          "y": y,
                                          "p_r": p_r } ) + "\n" )
                
                
    def evaluate_rationale(self, reviews, batches_x, batches_y,sess):
        args = self.args
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        aspect = str(args.aspect)
        
        p1, tot_mse, tot_prec1, tot_prec2 = 0.0, 0.0, 0.0, 0.0
        tot_z, tot_n = 1e-10, 1e-10
        cnt = 0
        for bx, by in zip(batches_x, batches_y):
            mask = bx != padding_id
            
            if bx.shape[1] != 256:
                print '\tSkipping tensor, size mismatch: ', bx.shape
                continue 
            
            feed_dict = {
                         self.x: bx,
                         self.y : by, 
                         self.generator.embedding_placeholder: self.embedding_layer.params[0], 
                         self.generator.dropout:1.0, 
                         self.generator.training: False, 
                         self.generator.lr: self.args.learning_rate
                         }
                         
                         
            
            bz, o, e, d = sess.run([self.z, self.encoder.obj,
                                    self.encoder.loss, self.encoder.pred_diff],
                                    feed_dict = feed_dict)
            
           
            tot_mse += e
            p1 += np.sum(bz*mask)/(np.sum(mask) + 1e-8)
            

            if args.aspect >= 0:
                
                for z,m in zip(bz.T, mask.T):
                    z = [ vz for vz,vm in zip(z,m) if vm ]
                    assert len(z) == len(reviews[cnt]["xids"])
                    truez_intvals = reviews[cnt][aspect]
                    prec = sum( 1 for i, zi in enumerate(z) if zi>0 and \
                                any(i>=u[0] and i<u[1] for u in truez_intvals) )
                    nz = sum(z)
                    if nz > 0:
                        tot_prec1 += prec/(nz+0.0)
                        tot_n += 1
                    tot_prec2 += prec
                    tot_z += nz
                    cnt += 1

        n = len(batches_x)
        return tot_mse/n, p1/n, tot_prec1/tot_n, tot_prec2/tot_z
        
