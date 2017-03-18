#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
rationale_dependent.py

"""
from options import load_arguments
from IO import create_embedding_layer, read_annotations, create_batches, read_rationales
import tensorflow as tf
import numpy as np

from models import Model
def main():
    print 'Parser Arguments' 
    for key, value in args.__dict__.iteritems():
        print u'{0}: {1}'.format(key, value)
        
    # ensure embeddings exist
    assert args.embedding, "Pre-trained word embeddings required."
    
    embed_layer = create_embedding_layer(
                                         args.embedding
                                         )
    
    max_len = args.max_len
    
    if args.train:
        train_x, train_y = read_annotations(args.train)
        train_x = [ embed_layer.map_to_ids(x)[:max_len] for x in train_x ]
                   
    if args.dev:      
        dev_x, dev_y = read_annotations(args.dev)
        dev_x = [ embed_layer.map_to_ids(x)[:max_len] for x in dev_x ]
    
    if args.load_rationale:
        rationale_data = read_rationales(args.load_rationale)
        for x in rationale_data:
            x["xids"] = embed_layer.map_to_ids(x["x"])
                   
    # TODO: create development and test sets and rationale stuff
            

    if args.train:
        with tf.Graph().as_default() as g:
            
            # used to be set to 2345
            tf.set_random_seed(2345)
            np.random.seed(2345)
            with tf.Session() as sess:
                # initialize Model
                #TODO: create encoder class in model
                
                model = Model(
                            args = args,
                            embedding_layer = embed_layer,
                            nclasses = len(train_y[0])
                        )
                model.ready()
                
                
                # added this for testing
                # TODO: Remove later
                model.train((train_x, train_y),
                            (dev_x, dev_y) if args.dev else None,
                            None,
                            rationale_data if args.load_rationale else None,
                            sess) 
                
                
            
            '''
            train_batches_x, train_batches_y = create_batches(
                                train_x, train_y, args.batch, model.generator.padding_id
                            )
            
            feed_dict={model.generator.x: train_batches_x[0],model.encoder.y : train_batches_y[0], model.generator.embedding_placeholder: embed_layer.params[0], 
                      model.generator.dropout: 0.9, model.generator.training: True}
                      
            
            init = tf.initialize_all_variables()
            
            
            sess.run(init)
            print 'past graph initialization'
            
            
            probs, logpz, ztotsum, weights = sess.run([model.generator.probs, model.generator.logpz, model.generator.ztotsum, model.generator.weights], feed_dict)
            
            print 'probs: ', probs
            print 'logpz: ', logpz
            print 'ztotsum: ', ztotsum
            print 'weights: ', weights
            
            '''
        
    

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
    
if __name__ == '__main__':
    args = load_arguments()
    
    # reset zie graph
    reset_graph()
    
    # start procedures
    main()

