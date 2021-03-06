#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
options.py

Source: Tao Lei's code.
"""

import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--load_rationale",
            type = str,
            default = "data/annotations.json",
            help = "path to annotated rationale data"
        )
    argparser.add_argument("--embedding",
            type = str,
            default = "data/review+wiki.filtered.200.txt.gz",
            help = "path to pre-trained word vectors"
        )
    argparser.add_argument("--save_model",
            type = str,
            default = "data/saves/best_model_{:.4f}.ckpt",
            help = "path to save model parameters"
        )
    argparser.add_argument("--load_model",
            type = str,
            default = "",
            help = "path to load model"
        )
    argparser.add_argument("--train",
            type = str,
            default = "data/reviews.aspect1.train.txt.gz",
            help = "path to training data"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "data/reviews.aspect1.heldout.txt.gz",
            help = "path to development data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--dump",
            type = str,
            default = "outputs.json",
            help = "path to dump rationale"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = -1,
            help = "evaluate model every k examples"
        )
    argparser.add_argument("--batch",
            type = int,
            default = 256,
            help = "mini-batch size"
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adam",
            help = "learning method"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.0005,# was set to 0.0005
            help = "learning rate"
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.1,
            help = "dropout probability"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 1e-6,
            help = "L2 regularization weight"
        )
    argparser.add_argument("-act", "--activation",
            type = str,
            default = "tanh",
            help = "type of activation function"
        )
    argparser.add_argument("-d", "--hidden_dimension",
            type = int,
            default = 200,
            help = "hidden dimension"
        )
    argparser.add_argument("-d2", "--hidden_dimension2",
            type = int,
            default = 30,
            help = "hidden dimension"
        )
    argparser.add_argument("--layer",
            type = str,
            default = "rcnn",
            help = "type of recurrent layer"
        )
    argparser.add_argument("--depth",
            type = int,
            default = 2,
            help = "number of layers"
        )
    argparser.add_argument("--pooling",
            type = int,
            default = 0,
            help = "whether to use mean pooling or the last state"
        )
    argparser.add_argument("--order",
            type = int,
            default = 2,
            help = "feature filter width"
        )
    argparser.add_argument("--use_all",
            type = int,
            default = 1,
            help = "whether to use the states of all layers"
        )
    argparser.add_argument("--max_len",
            type = int,
            default = 256,
            help = "max number of words in input"
        )
    argparser.add_argument("--sparsity",
            type = float,
            default = 0.0003 # was set to 0.0003
        )
    argparser.add_argument("--coherent",
            type = float,
            default = 2.0
        )
    argparser.add_argument("--aspect",
            type = int,
            default = 1
        )
    argparser.add_argument("--beta1",
            type = float,
            default = 0.9
        )
    argparser.add_argument("--beta2",
            type = float,
            default = 0.999
        )
    argparser.add_argument("--decay_lr",
            type = int,
            default = 1
        )
    argparser.add_argument("--fix_emb",
            type = int,
            default = 1
        )
    # added argument for initializer
    argparser.add_argument("--initialization",
            type = str,
            default = 'rand_uni',
            help = "initialization of the weights: rand_uni or Xavier"
            
        )
    
    argparser.add_argument("--email",
            type = str,
            default = None
        )
    argparser.add_argument("--password",
            type = str,
            default = None
        )
    args = argparser.parse_args()
    return args