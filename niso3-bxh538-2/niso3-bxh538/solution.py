#!/usr/bin/python2.7

import argparse
import random

import numpy as np

from genetic_algorithm import *

parser = argparse.ArgumentParser(add_help=False, description='NISO Lab 3')

parser.add_argument('-question', type=int, help='Question number')
parser.add_argument('-time_budget', type=int, help='Number of seconds per repetition')
parser.add_argument('-prob', help='Vector of n probabilities')
parser.add_argument('-strategy', help='A representation of the strategy')
parser.add_argument('-state', type=int, help='The state in week t')
parser.add_argument('-crowded', type=int, help='1 if the bar is crowded, 0 otherwise')
parser.add_argument('-repetitions', type=int, help='Number of repititions')
parser.add_argument('-max_t', type=int, help='Number of generations')
parser.add_argument('-weeks', type=int, help='Number of weeks')
parser.add_argument('-lam', '-lambda', type=int, help='Population size')
parser.add_argument('-h', type=int, help='Number of states')

args = parser.parse_args()

def parse_strategy(strategy):
    h = int(strategy[0])
    i = 1
    p = []
    a = []
    b = []
    while i < len(strategy):
        p.append(strategy[i])
        i += 1
        a_line = []
        b_line = []
        for j in range(i, i+h):
            a_line.append(strategy[j])
        i += h
        for j in range(i, i+h):
            b_line.append(strategy[j])
        i += h
        a.append(a_line)
        b.append(b_line)
    return Strategy(p, a, b)

if args.question == 1:
    for i in range(args.repetitions):
        probs = map(float, args.prob.split(' '))
        print sample_index_from_probs(probs)
elif args.question == 2:
    for i in range(args.repetitions):
        strategy = parse_strategy(map(float, args.strategy.split(' ')))
        (d, s) = apply_strategy(args.state, bool(args.crowded), strategy)
        print str(d) + '\t' + str(s)
elif args.question == 3:
    for i in range(args.repetitions):
        genetic_algorithm(10, args.lam, args.h, args.weeks, 0.1, 0.1, args.max_t)
