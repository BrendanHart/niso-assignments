#!/usr/bin/python2.7

import argparse
import random

from genetic_algorithm import genetic_algorithm

parser = argparse.ArgumentParser(description='NISO Lab 2')

parser.add_argument('-question', type=int, help='Question number')
parser.add_argument('-clause', help='A clause description')
parser.add_argument('-assignment', help='An assignment as a bit string')
parser.add_argument('-wdimacs', help='Name of file in WDIMACS format')
parser.add_argument('-time_budget', type=int, help='Number of seconds per repetition')
parser.add_argument('-repetitions', type=int, help='Number of repetitions')

args = parser.parse_args()

def parse_clause(clause_description):
    return map(int, clause_description.split(' ')[1:-1])

def satisfies_clause(clause, assignment):
    satisfied = False
    for v in clause:
        i = int(abs(v))-1
        satisfied = satisfied or (v > 0 and assignment[i] == '1') or (v < 0 and assignment[i] == '0')
    return int(satisfied)

def num_of_satisfied_clauses(clauses, assignment):
    count = 0
    for c in clauses:
        if satisfies_clause(c, assignment) == 1:
            count += 1
    return count

def get_from_wdimacs(file_name):
    f = open(file_name)
    clauses = []
    num_vars = 0
    for l in f:
        if l[0] == 'c':
            continue
        elif l[0] == 'p':
            num_vars = int(l.split(' ')[2])
            continue
        else:
            clauses.append(parse_clause(l)) 
    return (clauses, num_vars)

if args.question == 1:
    clause = parse_clause(args.clause)
    print satisfies_clause(clause, args.assignment)
    print num_of_satisfied_clauses([clause], args.assignment)
elif args.question == 2:
    (clauses, num_vars) = get_from_wdimacs(args.wdimacs)
    print num_of_satisfied_clauses(clauses, args.assignment)
elif args.question == 3:
    (clauses, num_vars) = get_from_wdimacs(args.wdimacs)
    for i in range(args.repetitions):
        (pop, gens, best) = genetic_algorithm(num_vars, 0.6, 2, 20, 0, (lambda x: num_of_satisfied_clauses(clauses, x)), len(clauses), args.time_budget)

