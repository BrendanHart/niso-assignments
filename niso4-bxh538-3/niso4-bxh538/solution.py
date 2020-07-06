#!/usr/bin/python2.7

import argparse
import random
import math
import copy
import signal
import time
import sys

import numpy as np

def evaluate(f_dict, sexp, X):
    f = None
    params = [X]

    try:
        return float(sexp)
    except TypeError:
        pass
    except ValueError:
        pass

    for elem in sexp:
        if type(elem) == tuple:
            f = elem[0]
        elif type(elem) == list:
            params.append(evaluate(f_dict, elem, X))
        else:
            params.append(elem)

    return f_dict[f][0](*params)

def crossover(x, y):
    copy_x = copy.deepcopy(x[0])
    copy_y = copy.deepcopy(y[0])


    branches_x = get_branches(copy_x)[1:]
    branches_y = get_branches(copy_y)[1:]
    if branches_x == [] and branches_y == []:
        return ((copy_y, None), (copy_x, None))
    elif branches_x == []:
        choice_y = random.choice(branches_y)
        y_branch = copy_y
        for i in range(len(choice_y)):
            y_branch = y_branch[choice_y[i]]

        tmp = copy.deepcopy(y_branch)
        y_branch = copy.deepcopy(copy_x)
        copy_x = tmp

        return ((copy_x, None), (copy_y, None))

    elif branches_y == []:
        choice_x = random.choice(branches_x)
        x_branch = copy_x
        for i in range(len(choice_x)):
            x_branch = x_branch[choice_x[i]]

        tmp = copy.deepcopy(x_branch)
        x_branch = copy.deepcopy(copy_y)
        copy_y = tmp

        return ((copy_x, None), (copy_y, None))

    else:
        choice_x = random.choice(branches_x)
        x_branch = copy_x
        for i in range(len(choice_x)):
            x_branch = x_branch[choice_x[i]]

        choice_y = random.choice(branches_y)
        y_branch = copy_y
        for i in range(len(choice_y)):
            y_branch = y_branch[choice_y[i]]


        tmp = copy.deepcopy(x_branch)
        x_branch = copy.deepcopy(y_branch)
        y_branch = tmp

        return ((copy_x, None), (copy_y, None))


def mutate(x, f_dict, rand_range, depth):
    copy_x = copy.deepcopy(x[0])
    branches = get_branches(copy_x)[1:]

    if branches == []:
        return (full_method(random.randint(1, depth), f_dict, rand_range), None)

    choice = random.choice(branches)
    branch_before = copy_x

    for i in range(len(choice)-1):
        branch_before = branch_before[choice[i]]

    branch_before[choice[-1]] = full_method(random.randint(1, depth), f_dict, rand_range)

    return (copy_x, None)

def tournament_selection(k, pop):
    s = []
    for i in range(k):
        s.append(random.choice(pop))
    fittest = []
    max_fitness = sys.float_info.max
    for member in s:
        if fittest == []:
            fittest.append(member)
            max_fitness = member[1]
        else:
            if member[1] < max_fitness:
                fittest = [member]
                max_fitness = member[1]
            elif member[1] == max_fitness:
                fittest.append(member)
    return random.choice(fittest)

def calc_fitness_np(f_dict, expr, X, y):
    eval_x = np.array([evaluate(f_dict, expr, x) for x in X])
    dif = y - eval_x
    return np.dot(dif ,dif) / float(len(y))

def calc_fitness(f_dict, expr, data):
    sum_error = 0

    for (X, y) in data:
        sum_error += np.power((y - evaluate(f_dict, expr, X)), 2)

    return sum_error / float(len(data))

def full_method(tree_depth, f_dict, rand_range):
    if tree_depth <= 0:
        return None

    random_branch = None
    f = random.choice(f_dict.keys())
    if tree_depth == 1:
        random_branch = [(f,)] + [random.uniform((-1 * rand_range), rand_range) for i in range(0, f_dict[f][1])]
    else:
        random_branch = [(f,)] + [full_method(tree_depth-1, f_dict, rand_range) for i in range(0, f_dict[f][1])]

    return random_branch

def get_branches(tree):
    nodes = [[]]
    
    for i in range(len(tree)):
        if type(tree[i]) == list:
            nodes += (map(lambda x : [i] + x, get_branches(tree[i])))
    return nodes

timeup = False

def handler(signum, frame):
    raise StopIteration("Error")


def genetic_algorithm(k, lam, mut_rate, tree_depth, mut_depth, f_dict, rand_range, data, timeout=None):

    timeup = False
    x_best = full_method(tree_depth, f_dict, rand_range)
    x_best = (x_best, calc_fitness(f_dict, x_best, data))

    signal.signal(signal.SIGALRM, handler)
    
    signal.alarm(timeout)

    try:
        population = []
        for i in range(lam):
            sexpr = full_method(tree_depth, f_dict, rand_range)
            population.append((sexpr, len(get_branches(sexpr)) + calc_fitness(f_dict, sexpr, data)))


        for p in population:
            if x_best == None:
                x_best = p
            else:
                if p[1] < x_best[1]:
                    x_best = p

        num_of_gens = 0

    #    start = time.time()
    #    current = start
    
        while True:
        #while (timeout == None) or ((current - start) < timeout):
            new_pop = []
            while len(new_pop) < lam:
                x = tournament_selection(k, population)
                y = tournament_selection(k, population)
                if random.random() < mut_rate:
                    x = mutate(x, f_dict, rand_range, mut_depth)
                if random.random() < mut_rate:
                    y = mutate(y, f_dict, rand_range, mut_depth)
                (c1, c2) = crossover(x, y)
                c1 = (c1[0], len(get_branches(c1[0])) + calc_fitness(f_dict, c1[0], data))
                c2 = (c2[0], len(get_branches(c2[0])) + calc_fitness(f_dict, c2[0], data))
                new_pop.append(c1)
                new_pop.append(c2)

            population = new_pop

            x_best = None
            for p in population:
                if x_best == None:
                    x_best = p
                else:
                    if p[1] < x_best[1]:
                        x_best = p
            num_of_gens += 1
            #print num_of_gens
        return x_best
    except StopIteration:
        timeup = True
        return x_best


parser = argparse.ArgumentParser(add_help=False, description='NISO Lab 3')

parser.add_argument('-question', type=int, help='Question number')
parser.add_argument('-time_budget', type=int, help='Number of seconds per repetition')
parser.add_argument('-repetitions', type=int, help='Number of repititions')
parser.add_argument('-max_t', type=int, help='Number of generations')
parser.add_argument('-lam', '-lambda', type=int, help='Population size')
parser.add_argument('-n', type=int, help='Length of the input vector')
parser.add_argument('-m', type=int, help='Number of data points')
parser.add_argument('-x', help='The input vector')
parser.add_argument('-expr', help='An expression')
parser.add_argument('-data', help='The name of the file containing the training data')


def fix_floats(sexpr):
    for i in range(len(sexpr)):
        if type(sexpr[i]) == tuple:
            try:
                f = float(sexpr[i][0])
                sexpr[i] = f
            except TypeError:
                pass
            except ValueError:
                pass
        elif type(sexpr[i]) == list:
            sexpr[i] = fix_floats(sexpr[i])

    return sexpr

args = parser.parse_args()

from string import whitespace

atom_end = set('()"\'') | set(whitespace)

def parse(sexp):
    stack, i, length = [[]], 0, len(sexp)
    while i < length:
        c = sexp[i]

        #print c, stack
        reading = type(stack[-1])
        if reading == list:
            if   c == '(': stack.append([])
            elif c == ')': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '"': stack.append('')
            elif c == "'": stack.append([('quote',)])
            elif c in whitespace: pass
            else: stack.append((c,))
        elif reading == str:
            if   c == '"': 
                stack[-2].append(stack.pop())
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
            elif c == '\\': 
                i += 1
                stack[-1] += sexp[i]
            else: stack[-1] += c
        elif reading == tuple:
            if c in atom_end:
                atom = stack.pop()
                if atom[0][0].isdigit(): stack[-1].append(eval(atom[0]))
                else: stack[-1].append(atom)
                if stack[-1][0] == ('quote',): stack[-2].append(stack.pop())
                continue
            else: stack[-1] = ((stack[-1][0] + c),)
        i += 1
    return fix_floats(stack.pop())


#def read_data(file_name):
#    f = open(file_name)
#    X = []
#    y = []
#    for l in f.read().splitlines():
#        s = map(float, l.split('\t'))
#        X.append(s[:len(s)-1])
#        y.append(s[len(s)-1])
#    return (X, y)

def read_data(file_name):
    f = open(file_name)
    X = []
    for l in f.read().splitlines():
        s = map(float, l.split('\t'))
        X.append((s[:len(s)-1], s[len(s)-1]))
    return X


def pow_catch(X, x, y):
    try:
        return x ** y
    except ValueError:
        return 0
    except ZeroDivisionError:
        return 0
    except OverflowError:
        return 0

def avg_catch(X, x, y):
    try:
        k = abs(math.floor(x)) % len(X)
        l = abs(math.floor(y)) % len(X)
        i = min(k, l)
        j = max(k, l)
        return (1.0 / abs(k - l)) * sum( X[int(i):int(j)] )
    except ValueError:
        return 0
    except ZeroDivisionError:
        return 0
    except OverflowError:
        return 0
     
def log_catch(X, x):
    try:
        return math.log(x, 2)
    except ValueError:
        return 0

def add_catch(X, x, y):
    try:
        return x + y
    except OverflowError:
        return 0

def sub_catch(X, x, y):
    try:
        return x - y
    except ValueError:
        return 0
    except OverflowError:
        return 0

def div_catch(X, x, y):
    try:
        return x / y
    except ValueError:
        return 0
    except ZeroDivisionError:
        return 0

def mul_catch(X, x, y):
    try:
        return x * y
    except OverflowError:
        return 0

def sqrt_catch(X, x):
    try:
        return math.sqrt(x)
    except ValueError:
        return 0

def exp_catch(X, x):
    try:
        return math.e ** x
    except ValueError:
        return 0
    except OverflowError:
        return 0

def data(X, x):
    r = abs(math.floor(x)) % len(X)
    try:
        return X[int(r)]
    except ValueError:
        return 0
    except TypeError:
        return 0

def diff(X, x, y):
    k = abs(math.floor(x)) % len(X)
    l = abs(math.floor(y)) % len(X)
    #print >> sys.stderr, (str(k) + " " + str(l) + " " + str(x) + " "+ str(y)) 
    try:
        return X[int(k)] - X[int(l)]
    except ValueError:
        return 0
    except TypeError:
        return 0

f_dict = dict()
f_dict['data'] = (data, 1)
f_dict['add'] = (add_catch, 2)
f_dict['sub'] = (sub_catch, 2)
f_dict['div'] = (div_catch, 2)
f_dict['mul'] = (mul_catch, 2)
f_dict['pow'] = (pow_catch, 2)
f_dict['sqrt'] = (sqrt_catch, 1)
f_dict['log'] = (log_catch, 1)
f_dict['exp'] = (exp_catch, 1)
f_dict['max'] = ((lambda X, x, y : max(x, y)), 2)
f_dict['ifleq'] = ((lambda X, w, x ,y ,z : y if w <= x else z), 4)
f_dict['diff'] = (diff, 2)
f_dict['avg'] = (avg_catch, 2)

def to_sexp(sexp):
    if type(sexp) == tuple:
        return str(sexp[0])
    if type(sexp) == list:
        s = '('
        for elem in sexp:
            s += to_sexp(elem)
            s += ' '
        s = s[:-1]
        s += ')'
        return s
    return str(sexp)

if args.question == 1:
    e = parse(args.expr)[0]
    print evaluate(f_dict, e, map(float, args.x.split(' ')))
elif args.question == 2:
    e = parse(args.expr)[0]
    data = read_data(args.data)
    print calc_fitness(f_dict, e, data)
elif args.question == 3:
    data = read_data(args.data)
    best = genetic_algorithm(2, args.lam, 0.3, 4, 2, f_dict, 100, data, args.time_budget)
    print to_sexp(best[0])
elif args.question == 5:
    f = open('k_results.csv', 'a')
    data = read_data(args.data)
    k_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for k in k_vals:
        f.write(str(k))
        for i in range(70):
            print (i, k)
            best = genetic_algorithm(k, 100, 0.3, 4, 2, f_dict, 100, data, args.time_budget)
            print best
            f.write(','+str(calc_fitness(f_dict, best, data)))
        f.write('\n')
