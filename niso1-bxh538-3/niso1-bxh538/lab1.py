import argparse
import random

parser = argparse.ArgumentParser(description='NISO Lab 1')

parser.add_argument('-question', type=int, help='Question number')
parser.add_argument('-bits_x', help='The input bitstring x')
parser.add_argument('-bits_y', help='The input bitstring y')
parser.add_argument('-chi', type=float, help='The specification of the mutation rate')
parser.add_argument('-repetitions', type=int, help='The number of repetitions')
parser.add_argument('-population', help='The population')
parser.add_argument('-k', type=int, help='The tournament size')
parser.add_argument('-n', type=int, help='The bitstring length')
parser.add_argument('-lam', '-lambda', type=int, help='The population size')

args = parser.parse_args()

def mutation_operator(bits_x, chi):
    rate = chi / float(len(bits_x))
    output = ""
    for c in bits_x:
        if random.random() < rate:
            output += '1' if c == '0' else '0'
        else:
            output += c
    return output

def uniform_crossover(bits_x, bits_y):
    output = ""
    for i in range(len(bits_x)):
        if bits_x[i] == bits_y[i]:
            output += bits_x[i]
        else:
            output += str(random.getrandbits(1))
    return output

def one_max(bits_x):
    result = 0
    for bit in bits_x:
        result += int(bit)
    return result

def tournament_selection(k, pop, fitness_function=one_max):
    s = []
    for i in range(k):
        s.append(random.choice(pop))
    fittest = []
    max_fitness = -100
    for i in range(len(s)):
        if fittest == []:
            fittest.append(s[i])
            max_fitness = fitness_function(s[i])
        else:
            if fitness_function(s[i]) > max_fitness:
                fittest = [s[i]]
                max_fitness = fitness_function(s[i])
            elif fitness_function(s[i]) == max_fitness:
                fittest.append(s[i])
    return random.choice(fittest)

def genetic_algorithm(n, chi, k, lam, timeout=None):
    population = []
    for i in range(lam):
        population.append("")
        for j in range(n):
            population[i] += str(random.getrandbits(1))

    num_of_gens = 0


    while not (len([True for p in population if one_max(p) == n]) > 0) and (timeout == None or num_of_gens < timeout):
        new_pop = []
        for i in range(lam):
            x = tournament_selection(k, population)
            y = tournament_selection(k, population)
            new_pop.append(uniform_crossover(mutation_operator(x, chi), 
                                             mutation_operator(y, chi)))
        population = new_pop
        num_of_gens += 1

    x_best = None
    for p in population:
        if x_best == None:
            x_best = p
        else:
            if one_max(p) > one_max(x_best):
                x_best = p

    print str(n) + "\t" + str(chi) + "\t" + str(lam) + "\t" + str(k) + "\t" + str(num_of_gens) + "\t" + str(one_max(x_best)) + "\t" + str(x_best)
    return (population, num_of_gens)

if args.question == 1:
    for i in range(args.repetitions):
        print mutation_operator(args.bits_x, args.chi)
elif args.question == 2:
    for i in range(args.repetitions):
        print uniform_crossover(args.bits_x, args.bits_y)
elif args.question == 3:
    print one_max(args.bits_x)
elif args.question == 4:
    pop = args.population.split(" ")
    for i in range(args.repetitions):
        print tournament_selection(args.k, pop)
elif args.question == 5:
    for i in range(args.repetitions):
        genetic_algorithm(args.n, args.chi, args.k, args.lam)
elif args.question == 6:

    
    k_intervals = [2, 3, 4, 5]
    k_results = dict([(k, []) for k in k_intervals])

    n_intervals = range(20, 200, 10)
    n_results = dict([(n, []) for n in n_intervals])

    chi_intervals = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    chi_results = dict([(chi, []) for chi in chi_intervals])

    lam_intervals = range(10, 1001, 45)
    lam_results = dict([(lam, []) for lam in lam_intervals])

    repeats = 100

    for i in range(repeats):
        print "REPEAT: " + str(i)
        print "k in " + str(k_intervals)
        for k in k_intervals:
            (pop, gens) = genetic_algorithm(200, 0.6, k, 100, 2500)
            k_results[k].append(gens * 100)
        print "n in " + str(n_intervals)
        for n in n_intervals:
            (pop, gens) = genetic_algorithm(n, 0.6, 2, 100, 2500)
            n_results[n].append(gens * 100)
        print "Chi in " + str(chi_intervals)
        for chi in chi_intervals:
            (pop, gens) = genetic_algorithm(200, chi, 2, 100, 2500)
            chi_results[chi].append(gens * 100)
        print "lamba in " + str(lam_intervals)
        for lam in lam_intervals:
            (pop, gens) = genetic_algorithm(200, 0.6, 2, lam, 2500)
            lam_results[lam].append(gens * lam)

    f = open('k_results.csv', 'w')
    for k, v in k_results.iteritems():
        if v == []:
            continue
        f.write(str(k) + ',' + reduce((lambda x, y: x + ',' + y), map(str, k_results[k])) + '\n')
    f = open('n_results.csv', 'w')
    for k, v in n_results.iteritems():
        if v == []:
            continue
        f.write(str(k) + ',' + reduce((lambda x, y: x + ',' + y), map(str, n_results[k])) + '\n')
    f = open('chi_results.csv', 'w')
    for k, v in chi_results.iteritems():
        if v == []:
            continue
        f.write(str(k) + ',' + reduce((lambda x, y: x + ',' + y), map(str, chi_results[k])) + '\n')
    f = open('lam_results.csv', 'w')
    for k, v in lam_results.iteritems():
        if v == []:
            continue
        f.write(str(k) + ',' + reduce((lambda x, y: x + ',' + y), map(str, lam_results[k])) + '\n')

