import random
import time

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

def tournament_selection(k, pop):
    s = []
    for i in range(k):
        s.append(random.choice(pop))
    fittest = []
    max_fitness = -100
    for i in range(len(s)):
        if fittest == []:
            fittest.append(s[i])
            max_fitness = s[i][1]
        else:
            if s[i][1] > max_fitness:
                fittest = [s[i]]
                max_fitness = s[i][1]
            elif s[i][1] == max_fitness:
                fittest.append(s[i])
    return random.choice(fittest)

def genetic_algorithm(n, chi, k, lam, elitism, fitness_function, goal_fitness=None, timeout=None):
    population = []
    for i in range(lam):
        population.append("")
        for j in range(n):
            population[i] += str(random.getrandbits(1))

    for i in range(len(population)):
        population[i] = (population[i], fitness_function(population[i]))


    num_of_gens = 0

    start = time.time()
    current = start

    while ((goal_fitness == None) or not (len([True for p in population if int(p[1]) == int(goal_fitness)]) > 0)) and ((timeout == None) or ((current - start) < timeout)):
        new_pop = []
        population = sorted(population, key = lambda p: p[1], reverse=True)
        for i in range(lam):

            if i < elitism:
                new_pop.append(population[i])
                continue

            x = tournament_selection(k, population)
            y = tournament_selection(k, population)
            child = uniform_crossover(mutation_operator(x[0], chi),
                                      mutation_operator(y[0], chi))

            new_pop.append((child, fitness_function(child)))

        population = new_pop
        num_of_gens += 1
        current = time.time()

    x_best = None
    for p in population:
        if x_best == None:
            x_best = p
        else:
            if p[1] > x_best[1]:
                x_best = p

    print str(lam * num_of_gens) + "\t" + str(x_best[1]) + "\t" + str(x_best[0])

    return (population, num_of_gens, x_best)
