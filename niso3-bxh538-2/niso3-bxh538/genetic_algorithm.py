import random
import sys
import time

def sample_index_from_probs(probs):
    distribution = zip(normalise(probs), range(len(probs)))
    r = random.random()
    cumulative = 0
    for probability in distribution:
        cumulative += probability[0]
        if r < cumulative:
            return probability[1]

#def sample_index_from_probs(probs):
#    p = normalise(probs)
#    acc = 0
#    r = random.random()
#    for i in range(0, len(p)):
#        acc += p[i]
#        if r < acc:
#            return i

def apply_strategy(state, crowded, strategy):
    trans_matrix = strategy.a if crowded else strategy.b

    new_state = sample_index_from_probs(trans_matrix[state])

    going_to_bar = 1 if random.random() < strategy.probs[new_state] else 0

    return (going_to_bar, new_state)

def evaluate_pop(generation, population, weeks):

    usage = 0

    pop_limit = 0.6*len(population)

    for member in population:
        member.payout = 0
        member.state = 0

    for member in population:
        member.going_bar = int(random.random() < member.strategy.probs[member.state])

    usage += (len([m for m in population if m.going_bar]) / float(len(population)))
    crowded = len([m for m in population if m.going_bar]) >= pop_limit
    for member in population:
        if (member.going_bar and not crowded) or ((not member.going_bar) and crowded):
            member.payout += 1 

    print '0\t'+str(generation)+'\t'+str(len([member for member in population if member.going_bar]))+'\t'+str(int(crowded))+'\t'+'\t'.join([str(member.going_bar) for member in population])

    for w in range(1, weeks):

        for member in population:
            (going, new_state) = apply_strategy(member.state, crowded, member.strategy)
            member.state = new_state
            member.going_bar = going

        usage += (len([m for m in population if m.going_bar]) / float(len(population)))

        crowded = len([m for m in population if m.going_bar]) >= pop_limit

        print str(w)+'\t'+str(generation)+'\t'+str(len([member for member in population if member.going_bar]))+'\t'+str(int(crowded))+'\t'+'\t'.join([str(member.going_bar) for member in population])

        for member in population:
            if (member.going_bar and not crowded) or ((not member.going_bar) and crowded):
                member.payout += 1 

    #print (usage / float(weeks))
    return (usage / float(weeks))

class Strategy:
    def __init__(self, probs, a, b):
        self.probs = probs
        self.a = a
        self.b = b

class Member:
    def __init__(self, strategy, state):
        self.strategy = strategy
        self.state = state
        self.payout = 0
        self.going_bar = False


def crossover(x, y):
    new_strategy = Strategy([], [], [])

    for i in range(len(x.strategy.probs)):
        prob = x.strategy.probs[i] if random.random() < 0.5 else y.strategy.probs[i]
        new_strategy.probs.append(prob)

    for i in range(len(x.strategy.a)):
        a_line = [a for a in x.strategy.a[i]] if random.random() < 0.5 else [a for a in y.strategy.a[i]]
        b_line = [b for b in x.strategy.b[i]] if random.random() < 0.5 else [b for b in y.strategy.b[i]]
        new_strategy.a.append(a_line)
        new_strategy.b.append(b_line)

    return Member(new_strategy, x.state)

def normalise(vector):
    total = sum(vector)
    if total == 0:
        return [1.0 / len(vector) for v in vector]
    else:
        return [float(v) / float(total) for v in vector]

def mutate(x, prob, sd):
    if random.random() >= prob:
        return
    for i in range(len(x.strategy.a)):
        #if random.random() >= prob:
        #    continue
        total_a = 0
        total_b = 0
        for j in range(len(x.strategy.a[i])):
            x.strategy.a[i][j] += random.gauss(0.0, sd) 
            x.strategy.a[i][j] = max(0.0, x.strategy.a[i][j])
            total_a += x.strategy.a[i][j]
            x.strategy.b[i][j] += random.gauss(0.0, sd)
            x.strategy.b[i][j] = max(0.0, x.strategy.b[i][j])
            total_b += x.strategy.b[i][j]

    for i in range(len(x.strategy.probs)):
        #if random.random() >= prob:
        #    continue
        x.strategy.probs[i] += random.gauss(0.0, sd)
        x.strategy.probs[i] = max(0.0, min(1.0, x.strategy.probs[i]))

def tournament_selection(k, pop):
    s = []
    for i in range(k):
        s.append(random.choice(pop))
    fittest = []
    max_fitness = -100
    for member in s:
        if fittest == []:
            fittest.append(member)
            max_fitness = member.payout
        else:
            if member.payout > max_fitness:
                fittest = [member]
                max_fitness = member.payout
            elif member.payout == max_fitness:
                fittest.append(member)
    return random.choice(fittest)


def uniform_member(h):
    a = []
    b = []
    probs = []
    for i in range(h):
        a.append([])
        probs.append(1.0 / 2)
        b.append([])
        for j in range(h):
            a[i].append(1.0 / float(h)) 
            b[i].append(1.0 / float(h)) 

    return Member(Strategy(probs, a, b), 0)

#def random_ind(h):
#    states = range(0, h)
#    p = [random.random() for i in states]   
#    a = [normalise([random.random() for i in states]) for i in states]
#    b = [normalise([random.random() for i in states]) for i in states]
#    return Member(Strategy(p, a, b), 0)

        
def genetic_algorithm(k, lam, h, weeks, mutation_rate, sd, timeout=None):
    #print timeout

    population = []
    for i in range(lam):
        population.append(uniform_member(h))

    usage = 0
    num_of_gens = 0

    start = time.time()
    current = start

    while (timeout == None) or (num_of_gens < timeout):
        new_pop = []

        evaluate_pop(num_of_gens, population, weeks)
        for i in range(lam):
            x = tournament_selection(k, population)
            y = tournament_selection(k, population)
            mutate(x, mutation_rate, sd)
            mutate(y, mutation_rate, sd)
            child = crossover(x, y)
            new_pop.append(child)

        #usage = evaluate_pop(num_of_gens, new_pop, weeks)
        num_of_gens += 1
        population = new_pop
        current = time.time()

    x_best = None
    for p in population:
        if x_best == None:
            x_best = p
        else:
            if p.payout > x_best.payout:
                x_best = p

    return (population, num_of_gens)
