# Jake Burton
# 40278490
# SET11508 Emergent Computing for Optimisation.


# Import relevant libraries.
import pandas as pd
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
# import matplotlib.pyplot as plt
# from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# read data
data = pd.read_csv("clean-data.csv").reset_index(drop=True)

# Initialise variables used throughout the algortihm
num_players = len(data.index)
points = data['Points']
costs = data['Cost']
MAX_COST = 100

# create lists with all elements initialised to 0
gk = np.zeros(num_players)
mid = np.zeros(num_players)
defe = np.zeros(num_players)
stri = np.zeros(num_players)

# Create arrays with boolean values representing the player types.
for i in range(num_players):
    if data['Position'][i] == 'GK':
        gk[i] = 1
    elif data['Position'][i] == 'DEF':
        defe[i] = 1
    elif data['Position'][i] == 'MID':
        mid[i] = 1
    elif data['Position'][i] == 'STR':
        stri[i] = 1

# Gets the total cost and points of the players, used to test the input file is
#  being read correctly, printing the results.
TOTAL_COST = 0
for i in range(num_players):
    TOTAL_COST += costs[i]
TOTAL_POINTS = 0
for i in range(num_players):
    TOTAL_POINTS += points[i]
print("Total Cost: "+str(TOTAL_COST))
print("Total Points: "+str(TOTAL_POINTS))


# Function evaluates the provided team to make sure that the number of players
# is 11.
def eval_num_players(individual):
    num_bool = True
    num_check = np.sum(individual)
    if num_check != 11:
        num_bool = False
    return(num_bool)


# Function checks that there is only one goalie in the provided team.
# Returns True if number of goalies is equal to 1, else returns False.
def eval_goalie(individual):
    goalie_bool = True
    goalie_check = np.sum(np.multiply(gk, individual))
    if goalie_check != 1:
        goalie_bool = False
    return goalie_bool


# Function checks that there are a valid number of defenders in the provided
# team.
# Returns True if num defenders is between 3 and 5, else returns False.
def eval_def(individual):
    def_bool = True
    def_check = np.sum(np.multiply(defe, individual))
    if def_check > 5 or def_check < 3:
        def_bool = False
    return def_bool


# Function checks that there are a valid number of midfielders in the provided
# team.
# Returns True if num midfielders is between 3 and 5, else returns False.
def eval_mid(individual):
    mid_bool = True
    mid_check = np.sum(np.multiply(mid, individual))
    if mid_check > 5 or mid_check < 3:
        mid_bool = False
    return mid_bool


# Function checks that there are a valid number of strikers in the provided
# team.
# Returns True if num strikers is between 1 and 3, else returns False.
def eval_stri(individual):
    stri_bool = True
    stri_check = np.sum(np.multiply(stri, individual))
    if stri_check > 3 or stri_check < 1:
        stri_bool = False
    return stri_bool


# Function collates all of the evaluation methods and returns True is all
# functions return True, else returns False
def eval_team(ind):
    nm = eval_num_players(ind)
    go = eval_goalie(ind)
    de = eval_def(ind)
    mi = eval_mid(ind)
    st = eval_stri(ind)
    tot = (nm and go and de and mi and st)
    # print(tot)
    return tot


# Evaluation Method, ensures that the team (individual) is valid.
def evalKnapsack(individual):
    cost = 0.0
    value = 0.0
    for item in range(num_players):
        if (individual[item] == 1):
            cost += data['Cost'][item]
            value += data['Points'][item]
    ev_te = eval_team(individual)
    if(cost > MAX_COST or ev_te is False):
        return 0,
    return value,


# Final Evaluation Method provided by Emma Hart
def check_constraints(individual):
    broken_constraints = 0
    # exactly 11 players
    c1 = np.sum(individual)
    if c1 != 11:
        broken_constraints += 1
        print("total players is %s " % (c1))
    # need cost <= 100"
    c2 = np.sum(np.multiply(costs, individual))
    if c2 > 100:
        broken_constraints += 1
        print("cost is %s " % (c2))
    # need only 1 GK
    c3 = np.sum(np.multiply(gk, individual))
    if c3 != 1:
        broken_constraints += 1
        print("goalies is %s " % (c3))
    # need less than 3-5 DEF"
    c4 = np.sum(np.multiply(defe, individual))
    if c4 > 5 or c4 < 3:
        broken_constraints += 1
        print("DEFE is %s " % (c4))
    # need 3- 5 MID
    c5 = np.sum(np.multiply(mid, individual))
    if c5 > 5 or c5 < 3:
        broken_constraints += 1
        print("MID is %s " % (c5))
    # need 1 -1 3 STR"
    c6 = np.sum(np.multiply(stri, individual))
    if c6 > 3 or c6 < 1:
        broken_constraints += 1
        print("STR is %s " % (c6))
    # get indices of players selected
    selectedPlayers = [idx for idx, element in enumerate(individual) if element == 1]
    totalpoints = np.sum(np.multiply(points, individual))
    print("total broken constraints: %s" % (broken_constraints))
    print("total points: %s" % (totalpoints))
    print("total cost is %s" % (c2))
    print("selected players are %s" % (selectedPlayers))
    return broken_constraints, totalpoints


# Initialises a valid team, always returning a valid individual.
# Takes in a blank Individual and the total number of players.
# Gets the values for the player positions by creating an array of random
# numbers between 0 and 1 and then multiplying all the values by the number of
# players. The calling of the eval_team method and the while loop ensures that
# the returned team does not have duplicates of the same player.
def initialise_team(incc, num):
    while(1):
        int_team = (np.random.rand(11)*num)
        ind = incc(np.zeros(num))
        for index in int_team:
            ind[int(index)] = 1
        ev_te = eval_team(ind)
        if(ev_te is False):
            continue
        # print(ind)
        return ind


MUTPB = 0.01
CXPB = 0.6
POPSIZE = 500
NGEN = 50
TNSIZE = 2
OPER = tools.cxTwoPoint

# Adds all of the relevant DEAP functions.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", initialise_team, creator.Individual, num_players)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Main function
def main():
    # All of the operators for the EA used.
    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", OPER)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)
    toolbox.register("select", tools.selTournament, tournsize=TNSIZE)
    pop = toolbox.population(n=POPSIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


# FINAL SOLUTION CHECKER
MUTPB = 0.01
CXPB = 0.6
POPSIZE = 5000
NGEN = 100
TNSIZE = 2
OPER = tools.cxTwoPoint
pop, log, hof = main()
best = hof[0].fitness.values[0]
max = log.select("max")
for i in range(200):
    fit = max[i]
    if fit == best:
        break
print(check_constraints(hof[0]))
print("max fitness found is %s at generation %s" % (best, i))


# TESTING AND EXPERIMENTATION
"""
# Population Size Testing
columns = ['popsize', 'fitness', 'genMaxFound']
df = pd.DataFrame(columns=columns)
for POPSIZE in range(500, 2001, 500):
    for reps in range(10):
        pop, log, hof = main()
        best = hof[0].fitness.values[0]
        max = log.select("max")
        for gen in range(NGEN):
            if max[gen] == best:
                break
        df = df.append({'popsize': POPSIZE, 'fitness': best, 'genMaxFound': gen}, ignore_index=True)
# plot the boxplot of fitness per population size
boxplot = df.boxplot(column=['fitness'], by=['popsize'])
plt.savefig('fitvpop.png')
# plot genMaxFound per population size
boxplot = df.boxplot(column=['genMaxFound'], by=['popsize'])
plt.savefig('genvpop.png')
p500 = df.fitness[df.popsize == 500]
p2000 = df.fitness[df.popsize == 2000]
stat, p1 = shapiro(p500)
stat, p2 = shapiro(p2000)
if(p1 > 0.05 and p2 > 0.05):
    print("Both Gaussian")
    stat, p = ttest_ind(p500, p2000)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
else:
    print("One or more are not Gaussian")
    stat, p = mannwhitneyu(p500, p2000)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
"""
"""
# Tournament Size Testing
columns = ['tournsize', 'fitness', 'genMaxFound']
df = pd.DataFrame(columns=columns)
for TNSIZE in range(2, 11, 4):
    for reps in range(10):
        pop, log, hof = main()
        best = hof[0].fitness.values[0]
        max = log.select("max")
        for gen in range(NGEN):
            if max[gen] == best:
                break
        df = df.append({'tournsize': TNSIZE, 'fitness': best, 'genMaxFound': gen}, ignore_index=True)
# plot the boxplot of fitness per population size
boxplot = df.boxplot(column=['fitness'], by=['tournsize'])
plt.savefig('fitvtourn.png')
# plot genMaxFound per population size
boxplot = df.boxplot(column=['genMaxFound'], by=['tournsize'])
plt.savefig('genvtourn.png')
p2t = df.fitness[df.tournsize == 2]
p10t = df.fitness[df.tournsize == 10]
stat, p1 = shapiro(p2t)
stat, p2 = shapiro(p10t)
if(p1 > 0.05 and p2 > 0.05):
    print("Both Gaussian")
    stat, p = ttest_ind(p2t, p10t)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
else:
    print("One or more are not Gaussian")
    stat, p = mannwhitneyu(p2t, p10t)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
"""
"""
# Mutation probability testing
columns = ['mutpb', 'fitness', 'genMaxFound']
df = pd.DataFrame(columns=columns)
for MUTPBts in range(1, 6, 1):
    MUTPB = MUTPBts/100
    for reps in range(10):
        pop, log, hof = main()
        best = hof[0].fitness.values[0]
        max = log.select("max")
        for gen in range(NGEN):
            if max[gen] == best:
                break
        df = df.append({'mutpb': MUTPB, 'fitness': best, 'genMaxFound': gen}, ignore_index=True)
boxplot = df.boxplot(column=['fitness'], by=['mutpb'])
plt.savefig('fitvmut.png')
boxplot = df.boxplot(column=['genMaxFound'], by=['mutpb'])
plt.savefig('genvmut.png')

p001 = df.fitness[df.mutpb == 0.01]
p005 = df.fitness[df.mutpb == 0.05]
stat, p1 = shapiro(p001)
stat, p2 = shapiro(p005)
if(p1 > 0.05 and p2 > 0.05):
    print("Both Gaussian")
    stat, p = ttest_ind(p001, p005)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
else:
    print("One or more are not Gaussian")
    stat, p = mannwhitneyu(p001, p005)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
"""

"""
# NGEN Testing
columns = ['ngen', 'fitness', 'genMaxFound']
df = pd.DataFrame(columns=columns)
for NGEN in range(10, 51, 10):
    for reps in range(10):
        pop, log, hof = main()
        best = hof[0].fitness.values[0]
        max = log.select("max")
        for gen in range(NGEN):
            if max[gen] == best:
                break
        df = df.append({'ngen': NGEN, 'fitness': best, 'genMaxFound': gen}, ignore_index=True)
boxplot = df.boxplot(column=['fitness'], by=['ngen'])
plt.savefig('fitvsngen.png')

p10 = df.fitness[df.ngen == 10]
p50 = df.fitness[df.ngen == 50]
stat, p1 = shapiro(p10)
stat, p2 = shapiro(p50)
if(p1 > 0.05 and p2 > 0.05):
    print("Both Gaussian")
    stat, p = ttest_ind(p10, p50)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
else:
    print("One or more are not Gaussian")
    stat, p = mannwhitneyu(p10, p50)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
"""
"""
# Mate operator testing
types = [tools.cxOnePoint, tools.cxTwoPoint]
columns = ['mate', 'fitness', 'genMaxFound']
df = pd.DataFrame(columns=columns)
for index in range(len(types)):
    OPER = types[index]
    for reps in range(10):
        pop, log, hof = main()
        best = hof[0].fitness.values[0]
        max = log.select("max")
        for gen in range(NGEN):
            if max[gen] == best:
                break
        df = df.append({'mate': (index+1), 'fitness': best, 'genMaxFound': gen}, ignore_index=True)
boxplot = df.boxplot(column=['fitness'], by=['mate'])
plt.savefig('fitvmate.png')
boxplot = df.boxplot(column=['genMaxFound'], by=['mate'])
plt.savefig('genvmate.png')
p1p = df.fitness[df.mate == 1]
p2p = df.fitness[df.mate == 2]
stat, p1 = shapiro(p1p)
stat, p2 = shapiro(p2p)
if(p1 > 0.05 and p2 > 0.05):
    print("Both Gaussian")
    stat, p = ttest_ind(p1p, p2p)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
else:
    print("One or more are not Gaussian")
    stat, p = mannwhitneyu(p1p, p2p)
    if p > 0.05:
        print("Probably from same distribution")
    else:
        print("Probably different distribution")
"""
