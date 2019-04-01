"""
Run and analyze all the algorithms on the corresponding benchmark functions
"""
import numpy as np
import TestFunction
from FireflyOpt import FireflyOptimizer
from DifferentialEvolution import diffevo
from ParticleSwarm import PSO
import matplotlib.pyplot as plt

# perform optimization experiments of firefly on the four-peak function
n_set = [10, 25, 50]    # population size
trials = 30         # number of trials performed per setting
start_seed = 2019
# different setting for alpha, beta, gamma
alphas = [0, 0.5, 1]
betas = [0, 1]
gammas = [0.1, 1]
four_peak_result = []
print('Run firefly optimization on four-peak function.')
# for each combination of parameters, run the firefly for 30 trials
print('pop_size, alpha, beta, gamma, mean_g_best, std_g_best:\n')
for n in n_set:
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                stats = [n, alpha, beta, gamma]
                g_best = []
                for trail in range(trials):
                    fireflyOpt = FireflyOptimizer(TestFunction.four_peaks, [-5, 5], pop_size=n, dims=2, max_iters=30,
                                                  alpha=alpha, beta_0=beta, gamma=gamma)
                    [pos_best, intensity_best] = fireflyOpt.run_optim()
                    g_best.append(intensity_best)
                mean_g = np.mean(g_best)
                std_g = np.std(g_best)
                stats.extend([mean_g, std_g])
                print(stats)
                four_peak_result.append(stats)

# perform experiments of firefly optimization on Egg Create Function and D-dimensional function
n_set = [25, 50, 75]    # population size
D = [2, 8, 16]          # number of dimensions for benchmark function
print('Running firefly on egg create function.')
print('pop_size, alpha, beta, gamma, mean_g_best, std_g_best:\n')
for n in n_set:
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                stats = [n, alpha, beta, gamma]
                g_best = []
                for trail in range(trials):
                    fireflyOpt = FireflyOptimizer(TestFunction.egg_create, [-5, 5], pop_size=n, dims=2, max_iters=30,
                                                  alpha=alpha, beta_0=beta, gamma=gamma)
                    [pos_best, intensity_best] = fireflyOpt.run_optim()
                    g_best.append(intensity_best)
                mean_g = np.mean(g_best)
                std_g = np.std(g_best)
                stats.extend([mean_g, std_g])
                print(stats)

print('Running firefly on D-dimensional function (exponential function).')
print('Dimension, pop_size, alpha, beta, gamma, mean_g_best, std_g_best:\n')
for d in D:
    for n in n_set:
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    stats = [d, n, alpha, beta, gamma]
                    g_best = []
                    for trail in range(trials):
                        fireflyOpt = FireflyOptimizer(TestFunction.exponential_, [-1, 1], pop_size=n, dims=d,
                                                      max_iters=30, alpha=alpha, beta_0=beta, gamma=gamma)
                        [pos_best, intensity_best] = fireflyOpt.run_optim()
                        g_best.append(intensity_best)
                    mean_g = np.mean(g_best)
                    std_g = np.std(g_best)
                    stats.extend([mean_g, std_g])
                    print(stats)

print('Running differential evolution on 8 dimensional Ackley function')
print('pop_size, F, Cr, mean_g_best, std_g_best:\n')
n_set = [20, 40, 60]        # population size
F_set = [0.5, 0.8]         # differential weight
Cr_set = [0.5, 0.75]        # crossover probability

for n in n_set:
    for F in F_set:
        for Cr in Cr_set:
            stats = [n, F, Cr]
            g_best = []
            for trail in range(trials):
                result = list(
                    diffevo(TestFunction.ackley, bounds=[(-35, 35)] * 8, F=F, Cr=Cr, pop_size=n, iterations=100))
                [pos_best, fitness_best] = result[-1]
                g_best.append(fitness_best)
            mean_g = np.mean(g_best)
            std_g = np.std(g_best)
            stats.extend([mean_g, std_g])
            print(stats)

print('Running PSO on 8 dimensional Ackley function')
print('pop_size, social, cognitive, inertia, mean_g_best, std_g_best:\n')
alpha_set = [0.5, 2]      # social constant
beta_set = [0.5, 2]       # cognitive constant
inertia_set = [0.5, 0.9]    # inertia constant
for n in n_set:
    for alpha in alpha_set:
        for beta in beta_set:
            for inertia in inertia_set:
                stats = [n, alpha, beta, inertia]
                g_best = []
                for trial in range(trials):
                    pso = PSO(TestFunction.ackley, 8, [-35, 35], pop_size=n, max_iter=100, alpha=alpha, beta=beta,
                              inertia=inertia, acc=False)
                    [pos_best, fitness_best] = pso.get_result()
                    g_best.append(fitness_best)
                mean_g = np.mean(g_best)
                std_g = np.std(g_best)
                stats.extend([mean_g, std_g])
                print(stats)

print('Running APSO on 8 dimensional Ackley function')
print('pop_size, social, cognitive, mean_g_best, std_g_best:\n')
alpha_set = [0.2, 0.4]           # social constant
beta_set = [0.1, 0.5, 0.7]       # cognitive constant
for n in n_set:
    for alpha in alpha_set:
        for beta in beta_set:
            stats = [n, alpha, beta]
            g_best = []
            for trial in range(trials):
                apso = PSO(TestFunction.ackley, 8, [-35, 35], pop_size=n, max_iter=100, alpha=alpha, beta=beta,
                           acc=True)
                [pos_best, fitness_best] = apso.get_result()
                g_best.append(fitness_best)
            mean_g = np.mean(g_best)
            std_g = np.std(g_best)
            stats.extend([mean_g, std_g])
            print(stats)

# plot DE with best configuration
for n in n_set:
    result = list(
        diffevo(TestFunction.ackley, bounds=[(-35, 35)] * 8, F=0.5, Cr=0.5, pop_size=n, iterations=100))
    pos_best, fitness_best = zip(*result)
    plt.plot(fitness_best, label='pop_size={}'.format(n))
plt.legend()
plt.show()

for n in n_set:
    pso = PSO(TestFunction.ackley, 8, [-35, 35], pop_size=n, max_iter=100, alpha=2, beta=2, inertia=0.5, acc=False)
    fitness_best = pso.get_result_hist()
    plt.plot(fitness_best, label='pop_size={}'.format(n))
plt.legend()
plt.show()

for n in n_set:
    apso = PSO(TestFunction.ackley, 8, [-35, 35], pop_size=n, max_iter=100, alpha=0.4, beta=0.1, acc=True)
    fitness_best = apso.get_result_hist()
    plt.plot(fitness_best, label='pop_size={}'.format(n))
plt.legend()
plt.show()