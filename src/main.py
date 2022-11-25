from psoFolder import pso
from gaFolder import ga

def pipeline():
    #sphere function
    psoRun = pso.psoClass('sphere', 200, 60, 10, 10) # variables = benchmark_function, iteration, no_particles, dimension, informants_no
    psoRun.run_algorithm()
    gaRun = ga.gaClass('sphere', 200, 60, 10, 10, 3, 80, 32) # variables = benchmark_func, generations, no_population, dimension, tournament_size, no_mutation, crossover_rate, mutation_rate
    gaRun.run_algorithm()

    print('\n*******************\n')

    # #schwefel function
    psoRun = pso.psoClass('schwefel', 250, 60, 10, 10) # variables = benchmark_function, iteration, no_particles, dimension, informants_no
    psoRun.run_algorithm()
    gaRun = ga.gaClass('schwefel', 250, 60, 10, 12, 2, 80, 30) # variables = benchmark_func, generations, no_population, dimension, tournament_size, no_mutation, crossover_rate, mutation_rate
    gaRun.run_algorithm()

    print('\n*******************\n')
 
    # #rosenbrock function
    psoRun = pso.psoClass('rosenbrock', 500, 80, 10, 15) # variables = benchmark_function, iteration, no_particles, dimension, informants_no
    psoRun.run_algorithm()
    gaRun = ga.gaClass('rosenbrock', 500, 80, 10, 30, 3, 80, 60) # variables = benchmark_func, generations, no_population, dimension, tournament_size, no_mutation, crossover_rate, mutation_rate
    gaRun.run_algorithm()

    
    print('\n*******************\n')
 
    #rastrign function
    psoRun = pso.psoClass('rastrign', 400, 100, 10, 15) # variables = benchmark_function, iteration, no_particles, dimension, informants_no
    psoRun.run_algorithm()
    gaRun = ga.gaClass('rastrign', 400 , 100, 10, 8, 1, 60, 45) # variables = benchmark_func, generations, no_population, dimension, tournament_size, no_mutation, crossover_rate, mutation_rate
    gaRun.run_algorithm()
    
    print('\n*******************\n')
 
    # #weierstrass function
    psoRun = pso.psoClass('weierstrass', 250, 80, 10, 10) # variables = benchmark_function, iteration, no_particles, dimension, informants_no
    psoRun.run_algorithm()
    gaRun = ga.gaClass('weierstrass', 250, 80, 10, 10, 2, 80, 72) # variables = benchmark_func, generations, no_population, dimension, tournament_size, no_mutation, crossover_rate, mutation_rate
    gaRun.run_algorithm()


if __name__ == "__main__":
    pipeline() #calls ga and pso class for experimentation
   
