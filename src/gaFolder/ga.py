#References
# line 115 - Crossover Point . Available at :https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

#import libraries
import numpy as np
import random
from optproblems import cec2005, Individual



class gaClass:
    #initialization of gaClass based on variables inputtted
    def __init__(self, benchmark_func, generations, no_population, dimension, tournament_size, no_mutation, crossover_rate, mutation_rate):
        self.dimension = dimension
        self.no_population = no_population
        # gaClass object that holds the population, population outcome and bestoutcome
        self.population_set = {"population":[], "population_outcome":[], "best_outcome": np.inf, "best_outcome_population":[]}
        self.benchmark_func = benchmark_func
        self.limit_low = 0 #function bounds low
        self.limit_high = 0 #function bounds high
        self.generations = generations
        self.tournament_size = tournament_size
        self.no_mutation = no_mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.function = None #variable for storing selected instance of cec2005 benchmark function
        self.new_population = [] #stores mutated and crossovered population
        self.select_limit_and_functionCode() # function to select benchmark function
        self.random_initialialize() # function to initialize chromosomes randomly
        print("***GA initialized****")

    def select_limit_and_functionCode(self):
        # select benchmark functions and their limits
        if self.benchmark_func == "sphere": #Shifted Sphere Function
            self.limit_high = 100
            self.limit_low = -100
            self.function = cec2005.F1(self.dimension)
        elif self.benchmark_func == "schwefel":# Shifted Schwefel’s Problem 1.2
            self.limit_high = 100
            self.limit_low = -100
            self.function = cec2005.F2(self.dimension)
        elif self.benchmark_func == "rosenbrock":# Shifted Rosenbrock’s Function 
            self.limit_high = 100
            self.limit_low = -100
            self.function = cec2005.F6(self.dimension)
        elif self.benchmark_func == "rastrign":# Shifted Rastrigin’s Function 
            self.limit_high = 5
            self.limit_low = -5
            self.function = cec2005.F9(self.dimension)
        elif self.benchmark_func == "weierstrass":#  Shifted Rotated Weierstrass Function 
            self.limit_high = 0.5
            self.limit_low = -0.5
            self.function = cec2005.F11(self.dimension)
        else:
            raise Exception(
                "Your benchmark function is not implemented")

    def random_initialialize(self):
        #initialize chromosomes within bound
        for x in range(self.no_population): #iterate through no of chromosomes
            self.population_set["population"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension))
       

    def run_algorithm(self):
        #function to run algorithm process
        for x in range(self.generations): #iterate through number of generations
            for y in range(self.no_population): #iterate through each chromosone
                individual_population = Individual(self.population_set["population"][y]) #intilialize chromosone as instance of individual class for Cec2005 library
                self.function.evaluate(individual_population) #evaluate chromosome
                self.population_set["population_outcome"].append(individual_population.objective_values) #append its outcome to outcome array
                self.compare_best( individual_population.phenome,individual_population.objective_values) #compare with best of experiment
           
            for y in range(0,self.no_population,2): #iterate over population in steps of two
                if np.random.randint(0,100)<= self.crossover_rate: #if random no below cross over rate, crossover
                    #select two parents for crossover using tournament selection
                    parent1 = self.selection() # call tournament selection function
                    parent2 = self.selection() # call tournament selection function
                    child1, child2 = self.crossover(self.population_set["population"][parent1],self.population_set["population"][parent2]) #carry out one point crossover between the two parents to reproduce two children
                else:
                    # if no crossover just select children as identical copies of parents
                    child1 = self.population_set["population"][y]
                    child2 = self.population_set["population"][y+1]

                for z in [child1,child2]: 
                    if np.random.randint(0,100) <= self.mutation_rate: #if random no below mutation rate, mutate
                        self.new_population.append(self.mutation(z)) #mutate each child and append to new_population list
                    else:
                        #if no mutation append child directly to new_population list
                        self.new_population.append(z) 
            self.population_set["population"] = self.new_population #set population as the newly formed population
            self.new_population = [] #reinitialize new_population list for next generation
        print("{} benchmark function best outcome and position is {} and {} respectively".format(self.benchmark_func,self.population_set["best_outcome"],self.population_set["best_outcome_population"]))

    def compare_best(self, phenome, outcome):
        #function to compare chromosomes outcome and set global best
        if outcome < self.population_set["best_outcome"]: #if outcome lest than best make global best
            self.population_set["best_outcome"] =  outcome 
            self.population_set["best_outcome_population"] =  phenome
    
    def selection(self): 
        #selection function using tournament selection
        random_individuals = np.random.randint(0, self.no_population, self.tournament_size) #pick random individual index from population according to tournament size
        selected = 0
        objective_value = np.inf #set fitness/objective value as infinity
        for value in random_individuals: # loop through the random index's and select one with best fitness function, which in this case is lowest outcome
            if self.population_set["population_outcome"][value] < objective_value:
                selected = value
                objective_value = self.population_set["population_outcome"][value]
            return selected #return most fit candidate
    
    def crossover(self, parent1,parent2):
        # one point crossover function
        # Available at : https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/. Line 115
        crossover_point = np.random.randint(1,self.dimension-2) #randomly select crossover point
        # crossver between two parents at randomly selected points to form two children
        child1 = np.concatenate((parent1[0:crossover_point], parent2[crossover_point:self.dimension]), axis=0)
        child2 = np.concatenate((parent2[0:crossover_point], parent1[crossover_point:self.dimension]), axis=0)
        return child1.tolist(), child2.tolist()


    def mutation(self, child):
        #mutation function
        #select mutation points according to number specified
        mutation_point = random.sample(range(0,self.dimension),self.no_mutation)# distinct points within set dimension
        for x in mutation_point:
            # select random number within function bounds and substitute it at mutation point in the child
            random_substitute = np.random.uniform(self.limit_low, self.limit_high)
            child[x] = random_substitute
        return child #return mutated child