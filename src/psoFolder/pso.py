#import libraries
import numpy as np
import random
from optproblems import cec2005, Individual



class psoClass:
    # initialization of psoClass based on variables inputtted
    def __init__(self,benchmark_func, iterations, no_particles, dimension, informants_no):
        self.dimension = dimension
        self.no_particles = no_particles
        # psoClass object that holds the  particles position, paritcle informants, particles velocity, particles fitness outcome, best position and outcome for particles,informants and global best position and outcome
        self.swarm = {"particles_position":[],
         "particles_velocity":[], 
         "particles_informants":[],
         "particle_best_position": [], 
         "particle_outcome":[0] * self.no_particles,
         "particle_best_outcome": [],
         "best_informant_position": [], 
         "best_informant_outcome": [],
         "particle_gbest_outcome": np.inf ,
         "particle_gbest_position" : []
         }
        self.alpha = 0.5 #inertia weight
        self.benchmark_func = benchmark_func
        self.limit_low = 0 #function bounds low
        self.limit_high = 0 #function bounds high
        self.iterations = iterations
        
        self.vel_limit_low = -1 #velocity limit low (for initialization)
        self.vel_limit_high = 1 #velocity limit high (for initialization)
        self.informants_no = informants_no #number of informants
        self.function = None #variable for storing selected instance of cec2005 benchmark function
        self.select_limit_and_functionCode() # function to select benchmark function
        self.random_initialialize() # function to initialize particles randomly
        print("***Swarm initialized****")

    def select_limit_and_functionCode(self):
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
        # initialize particles and velcoity within bound
        for x in range(self.no_particles): #iterate through no of particles
            self.swarm["particles_position"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension)) # particle postition
            self.swarm["particles_velocity"].append(np.random.uniform(self.vel_limit_low, self.vel_limit_high, self.dimension)) #particle velocity
            self.swarm["particles_informants"].append(random.sample(range(0,self.no_particles),self.informants_no)) # randomly select particle positions in population for informants
            self.swarm["particle_best_position"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension)) #initialize randomly particle best position
            self.swarm["best_informant_position"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension)) #initialize randomly particle informant best position
            self.swarm["particle_best_outcome"].append(np.inf) #set particle best outcome to infinity so that in first iteration it will be changed
            self.swarm["best_informant_outcome"].append(np.inf) #set particle best informant outcome to infinity so that in first iteration it will be changed
            

    def run_algorithm(self):
        #function to run algorithm process
        for x in range(self.iterations): #iterate through number of iterations
            for y in range(self.no_particles): #iterate through each particle
                particle = Individual(self.swarm["particles_position"][y])  #intilialize particle as instance of individual class for Cec2005 library
                self.function.evaluate(particle) #evaluate particle
                self.compare_personal(y, particle.phenome, particle.objective_values) #compare outcome to personal best and global best
            for y in range(self.no_particles): #iterate through each particle
                #compare informants for each particle
                self.compare_informants(y)
            
            for y in range(self.no_particles): #iterate through each particle
                # update velocity and position
                self.velocity_update(y)
                self.position_update(y)
        print("{} benchmark function best outcome and position is {} and {} respectively".format(self.benchmark_func,self.swarm["particle_gbest_outcome"],self.swarm["particle_gbest_position"]))
            
    def compare_personal(self, y, position, outcome):
        #function to compare outcome to particle best and global best
        self.swarm["particle_outcome"][y] = outcome #store outcome in swarm object
        #if outcome less than global best , make global best and update global best position
        if outcome < self.swarm["particle_gbest_outcome"]:
            self.swarm["particle_gbest_outcome"] =  outcome 
            self.swarm["particle_gbest_position"] = position
        #if outcome less than personal best , make personal best and update personal best position
        if outcome < self.swarm["particle_best_outcome"][y]:
            self.swarm["particle_best_outcome"][y] = outcome
            self.swarm["particle_best_position"][y] = self.swarm["particles_position"][y]
    def compare_informants(self, y):
        # function to compare informants
        best_informant_outcome = np.inf #set best informant value as infinity
        best_informant_position = 0 #initialize best informant position
        for position in self.swarm["particles_informants"][y]: # loop through particles informants
            if self.swarm["particle_outcome"][position] < best_informant_outcome: #if informant value less than the best value set as best 
                best_informant_position = position
                best_informant_outcome = self.swarm["particle_outcome"][position]
        self.swarm["best_informant_outcome"][y] = best_informant_outcome #set informant outcome 
        self.swarm["best_informant_position"][y] = self.swarm["particles_position"][best_informant_position] #set informant position
   
    def velocity_update(self,y):
        # velocity update function
        for x in range(self.dimension): #iterate through particles dimensions to update individual velocity component of particle
            b = np.random.uniform(0,1) #randomly select cognitive weight
            c = np.random.uniform(0,1) #randomly select social weight
            cognitive_component = b * (np.array(self.swarm["particle_best_position"][y][x])-np.array(self.swarm["particles_position"][y][x])) #compute cognitive component
            social_component = c * (np.array(self.swarm["best_informant_position"][y][x])-np.array(self.swarm["particles_position"][y][x])) #compute social component
            self.swarm["particles_velocity"][y][x] = self.alpha * self.swarm["particles_velocity"][y][x]+ cognitive_component + social_component #update velocity compoonent
    def position_update(self,y):
        #position update function
        for x in range(self.dimension): #iterate through particles dimensions to update individual position component of particle
            new_position =  self.swarm["particles_position"][y][x] + self.swarm["particles_velocity"][y][x] #compute new position

            if new_position > self.limit_high or new_position < self.limit_low: #if position out of bounds do nothing
                #do nothing
                pass
            else: #if position within bounds update
                self.swarm["particles_position"][y][x] = new_position
   