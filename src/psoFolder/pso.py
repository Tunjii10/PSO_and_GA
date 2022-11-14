from optproblems import cec2005, Individual
import numpy as np

class psoClass:
    def __init__(self,benchmark_func, iterations, no_particles):
        self.dimension = 30
        self.no_particles = no_particles
        self.swarm = {"particles_position":[],
         "particles_velocity":[], 
         "particles_informants":[],
         "particle_best_position": [], 
         "particle_outcome":[0] * self.no_particles,
         "particle_best_outcome": [],
         "best_informant_position": [], 
         "best_informant_outcome": [],
         "particle_gbest_outcome": np.inf ,
         "particle_gbest_position" : [0]
         }
        self.alpha = 0.5
        self.benchmark_func = benchmark_func
        self.limit_low = 0
        self.limit_high = 0
        self.iterations = iterations
        
        self.vel_limit_low = -1
        self.vel_limit_high = 1
        self.function = None
        self.select_limit_and_functionCode()
        self.random_initialialize()
        print("***Swarm initialized****")

    def select_limit_and_functionCode(self):
        if self.benchmark_func == "sphere": #Shifted Sphere Function
            self.limit_high = 100
            self.limit_low = -100
            self.function = cec2005.F1(self.dimension)
        elif self.benchmark_func == "ackley":# Shifted Rotated Ackley’s Function
            self.limit_high = 32
            self.limit_low = -32
            self.function = cec2005.F8(self.dimension)
        elif self.benchmark_func == "rastrign":# Shifted Rastrigin’s Function 
            self.limit_high = 5
            self.limit_low = -5
            self.function = cec2005.F9(self.dimension)
        elif self.benchmark_func == "shwefel":#  Schwefel’s Problem 2.6 
            self.limit_high = 100
            self.limit_low = -100
            self.function = cec2005.F1(self.dimension)
        else:
            raise Exception(
                "Your benchmark function is not implemented")

    def random_initialialize(self):
        for x in range(self.no_particles):
            self.swarm["particles_position"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension))
            self.swarm["particles_velocity"].append(np.random.uniform(self.vel_limit_low, self.vel_limit_high, self.dimension))
            self.swarm["particles_informants"].append(np.random.randint(0,self.no_particles,10))
            self.swarm["particle_best_position"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension))
            self.swarm["best_informant_position"].append(np.random.uniform(self.limit_low, self.limit_high, self.dimension))
            self.swarm["particle_best_outcome"].append(np.inf)
            self.swarm["best_informant_outcome"].append(np.inf)
            




    def run_algorithm(self):
        for x in range(self.iterations):
            for y in range(self.no_particles):
                particle = Individual(self.swarm["particles_position"][y])
                self.function.evaluate(particle)
                self.compare_personal(y, particle.phenome, particle.objective_values)
            for y in range(self.no_particles):
                self.compare_informants(y)
            
            for y in range(self.no_particles):
                self.velocity_update(y)
                self.position_update(y)
            print("iteration {} done".format(x))
        print(self.swarm["particle_gbest_outcome"],self.swarm["particle_gbest_position"])
            
    def compare_personal(self, y, position, outcome):
        self.swarm["particle_outcome"][y] = outcome
        if outcome < self.swarm["particle_gbest_outcome"]:
            self.swarm["particle_gbest_outcome"] =  outcome 
            self.swarm["particle_gbest_position"] = position
        if outcome < self.swarm["particle_best_outcome"][y]:
            self.swarm["particle_best_outcome"][y] = outcome
            self.swarm["particle_best_position"][y] = self.swarm["particles_position"][y]
    def compare_informants(self, y):
        for value in self.swarm["particles_informants"][y]:
            if self.swarm["particle_outcome"][y]>self.swarm["particle_outcome"][value]:
                self.swarm["best_informant_outcome"][y] = self.swarm["particle_outcome"][value]
                self.swarm["best_informant_position"][y] = self.swarm["particles_position"][value]
   
    def velocity_update(self,y):
        for x in range(self.dimension):
            b = np.random.uniform(0,1)
            c = np.random.uniform(0,1)
            cognitive_component = b * (np.array(self.swarm["particle_best_position"][y][x])-np.array(self.swarm["particles_position"][y][x]))
            social_component = c * (np.array(self.swarm["best_informant_position"][y][x])-np.array(self.swarm["particles_position"][y][x]))
            self.swarm["particles_velocity"][y][x] = self.alpha * self.swarm["particles_velocity"][y][x]+ cognitive_component + social_component
    def position_update(self,y):
        for x in range(self.dimension):
            new_position =  self.swarm["particles_position"][y][x] + self.swarm["particles_velocity"][y][x]

            if new_position > self.limit_high or new_position < self.limit_low:
                #do nothing
                return
            else:
                self.swarm["particles_position"][y][x] = new_position
           
    # optimal = func.get_optimal_solutions()
    # for opt in optimal:
    #     print(opt.phenome, opt.objective_values)
    # solution = Individual([3.6267, -82.9123, -12.6423, -0.5815, 83.1552])
    # solution2 = Individual([3.006267, -2.9123, -12.642323, -0.5815, 83.1552])
    # func.batch_evaluate([solution, solution2])
    # print(solution.phenome, solution.objective_values)
    # #result = func.evaluate(solution)
    # print(solution2.objective_values, solution.objective_values)