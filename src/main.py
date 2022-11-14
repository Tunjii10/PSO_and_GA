import numpy as np
from optproblems import cec2005, Individual
from psoFolder import pso
from gaFolder import ga

def pipeline():
    psoRun = pso.psoClass('sphere', 180, 50) #function, iteration, particles
    psoRun.run_algorithm()
    #ga.gaClass()


if __name__ == "__main__":
    pipeline()
    #print(np.array([2,3,4])-np.array([1,3,4]))
