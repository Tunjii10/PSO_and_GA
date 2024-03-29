# Particle Swarm Algorithm and Genetic Algorithm
This is an implementation and testing of a particle swarm algorithm and a genetic algorithm from scratch. Both algorithms are tested against five benchmark functions from the CEC 2005 benchmark functions suite. Variations from normal implementation of both algorithms are explained in the comments in the code.

## Environment
**=>** Note : This project was done using python3 in a python venv environment. The OS used was WSL(Windows Subsystem for linux).

## Installation
**1.** Run 
```
pip install -r requirements.txt
```
to install. Note you must be in the root directory.

## How To Use

**1.** Navigate to the src directory

```
cd src/
```

**2.** Run the main.py file by typing following command

```
python3 main.py
```

**3.** The main.py file runs the pso and ga algorithm for 5 different selected benchmark functions and prints the best outcome and position. 

**4.** For experimentation you can change the variables as needed. Note it will only work for the five implemented benchmark functions and be wary of inputs as some inputs wont work, for example picking tournament size that is higher than the population.

