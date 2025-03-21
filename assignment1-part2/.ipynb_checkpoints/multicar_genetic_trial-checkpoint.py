from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import sys
from multicar_genetic_algorithms_function import calculate_fitness, \
    select_in_tournament, order_crossover, mutate, \
    generate_unique_population

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Process {rank} started", flush=True)

# Start timing (only rank 0 tracks time)
start_time = time.time() if rank == 0 else None  

# Load the distance matrix (root process loads it, then broadcasts)
if rank == 0:
    distance_matrix = pd.read_csv("city_distances_extended.csv").to_numpy()
else:
    distance_matrix = None

distance_matrix = comm.bcast(distance_matrix, root=0)  # Broadcast to all processes

# Parameters
num_nodes = distance_matrix.shape[0]
num_vehicles = 3  # Number of vehicles
population_size = 1000
num_generations = 200
mutation_rate = 0.1
stagnation_limit = 10  

# Distribute population across MPI processes
local_population_size = max(1, population_size // size)  # Ensure at least 1 individual per rank
local_population = generate_unique_population(local_population_size, num_nodes, num_vehicles)

# Initialize tracking variables
best_fitness = float('inf')
stagnation_counter = 0

# Main GA loop
for generation in range(num_generations):
    print(f"Process {rank}: Starting generation {generation}", flush=True)

    # Calculate fitness for local population
    local_fitness = np.array([calculate_fitness(routes, distance_matrix) for routes in local_population])

    # Gather all fitness values at root
    all_fitness = comm.gather(local_fitness, root=0)

    if rank == 0:
        all_fitness = np.concatenate(all_fitness)  
        current_best_fitness = np.min(all_fitness)
        print(f"Generation {generation}: Best fitness = {current_best_fitness}", flush=True)
    else:
        current_best_fitness = None  

    # Broadcast best fitness to all ranks
    current_best_fitness = comm.bcast(current_best_fitness, root=0)

    # Handle stagnation
    if rank == 0:
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_limit:
            print(f" Stagnation detected at generation {generation}. Increasing mutation rate.", flush=True)
            mutation_rate = min(0.6, mutation_rate * 1.2)  
            stagnation_counter = 0  

    # Broadcast updated mutation rate to all ranks
    mutation_rate = comm.bcast(mutation_rate, root=0)

    # Selection, crossover, and mutation (done locally)
    tournament_size = min(8, len(local_population))  
    local_selected = select_in_tournament(local_population, local_fitness, tournament_size=tournament_size)
    local_offspring = []
    for i in range(0, len(local_selected), 2):
        if i + 1 >= len(local_selected):  
            continue
        parent1, parent2 = local_selected[i], local_selected[i + 1]
        route1 = order_crossover(parent1, parent2, num_vehicles)
        local_offspring.append(route1)

    local_offspring = [mutate(route, mutation_rate) for route in local_offspring]

    # Gather new offspring at root
    all_offspring = comm.gather(local_offspring, root=0)

    if rank == 0:
        all_offspring = [item for sublist in all_offspring for item in sublist]  
        local_population = all_offspring[:population_size]  

    # Broadcast new population to all ranks
    local_population = comm.bcast(local_population, root=0)

# Stop timing
if rank == 0:
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"⏳ Execution Time: {execution_time:.5f} sec", flush=True)

# Gather entire population at rank 0 to find the best solution
all_population = comm.gather(local_population, root=0)

if rank == 0:
    all_population = [item for sublist in all_population for item in sublist]  
    best_idx = np.argmin(all_fitness)
    best_solution = all_population[best_idx]

    print(" Best Solution:", best_solution, flush=True)
    print(" Total Distance:", calculate_fitness(best_solution, distance_matrix), flush=True)
