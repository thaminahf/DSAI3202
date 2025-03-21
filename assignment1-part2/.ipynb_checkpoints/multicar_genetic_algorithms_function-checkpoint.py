import numpy as np

def calculate_fitness(routes, distance_matrix):
    """
    Calculates the total distance of the routes while applying penalties for unreachable nodes.
    """
    total_distance = 0
    for vehicle_route in routes:
        if len(vehicle_route) < 2:
            continue  # Skip empty routes

        for i in range(len(vehicle_route) - 1):
            node1, node2 = vehicle_route[i], vehicle_route[i + 1]
            distance = distance_matrix[node1][node2]

            if distance == 100000:  # Large penalty for invalid routes
                print(f" Large penalty applied: {node1} → {node2}")
                total_distance += np.median(distance_matrix[distance_matrix < 100000]) * 5
            else:
                total_distance += distance

    return -total_distance  # Return negative for GA optimization



def generate_unique_population(population_size, num_nodes, num_vehicles):
    """
    Generate a unique population where cities are evenly assigned across vehicles.
    """
    population = []
    cities = list(range(1, num_nodes))  # Exclude depot (0)

    for _ in range(population_size):
        np.random.shuffle(cities)  # Shuffle cities for randomness
        split_routes = np.array_split(cities, num_vehicles)
        individual = [[0] + list(route) for route in split_routes]
        population.append(individual)

    return population

def select_in_tournament(population, scores, tournament_size=8):
    """
    Tournament selection: Selects the best individuals from small random groups.
    """
    selected = []
    for _ in range(len(population)):
        idx = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = idx[np.argmin(scores[idx])]  
        selected.append(population[best_idx])
    return selected

def order_crossover(parent1, parent2, num_vehicles):
    """
    Performs order crossover (OX) for multiple vehicle routes while ensuring valid routes.
    """
    offspring = []
    for v in range(num_vehicles):
        size = len(parent1[v])
        if size < 2:  # Skip empty routes
            offspring.append(parent1[v])
            continue

        # Select a random segment for crossover
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        route = [-1] * size
        route[start:end] = parent1[v][start:end]

        # Fill in missing values while ensuring no duplicates
        existing_nodes = set(route)
        remaining_values = [x for x in parent2[v] if x not in existing_nodes and x != -1]

        idx = 0
        for i in range(size):
            if route[i] == -1:
                if idx < len(remaining_values):
                    route[i] = remaining_values[idx]
                    idx += 1
                else:
                    # Fill in missing nodes
                    missing_nodes = list(set(range(size)) - set(route))
                    np.random.shuffle(missing_nodes)
                    route[i] = missing_nodes.pop()

        offspring.append(route)

    return offspring


def mutate(route, mutation_rate=0.2):
    """
    Mutation operator: swaps two cities in a route while ensuring no duplicates.
    """
    if np.random.rand() < mutation_rate and len(route) > 2:
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]  # Swap instead of replacing

    return route


