import numpy as np

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])*2 + (point1[1] - point2[1])*2)

# Função para calcular o comprimento total da rota
def calculate_route_distance(route, points):
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(points[route[i]], points[route[i + 1]])
    return distance

# Função de fitness
def fitness_function(population, points):
    fitness = np.zeros(len(population))
    for i, individual in enumerate(population):
        fitness[i] = 1 / calculate_route_distance(individual, points)  # Inverso da distância
    return fitness

# Seleção por roleta
def select_parents(population, fitness, num_parents):
    fitness_sum = np.sum(fitness)
    probabilities = fitness / fitness_sum
    selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities)
    return population[selected_indices]

# Crossover por ordem (OX)
def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer >= len(parent1):
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        offspring.append(child)
    return np.array(offspring)

# Mutação
def mutation(offspring):
    for individual in offspring:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]  # Troca dois genes
    return offspring

# Inicialização da população
def initialize_population(num_individuals, num_points):
    return np.array([np.random.permutation(num_points) for _ in range(num_individuals)])

# Configurações do algoritmo
points = [(0, 0), (2, 3), (5, 1), (6, 4), (8, 0)]  # Coordenadas dos pontos
num_generations = 100
population_size = 20
num_parents_mating = 10

# Iniciar a população
population = initialize_population(population_size, len(points))

# Evolução
for generation in range(num_generations):
    fitness = fitness_function(population, points)
    parents = select_parents(population, fitness, num_parents_mating)
    offspring_size = population_size - len(parents)
    offspring_crossover = crossover(parents, offspring_size)
    offspring_mutation = mutation(offspring_crossover)
    population[:len(parents)] = parents
    population[len(parents):] = offspring_mutation

    # Melhor solução da geração
    best_fitness = np.max(fitness)
    best_individual = population[np.argmax(fitness)]
    print(f"Generation {generation + 1}: Best Distance = {1 / best_fitness:.2f}")

# Melhor solução final
final_best_individual = population[np.argmax(fitness)]
final_best_distance = 1 / np.max(fitness)
print("Best Route:", final_best_individual)
print("Best Distance:", final_best_distance)
