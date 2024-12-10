import numpy as np

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
        start, end = sorted(np.random.choice(len(parent1) - 2, 2, replace=False) + 1)  # Evita fixos
        child = [-1] * len(parent1)
        child[0], child[-1] = parent1[0], parent1[-1]  # Preserva entrada e saída
        child[start:end] = parent1[start:end]
        pointer = end
        for gene in parent2[1:-1]:  # Evita entrada e saída
            if gene not in child:
                if pointer >= len(parent1) - 1:
                    pointer = 1  # Recomeça depois do fixo inicial
                child[pointer] = gene
                pointer += 1
        offspring.append(child)
    return np.array(offspring)

# Mutação
def mutation(offspring):
    for individual in offspring:
        idx1, idx2 = np.random.choice(range(1, len(individual) - 1), 2, replace=False)  # Evita fixos
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]  # Troca dois genes
    return offspring

# Inicialização da população
def initialize_population(num_individuals, num_points):
    population = []
    for _ in range(num_individuals):
        route = list(range(1, num_points - 1))
        np.random.shuffle(route)
        population.append([0] + route + [num_points - 1])  # Inclui entrada e saída fixas
    return np.array(population)

# Configurações do algoritmo
points = [
    (0, 0),    # Entrada
    (50, 150), # Eletrodomésticos
    (100, 150),# Utilidades
    (200, 130),# Fliperama
    (310, 130),# Esportes
    (440, 180),# Alimentação
    (260, 280),# Brinquedos
    (70, 320), # Roupas
    (120, 440),# Calçados
    (360, 440),# Perfumes
    (370, 0)   # Saída
]

num_generations = 10000
population_size = 20
num_parents_mating = 10

# Para salvar as populações
initial_population = None
mid_generation_population = None
final_population = None

# Iniciar a população
population = initialize_population(population_size, len(points))
initial_population = population.copy()

# Evolução
for generation in range(num_generations):
    fitness = fitness_function(population, points)
    parents = select_parents(population, fitness, num_parents_mating)
    offspring_size = population_size - len(parents)
    offspring_crossover = crossover(parents, offspring_size)
    offspring_mutation = mutation(offspring_crossover)
    population[:len(parents)] = parents
    population[len(parents):] = offspring_mutation

    # Salvar população intermediária na metade das gerações
    if generation == num_generations // 2:
        mid_generation_population = population.copy()

    # Melhor solução da geração
    best_fitness = np.max(fitness)
    best_individual = population[np.argmax(fitness)]
    best_distance = 1 / best_fitness

    print(f"Generation {generation + 1}: Best Distance = {best_distance:.2f}")

# Salvar a população final
final_population = population.copy()

# Função para calcular a melhor rota e distância em uma população
def get_best_route_and_distance(population, points):
    best_route = None
    best_distance = float('inf')
    for individual in population:
        distance = calculate_route_distance(individual, points)
        if distance < best_distance:
            best_distance = distance
            best_route = individual
    return best_route, best_distance

# Resultados para as populações
initial_best_route, initial_best_distance = get_best_route_and_distance(initial_population, points)
mid_best_route, mid_best_distance = get_best_route_and_distance(mid_generation_population, points)
final_best_route, final_best_distance = get_best_route_and_distance(final_population, points)

# Exibir resultados
print("\n--- Resultados ---")
print("População Inicial:")
print("Melhor Rota:", initial_best_route)
print("Melhor Distância:", initial_best_distance)

print("\nPopulação do Meio:")
print("Melhor Rota:", mid_best_route)
print("Melhor Distância:", mid_best_distance)

print("\nPopulação Final:")
print("Melhor Rota:", final_best_route)
print("Melhor Distância:", final_best_distance)
