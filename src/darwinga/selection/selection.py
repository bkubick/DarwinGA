# coding: utf-8

import random
import numpy as np

from src.models.chromosome import Chromosome
from src.models.crossover import Crossover

class Selection:

    @staticmethod
    def roulette(population):
        # Fitnesses
        population_fitnesses = np.array([population[i].fitness for i in range(len(population))])
        total_fitness = np.sum(population_fitnesses)

        # Looking for the minimum value so a smaller fitness is a higher probability
        probabilities = 1 - np.array([population_fitness/total_fitness for population_fitness in population_fitnesses])

        # Calculating the cumulative Sum
        cumulative_sum = np.cumsum(probabilities)

        # Normalizing cumulative_sum
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]

        offspring = []
        # Grabbing indices of parents based on random_num falling inbetween bins
        for i in range(int(len(population)/2)):
            parent_1 = parent_2 = 0
            while parent_1 == parent_2:
                parent_1 = np.digitize(random.random(), normalized_cumulative_sum, right=True)
                parent_2 = np.digitize(random.random(), normalized_cumulative_sum, right=True)

            genes_1, genes_2 = Crossover.double_point(population[parent_1].genes, population[parent_2].genes)
            offspring.extend([Chromosome(genes_1), Chromosome(genes_2)])

        return offspring
