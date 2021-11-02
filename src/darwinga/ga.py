# coding: utf-8

from abc import ABC
import random
from types import ModuleType

import numpy as np


class GeneticAlgorithmBase(ABC):

    def __init__(self, fitness: ModuleType, chromosome: ModuleType, mutation: ModuleType, selection: ModuleType):
        self.fitness = fitness
        self.chromosome = chromosome
        self.mutation = mutation
        self.selection = selection
    
    def __eq__(self, o: object) -> bool:
        return self.chromosome == o.chromosome
    
    def __bool__(self):
        return bool(self.chromosome) and bool(self.fitness) and bool(self.mutation) and bool(self.selection)
    
    def __repr__(self):
        return f'GeneticAlgorithm {self.__class__.__name__}'
    
    def __str__(self):
        return self.__repr__()
    
    def _module_validation_check(self, mutation_mechanism: str = None, selection_mechanism: str = None):
        if not hasattr(self.chromosome, 'genes'):
            raise AttributeError('Chromosome Module needs to have genes as the data array.')

        if not hasattr(self.selection, selection_mechanism) or not callable(getattr(self.selection, 'selection_mechanism')):
            raise AttributeError(f'Selection Module does not include {selection_mechanism} as a function.')
        
        if not hasattr(self.mutation, mutation_mechanism) or not callable(getattr(self.mutation, 'mutation_mechanism')):
            raise AttributeError(f'Mutation Module does not include {mutation_mechanism} as a function.')
    
    def run_simple(self, population_size: int, generations: int, mutation_probability: float, selection_mechanism: str, mutation_mechanism: str, *args, **kwargs) -> np.array:
        self._module_validation_check(mutation_mechanism, selection_mechanism)

        # Declaring mechanisms
        selection = getattr(self.selection, selection_mechanism)
        mutation = getattr(self.mutation, mutation_mechanism)

        # Declare population
        population = [self.chromosome() for i in range(population_size)]

        # Storing Best Generation Data
        max_per_generation = np.zeros(generations)

        # Starting the iterations
        for generation in range(generations):
            population_2 = self.selection.roulette(population_1)

            # Mutation
            for chromosome in population_2:
                if random.random() <= mutation_probability:
                    mutation(chromosome.genes)
                    chromosome.set_fitness()

            # Sorting population_3 to grab top 50%
            combined_population = sorted(population_1 + population_2)
            
            # Setting population_1 to copied population_3
            population_1 = copy.deepcopy(combined_population[:population_size])

            # Storing Best Fitness for that Generation
            max_per_generation[generation] = population_1[0].fitness
        
        # Returning Progress
        return max_per_generation
