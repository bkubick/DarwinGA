# coding: utf-8


class Chromosome:
    FitnessModule = None

    def __init__(self, genes):
        self.genes = genes
        self.set_fitness()
    
    def __lt__(self, other):
        return other.fitness > self.fitness
    
    def __gt__(self, other):
        return other.fitness < self.fitness

    def __eq__(self, other):
        return other.fitness == self.fitness
    
    def set_genes(self, genes):
        self.genes = genes
        self.set_fitness()
    
    def set_fitness(self):
        if not self.FitnessModule:
            raise ValueError('Fitness module not declared')
        self.fitness = self.FitnessModule.calculate(self.genes)
