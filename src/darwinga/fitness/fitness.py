# coding: utf-8

import numpy as np


class Fitness:
    fitness_matrix = None

    @classmethod
    def calculate(cls, order):
        fitness = 0
        points = len(order)

        for i in range(points-1):
            m = int(order[i])
            n = int(order[i+1])
            fitness += cls.fitness_matrix[m][n]

        m = int(order[points-1])
        n = int(order[0])
        fitness += cls.fitness_matrix[m][n]
        return fitness

    @classmethod
    def initialize_matrix(cls, data):
        v, w = np.split(data, [-1], axis=1)
        X = np.column_stack(np.repeat(v, len(v),  axis=1)).T
        Y = np.column_stack(np.repeat(w, len(w),  axis=1)).T
        cls.fitness_matrix = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
