# coding: utf-8

import random
import numpy as np

class Crossover:

    @staticmethod
    def double_point(genes_1, genes_2):
        gene_size = len(genes_1)

        # Declaring new_genes as numpy arrays
        new_genes_1 = -np.ones(gene_size)
        new_genes_2 = -np.ones(gene_size)

        # Calculating split point
        split_1 = split_2 = 0
        while split_1 == split_2:
            split_1 = random.randint(0,gene_size)
            split_2 = random.randint(0,gene_size)
        
        # Inserting elements
        if split_2 < split_1:
            temp = split_2
            split_2 = split_1
            split_1 = temp

        # Inserting Middle
        new_genes_1[split_1:split_2] = genes_2[split_1:split_2]
        new_genes_2[split_1:split_2] = genes_1[split_1:split_2]

        # Inserting Front
        for i in range(split_1):
            for j in range(gene_size):
                if genes_1[j] not in new_genes_1:
                    new_genes_1[i] = genes_1[j]
                    break
            for j in range(gene_size):
                if genes_2[j] not in new_genes_2:
                    new_genes_2[i] = genes_2[j]
                    break

        # Inserting Back
        for i in range(split_2, gene_size):
            for j in range(gene_size):
                if genes_1[j] not in new_genes_1:
                    new_genes_1[i] = genes_1[j]
                    break
            for j in range(gene_size):
                if genes_2[j] not in new_genes_2:
                    new_genes_2[i] = genes_2[j]
                    break
        return new_genes_1, new_genes_2
