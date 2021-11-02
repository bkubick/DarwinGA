# coding: utf-8

import random

from typing import List


class Mutation:

    @classmethod
    def reverse_gene_subset_random_size(cls, genes: List):
        """ Reverses a subset of random size
            args:
                genes (List): list of the genes
        """
        # Grabbing two random indices
        subset_indices = cls._select_k_unique_indices(2, max_index=len(genes)-1)

        # Sorting two randomly selected indices
        subset_indices.sort()

        # Reversing the order of the subset
        genes[subset_indices[0]:subset_indices[1]+1] = genes[subset_indices[1]:subset_indices[0]-1:-1]

    @classmethod
    def reverse_gene_subset_exact_size(cls, genes: List, subset_size: int = 2):
        """ Reverses a subset of size subset_size
            args:
                genes (List): list of the genes
                
                subset_size (int): length of the subset to be shuffled
        """
        # Declare the min and max index for the subset inside genes
        subset_min_index = random.randint(0, len(genes)-subset_size)
        subset_max_index = subset_min_index + subset_size - 1

        # Reversing the order of the subset
        genes[subset_min_index:subset_max_index + 1] = genes[subset_max_index:subset_min_index-1:-1]

    @classmethod
    def shuffle_random_gene_subset(cls, genes: List, subset_size: int = 2):
        """ Shuffles k randomly selected genes from an inline subset of a list.
            args:
                genes (List): list of the genes
                
                subset_size (int): length of the subset to be shuffled
        """
        # Declare the min and max index for the subset inside genes
        subset_min_index = random.randint(0, len(genes)-subset_size)
        subset_max_index = subset_min_index + subset_size - 1

        random_gene_indices = cls._select_k_unique_indices(subset_size, subset_max_index, subset_min_index)
        cls._shuffle_or_shift(genes, random_gene_indices)
    
    @classmethod
    def shift_k_random_genes(cls, genes: List, k: int = 2):
        """ Shifts k randomly selected genes from a list.
            args:
                genes (List): list of the genes
                
                k (int): number of genes to be switched

                reverse (bool): direction genes are shifted
        """
        # Initialize k random genes
        random_gene_indices = cls._select_k_unique_indices(k=k, max_index=len(genes)-1)
        random_gene_indices.sort()

        cls._shuffle_or_shift(genes, random_gene_indices)

    @classmethod
    def shuffle_k_random_genes(cls, genes: List, k: int = 2):
        """ Shuffles k randomly selected genes from a list and returns the new list.
            args:
                genes (List): list of the genes
                
                k (int): number of genes to be switched
        """
        # Initialize k random genes
        random_gene_indices = cls._select_k_unique_indices(k=k, max_index=len(genes)-1)

        cls._shuffle_or_shift(genes, random_gene_indices)
    
    @classmethod
    def _shuffle_or_shift(cls, genes: List, random_gene_indices: List):
        """ Shuffles k randomly selected genes from a list and returns the new list.
            args:
                genes (List): list of the genes
                
                k (int): number of genes to be switched
        """
        # Save last random_gene index gene value
        last_gene = genes[random_gene_indices[len(random_gene_indices)-1]]

        # Set random_index[i] gene to random_index[i-1] gene 
        for i in range(len(random_gene_indices)-1, 0, -1):
            genes[random_gene_indices[i]] = genes[random_gene_indices[i-1]]

        # Set first random_gene index gene value to last_gene value
        genes[random_gene_indices[0]] = last_gene

    @classmethod
    def _select_k_unique_indices(cls, k: int, max_index: int = 9, min_index: int = 0,) -> List:
        """ Selectes k unique gene indices between the min and max numbers declared
            args:
                k (int): number of unique genes to be selected

                max (int): max index (typically len(list)-1 to be selected from)

                min (int): min index from list to be selected from
            
            returns:
                list of k unique randomly generated indices
        """
        # Initialize set of genes for uniqueness
        gene_indices = set()

        # Add random int between min and max value to genes
        while len(gene_indices) < k:
            gene_indices.add(random.randint(min_index, max_index))

        # Return random set of genes converted as a list
        return list(gene_indices)
