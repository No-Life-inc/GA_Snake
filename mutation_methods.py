import torch

def random_mutation(child, mutation_rate):
    # Randomly change some genes in the child's genome
    for i, layer in enumerate(child.genome):
        mutation_mask = torch.rand_like(layer) < mutation_rate
        random_genes = torch.randn_like(layer)  # Assuming normal distribution for random genes
        child.genome[i] = torch.where(mutation_mask, random_genes, layer)

def swap_mutation(child, mutation_rate):
    for i, layer in enumerate(child.genome):
        if torch.rand(1).item() < mutation_rate:
            # Select two random indices within the layer to swap
            idx1, idx2 = torch.randint(0, layer.shape[0], (2,)).tolist()
            layer[idx1], layer[idx2] = layer[idx2], layer[idx1]

def inversion_mutation(child, mutation_rate):
    for i, layer in enumerate(child.genome):
        if torch.rand(1).item() < mutation_rate:
            # Select a random range within the layer to invert
            start, end = torch.sort(torch.randint(0, layer.shape[0], (2,)))
            child.genome[i][start:end] = child.genome[i][start:end].flip(dims=[0])