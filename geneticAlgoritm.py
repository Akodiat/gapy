import random
from multiprocess import Pool

class GeneticAlgorithm:
    def __init__(self, population, fitnessFunc, fitnessTarget=None):
        self.population = population
        self.fitnessFunc = fitnessFunc
        self.fitnessTarget = fitnessTarget

    def stepGeneration(self, nProcesses=1):
        if nProcesses > 1:
            with Pool(nProcesses) as p:
                fitnesses = p.map(self.fitnessFunc, self.population)
        else:
            fitnesses = [self.fitnessFunc(i) for i in self.population]
        maxFitness = max(fitnesses)
        best = self.population[fitnesses.index(maxFitness)]

        if self.fitnessTarget and maxFitness >= self.fitnessTarget:
            print("Reached target fitness: {}".format(maxFitness))
            return

        # Repopulate with roulette wheel selection
        if maxFitness != 0: # Cannot sample with invalid fitness
            self.population = [g.clone() for g in random.choices(
                self.population,
                weights=fitnesses,
                k=len(self.population) - 1
            )]
            self.population.append(best) # Elitism
        for g in self.population:
            g.mutate()

        return maxFitness, best

    def run(self, nGenerations, onGenerationStep=None, nProcesses=1):
        for gen in range(nGenerations):
            maxFitness, best = self.stepGeneration(nProcesses)
            if onGenerationStep is not None:
                onGenerationStep(gen, maxFitness, best)
        return maxFitness, best