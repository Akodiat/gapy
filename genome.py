import random

class Genome:
    def __init__(self):
        self.age = 0

    def clone():
        raise NotImplementedError("Clone not implemented in baseclass")

    def mutate():
        raise NotImplementedError("Mutate not implemented in baseclass")

class MultiGenome(Genome):
    def __init__(self, genomes):
        super().__init__()
        self.genomes = genomes

    def clone(self):
        other = MultiGenome([g.clone() for g in self.genomes])
        other.age = self.age + 1
        return other

    def mutate(self):
        for g in self.genomes:
            g.mutate()

class ArrayGenome(Genome):
    def __init__(self, values, mutationRate = 1, minVal=None, maxVal=None, mutationDeviation = 0.5):
        super().__init__()
        self.values = values
        self.mutationRate = mutationRate
        self.minVal = minVal
        self.maxVal = maxVal
        self.mutationDeviation = mutationDeviation

    def clone(self):
        other = ArrayGenome(
            self.values[:],
            self.mutationRate,
            self.minVal,
            self.maxVal,
            self.mutationDeviation
        )
        other.age = self.age + 1
        return other

    def mutate(self):
        if random.random() < self.mutationRate:
            for _ in range(max(self.mutationRate, 1)):
                i = random.randrange(len(self.values))
                self.values[i] += random.gauss(0, self.mutationDeviation)
                if self.minVal is not None:
                    self.values[i] = max(self.minVal, self.values[i])
                if self.maxVal is not None:
                    self.values[i] = min(self.maxVal, self.values[i])

    def __str__(self) -> str:
        return str(self.values)


class PolynomialGenome(ArrayGenome):
    # a(x - e)^3 + b(x - e)^2 + c(x - e) + d
    def __init__(self, mutationRate=1, mutationDeviation = 0.01):
        values = [random.gauss(0, 1) for _ in range(5)]
        super().__init__(values, mutationRate, minVal=None, maxVal=None, mutationDeviation=mutationDeviation)

    def evaluate(self, x):
        [a,b,c,d,e] = self.values
        return a*(x - e)**3 + b*(x - e)**2 + c*(x - e) + d

    def clone(self):
        other = PolynomialGenome(self.mutationRate, self.mutationDeviation)
        other.values = self.values[:]
        other.age = self.age + 1
        return other

    def __str__(self) -> str:
        [a,b,c,d,e] = self.values
        return f'{a}(x - {e})^3 + {b}(x - {e})^2 + {c}(x - {e}) + {d}'
