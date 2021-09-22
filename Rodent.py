from math import sqrt

import numpy as np
from settings import Settings
class Part(np.ndarray):
    def __new__(cls, arr, name, likelihood):
        obj = np.asarray(arr).view(cls)
        obj.name = name
        obj.likelihood = likelihood
        return obj

    def distance(self, obj):
        return sqrt(np.sum(np.square(np.subtract(obj, self))))

    def magnitude(self):
        return sqrt(np.sum(np.square(self)))

    def __lt__(self, other):
        return self.likelihood < other

    def __gt__(self, other):
        return self.likelihood > other

class Rodent:
    def __init__(self, names: list, csv, arr: dict = None,prob:dict = None):

        self.alias=Settings.params['alias']
        self.names=names
        self.parts = {}
        self.partsLikelihood={}
        if csv is not None:
            i = 1
            for name in names:
                # setattr(self, name, Part([float(csv[i]), float(csv[i + 1]), 0.0], name, float(csv[i + 2])))
                self.parts[name]=Part([float(csv[i]), float(csv[i + 1]), 0.0], name, float(csv[i + 2]))
                self.partsLikelihood[name]=float(csv[i+2])
                i = i + 3
        else:
            nonConf = names.copy()
            for name in arr.keys():
                nonConf.remove(name)
                # setattr(self, name, Part(arr[name], name, 1.0))
                self.parts[name]=Part(arr[name], name, 1.0)
                self.partsLikelihood[name]=prob[name]
            for name in nonConf:
                # setattr(self, name, Part([.0, .0, .0], name, .0))
                self.parts[name]=Part([.0, .0, .0], name, .0)
                self.partsLikelihood[name] = .0

    def translate_z(self):
        min=None
        for p in self.names:
            if min is None or self.parts[p][2] < min:
                min=self.parts[p][2]
        for p in self.names:
            self.parts[p][2]-=min

    def __getitem__(self, item):
        return self.parts[self.alias[item]] if item in self.alias.keys() else self.parts[item] if item in self.parts.keys() else None

    def __setitem__(self, name, val):
        # setattr(self, name, val)
        self.parts[self.alias[name] if name in self.alias.keys() else name]=val
    def __str__(self):
        ret = ""
        for p in self.parts:
            ret += str(self[p]) + ","
        return ret
    def __rsub__(self,other):
        val = {}
        prob = self.partsLikelihood.copy()
        for name in self.parts:
            val[name] = ( other[name] -self[name]) if type(other) == Rodent else other - self[name]
            if type(other) == Rodent:
                prob[name] = min(self.partsLikelihood[name], other.partsLikelihood[name])
        return Rodent(list(self.parts.keys()), None, val, prob)
    def __sub__(self, other):
        val = {}
        prob = self.partsLikelihood.copy()
        for name in self.parts:
            val[name] = (self[name] - other[name]) if type(other) == Rodent else self[name]-other
            if type(other) == Rodent:
                prob[name]=min(self.partsLikelihood[name],other.partsLikelihood[name])
        return Rodent(list(self.parts.keys()), None, val,prob)
    def __radd__(self,other):
        val = {}
        prob = self.partsLikelihood.copy()
        for name in self.parts:
            val[name] = (self[name] + other[name]) if type(other) == Rodent else self[name] + other
            if type(other) == Rodent:
                prob[name] = min(self.partsLikelihood[name], other.partsLikelihood[name])
        return Rodent(list(self.parts.keys()), None, val, prob)
    def __add__(self, other):
        val = {}
        prob = self.partsLikelihood.copy()
        for name in self.parts:
            val[name] = (self[name] + other[name]) if type(other) == Rodent else self[name]+other
            if type(other) == Rodent:
                prob[name] = min(self.partsLikelihood[name], other.partsLikelihood[name])
        return Rodent(list(self.parts.keys()), None, val,prob)

    def __mul__(self, other):
        val = {}
        prob = self.partsLikelihood.copy()
        for name in self.parts:
            val[name] = (self[name] * other[name]) if type(other) == Rodent else self[name]*other
            if type(other)==Rodent:
                prob[name] = min(self.partsLikelihood[name], other.partsLikelihood[name])
        return Rodent(list(self.parts.keys()), None, val,prob)

    def __rmul__(self, other):
        val = {}
        prob = self.partsLikelihood.copy()
        for name in self.parts:
            val[name] = (self[name] * other[name]) if type(other) == Rodent else self[name]*other
            if type(other)==Rodent:
                prob[name] = min(self.partsLikelihood[name], other.partsLikelihood[name])
        return Rodent(list(self.parts.keys()), None, val,prob)