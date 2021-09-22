from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from Rodent import Rodent
import numpy as np
from utils import *

class Tracker:
    def __init__(self,data,dt):
        self.dt=dt
        self.tracker=self.getKalmanFilter(data)

    def getKalmanFilter(self,data):
        kalman = KalmanFilter(len(data) * 2, len(data))
        kalman.x = np.hstack((data, [0.0, 0.0, 0.0])).astype(np.float)
        kalman.F = np.array(
            [[1, 0, 0, self.dt, 0, 0], [0, 1, 0, 0, self.dt, 0], [0, 0, 1, 0, 0, self.dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        kalman.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        kalman.P *= 1000
        kalman.R = 0.00001
        kalman.Q = Q_discrete_white_noise(2, dt=self.dt, var=0.5, block_size=3)
        kalman.B = 0
        return kalman

    def get_next_pred(self):
        return self.tracker.H@self.tracker.get_prediction()[0]

    def update(self,data,likelihood=1.0,threshold=0.9):
        self.tracker.predict()
        if likelihood < threshold:
            self.tracker.update(None)
        else:
            self.tracker.update(data)
        return self.tracker.x[:3]


class RodentTracker:
    def __init__(self,rodent:Rodent,dt):
        self.parts={}
        self.dt=dt
        for part in rodent.parts.keys():
            self.parts[part]=Tracker(rodent[part])


    def get_next_pred(self,part):
        return self.parts[part].get_next_pred()

    def update(self,rodent:Rodent,threshold=.80):
        for part in self.parts:
            rodent[part]=self.parts[part].update(rodent[part],rodent.partsLikelihood[part],threshold)
        return rodent

