import csv as CSV
import json
import math
import os
import pickle
import sys
import time
# from StructureRNN import StructureRnn
# from RegressionModel import RegressionModel
# from keras.layers import MultiHeadAttention
# from tensorflow.keras.layers import Layer,Dense,Input,Reshape,Bidirectional,GRU,Masking
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from imutils.video import FileVideoStream
from matplotlib import gridspec
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import utils
from DLT import DLTrecon
from KalmanFilter import Tracker
from Rodent import *
from settings import Settings
from utils import *
import threading
matplotlib.use("Agg")

def readCSV(path):
    return list(CSV.reader(open(path)))


def drawUI(gImg, rImg, params, g: Rodent, r: Rodent, color, angleMap):
    global bodyVectors, colors
    origin = params['fixedPoint']
    cv2.circle(gImg, (origin[0][0], origin[0][1]), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(rImg, (origin[1][0], origin[1][1]), radius=3, color=(0, 255, 0), thickness=-1)
    # cv2.putText(gImg, '(<distance(mm) from dot>, <Yaw>,<Pitch>)', (20, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(rImg, '(<distance(mm) from dot>, <Yaw>,<Pitch>)', (20, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    vectors = params['vectors']
    for v, c in zip(vectors, colors):
        cv2.arrowedLine(gImg, tuple((g[v[0]][:2]).astype(int)), tuple((g[v[1]][:2]).astype(int)), c, 4)
        cv2.arrowedLine(rImg, tuple((r[v[0]][:2]).astype(int)), tuple((r[v[1]][:2]).astype(int)), c, 4)
    start = np.array([30, 30])
    for key in angleMap.keys():
        loc = tuple(start)
        parts = key.split("-")
        txt = '''yaw | pitch between {} and {}'''.format("-".join(bodyVectors[parts[0]]),
                                                         "-".join(bodyVectors[parts[1]])).ljust(50)
        cv2.putText(gImg, txt + ''':{:^10}|{:^10}'''.format(str(angleMap[key][0]), str(angleMap[key][1])), loc,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
        cv2.putText(rImg, txt + ''':{:^10}|{:^10}'''.format(str(angleMap[key][0]), str(angleMap[key][1])), loc,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
        start += [0, 20]
    return gImg, rImg


def aggregatePastPoints(pastPoints):
    if len(pastPoints) == 0:
        return None
    length = len(pastPoints)
    init = pastPoints[0] * (1 / length)
    for weight, p in enumerate(pastPoints[1:]):
        init = init + ((weight + 2) / length) * p
    return init * (length / sum(list(range(1, length + 1))))


def formatExportData(data):
    for i in range(len(data)):
        data[i] = round(data[i], 2)
    return str(list(data))


def on_key(event):
    global exitFlag
    sys.stdout.flush()
    if event.key == 'q':
        exitFlag = True


def rotate(p, rotation, isInv=False):
    if not isInv:
        op = np.matmul(p, rotation) * np.array([1, 1, -1])
    else:
        op = np.matmul(p * np.array([1, 1, -1]), rotation)
    return op


def reconstruct3D(gcsv, rcsv, params, threshold=0.95):
    length = len(gcsv) if len(gcsv) <= len(rcsv) else len(rcsv)
    origin2D = [params['g_origin'],params['r_origin']]
    target2D = [[params['g_topRight'],params['r_topRight']],[params['g_bottomLeft'],params['r_bottomLeft']]]
    dltCoeff = params['dltCoeff']
    isScaled = params['isScaled']
    nor = Part((DLTrecon(3, 2, dltCoeff, origin2D)), "origin", 1)
    ntr = Part((DLTrecon(3, 2, dltCoeff, target2D[0])), "topRight", 1)
    nbl = Part((DLTrecon(3, 2, dltCoeff, target2D[1])), "bottomLeft", 1)
    rotation = R.align_vectors([nbl - nor, ntr - nor], [[1, 0, 0], [0, 1, 0]])[0]
    rotation = rotation.as_matrix()
    invRotation = np.linalg.inv(rotation)
    origin = Part(rotate(DLTrecon(3, 2, dltCoeff, origin2D), rotation), "origin", 1)
    tr = Part(rotate(DLTrecon(3, 2, dltCoeff, target2D[0]), rotation), "topRight", 1)
    bl = Part(rotate(DLTrecon(3, 2, dltCoeff, target2D[1]), rotation), "bottomLeft", 1)
    trans_mat = -origin
    scale = [params['dBottomLeft'] / origin.distance(bl), params['dTopRight'] / origin.distance(tr),
             1] if isScaled else 1.0
    scale[2] = (scale[1] + scale[0]) / 2.0
    scale = np.array(scale)
    file = open(os.path.join(params['output'], "3DRecon.csv"), "w")
    csvWriter = CSV.writer(file,delimiter=';')
    csvWriter.writerow(params['parts'])
    for i in range(length):
        print("\rReconstructing 3D Scene ", round(i / length * 100), "% complete",end='')
        gRodent = Rodent(params['parts'], gcsv[i])
        rRodent = Rodent(params['parts'], rcsv[i])
        reconData = {}
        probData = {}
        for name in params['parts']:
            reconData[name] = rotate(DLTrecon(3, 2, dltCoeff, [gRodent[name], rRodent[name]]), rotation)
            probData[name] = min(gRodent.partsLikelihood[name], rRodent.partsLikelihood[name])
        tRodent = Rodent(params['parts'], None, reconData, probData) * scale
        if params['isFixedPointShifted']:
            for part in tRodent.parts:
                tRodent[part] = tRodent[part] + scale * trans_mat
        csvWriter.writerow(
            [tRodent[part].tolist() if tRodent.partsLikelihood[part] > threshold else None for part in params['parts']])
    print("\rReconstructing 3D Scene 100% complete")





def analyze3DReconData(csv, params, targetColumns=None):
    nanDataPoints = {}
    accurateDataPoints = []
    nanCluster = {}
    targetColumns = list(csv.columns) if targetColumns is None else targetColumns
    for column in targetColumns:
        nanDataPoints[column] = []
        nanCluster[column] = {'begin': -2, 'end': -2}
    total = len(csv)
    accurateCluster = {'begin': -2, 'end': -2}
    for index, row in csv.iterrows():
        print('{}/{}'.format(index,len(csv)))
        accurate = True
        for column in targetColumns:
            if -4668 in convert_to_numpy(row[column]):
                cluster = nanCluster[column]
                if cluster['end'] + 1 == index:
                    cluster['end'] = index
                else:
                    if cluster['begin'] != -2:
                        nanDataPoints[column].append(cluster.copy())
                    cluster['begin'] = cluster['end'] = index
                accurate = False
        if accurate:
            if accurateCluster['end'] + 1 == index:
                accurateCluster['end'] = index
            else:
                if accurateCluster['begin'] != -2:
                    accurateDataPoints.append(accurateCluster.copy())
                accurateCluster['begin'] = accurateCluster['end'] = index

    pickle.dump(nanDataPoints, open(os.path.join(params['output'], 'nanDataPoints.pkl'), 'wb'))
    pickle.dump(accurateDataPoints, open(os.path.join(params['output'], 'accDataPoints.pkl'), 'wb'))



def interpolateDataPoints(csv, params, nanDataPoints=None, maxClusterSize=20):
    class InterpolateThread(threading.Thread):
        def __init__(self,csv,col,nanDatapoints):
            super().__init__()
            self.csv=csv
            self.name=col
            self.nanDatapoints=nanDatapoints
            self.progress=0
        def getProgress(self):
            return self.progress
        def run(self):
            for y, candidate in enumerate(self.nanDatapoints):
                self.progress+=1
                if candidate['begin'] == 0 or candidate['end'] == len(csv) - 1:
                    continue
                if candidate['end'] - candidate['begin'] < maxClusterSize:
                    begin = convert_to_numpy(self.csv[self.name][candidate['begin'] - 1])
                    end = convert_to_numpy(self.csv[self.name][candidate['end'] + 1])
                    vector = (end - begin) / (candidate['end'] - candidate['begin'] + 2)
                    current = begin + vector
                    for i in range(candidate['begin'], candidate['end'] + 1):
                        self.csv[self.name][i] = current.tolist()
                        current += vector
    if nanDataPoints is None:
        nanDataPoints = pickle.load(open(os.path.join(params['output'], 'nanDataPoints.pkl'), 'rb'))
    threads=[]
    total=len(csv)
    string=""
    cols=csv.columns
    # cols=['tailTip']
    for col in cols:
        string+=col+" {}/"+str(len(nanDataPoints[col]))+" "
        t=InterpolateThread(csv,col,nanDataPoints[col])
        t.start()
        threads.append(t)
    while any([t.is_alive() for t in threads]):
        print('\r'+string.format(*[t.getProgress() for t in threads]),end='')
        time.sleep(0.4)
    return csv

def plot(csv, params,queue=None):
    global exitFlag
    exitFlag = False
    # csv = pandas.read_csv(file)
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3.2, 1, 1])
    fig.add_subplot(gs[0], projection='3d')
    fig.add_subplot(gs[1])
    fig.add_subplot(gs[2])
    ax, ax1, ax2 = fig.get_axes()
    ax1.set_xlabel("frame number")
    ax2.set_xlabel("frame number")
    ax1.set_ylabel("Angle °")
    ax2.set_ylabel("Angle °")
    fig.canvas.mpl_connect('key_press_event', on_key)
    # gCap = cv2.VideoCapture(params['gvid'])
    # rCap = cv2.VideoCapture(params['rvid'])
    gCap = FileVideoStream(params['gvid'], queue_size=params['framerate'] * 3).start()
    rCap = FileVideoStream(params['rvid'], queue_size=params['framerate'] * 3).start()
    frameTime = 1 / params['framerate']
    vid = None
    ax.view_init(elev=30., azim=0)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    setupBlit = False
    lines = []
    mainPlot = []
    HDData = []
    MDData = []
    BDData = []
    timeline = []
    avgBP = []
    timelineSize = 100
    avgSize = 30
    prevBP = None
    fig.canvas.draw()
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
    padding = None
    finalImage = None
    for i in range(0, len(csv)):
        start = (time.time() * 1000)
        gImg = gCap.read()
        rImg = rCap.read()
        dims = (gImg.shape[0] // 2, gImg.shape[1] // 2)
        gImg = cv2.resize(gImg, dims)
        rImg = cv2.resize(rImg, dims)
        merged = np.concatenate((gImg, rImg), axis=0)
        if exitFlag:
            break
        row = csv.iloc[[i]]
        if row.isnull().any().any():
            avgBP.clear()
            timeline.clear()
            MDData.clear()
            BDData.clear()
            HDData.clear()
            prevBP = None
            continue
        lines.clear()
        for vector in params['vectors']:
            v1 = convert_to_numpy(row[vector[0]].to_numpy())[0] * (
                params['multiply'][vector[0]] if vector[0] in params['multiply'].keys() else 1.0)
            v2 = convert_to_numpy(row[vector[1]].to_numpy())[0] * (
                params['multiply'][vector[1]] if vector[1] in params['multiply'].keys() else 1.0)
            px = np.array([v1[0], v2[0]])
            py = np.array([v1[1], v2[1]])
            pz = np.array([v1[2], v2[2]])
            lines.append([px, py, pz])
            # mainPlot=ax.plot(px, py, pz)
        bp = np.mean([convert_to_numpy(row['headBase'].to_numpy())[0], convert_to_numpy(row['sp2'].to_numpy())[0]],
                     axis=0)
        if prevBP is None:
            prevBP = bp
        avgBP.append(bp)
        if len(avgBP) > avgSize:
            avgBP.pop(0)
        bp = np.mean(avgBP, axis=0)
        MDData.append(utils.vectorYawPitch(bp - prevBP)[0])
        HDData.append(utils.vectorYawPitch(
            convert_to_numpy(row['snout'].to_numpy())[0] - convert_to_numpy(row['headBase'].to_numpy())[0])[0])
        BDData.append(utils.vectorYawPitch(
            convert_to_numpy(row['headBase'].to_numpy())[0] - convert_to_numpy(row['sp2'].to_numpy())[0])[0])
        timeline.append(i)
        if len(timeline) > timelineSize:
            MDData.pop(0)
            HDData.pop(0)
            BDData.pop(0)
            timeline.pop(0)
        prevBP = bp
        if not setupBlit:
            setupBlit = True
            for line in lines:
                mainPlot.extend(ax.plot(line[0], line[1], line[2]))
            # xlim3d, ylim3d, zlim3d = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
            ax.set_xlim3d(0, 1000)
            ax.set_ylim3d(0, 1000)
            ax.set_zlim3d(0, 500)
            plot1 = [ax1.plot(timeline, HDData, linestyle='-.', color='k', label='HeadDirection')[0],
                     ax1.plot(timeline, MDData, linestyle='--', color='m', label='MovementDirection')[0]]
            ax1.legend()
            ax1.axis([timeline[0], timeline[0] + 100, -200, 200])
            plot2 = [ax2.plot(timeline, BDData, linestyle='-.', color='g', label='BodyDirection')[0],
                     ax2.plot(timeline, MDData, linestyle='--', color='m', label='MovementDirection')[0]]
            ax2.axis([timeline[0], timeline[0] + 100, -200, 200])
            # ax.set_zlim3d(min(zlim3d), min(zlim3d) + 150)
            ax2.legend()
            fig.canvas.draw()
            ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
            ax2background = fig.canvas.copy_from_bbox(ax2.bbox)
        else:
            for p, line in zip(mainPlot, lines):
                p.set_data(line[0], line[1])
                p.set_3d_properties(line[2])
            fig.canvas.restore_region(axbackground)  # restore background
            for p in mainPlot:
                ax.draw_artist(p)
            fig.canvas.blit(ax.bbox)

            ax1.axis([timeline[0], timeline[0] + 100, -200, 200])
            plot1[0].set_data(timeline, HDData)
            plot1[1].set_data(timeline, MDData)
            fig.canvas.restore_region(ax1background)
            ax1.draw_artist(plot1[0])
            ax1.draw_artist(plot1[1])
            fig.canvas.blit(ax1.bbox)

            ax2.axis([timeline[0], timeline[0] + 100, -200, 200])
            plot2[0].set_data(timeline, BDData)
            plot2[1].set_data(timeline, MDData)
            fig.canvas.restore_region(ax2background)
            ax2.draw_artist(plot2[0])
            ax2.draw_artist(plot2[1])
            fig.canvas.blit(ax2.bbox)

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if padding is None:
            padding = np.full((merged.shape[0], data.shape[1], 3), 255)
        padding[:data.shape[0], :, :] = data
        if finalImage is None:
            finalImage = np.concatenate((padding, merged), axis=1).astype(np.uint8)
        else:
            finalImage[:, 0:padding.shape[1], :] = padding
            finalImage[:, padding.shape[1]:, :] = merged
        cv2.putText(finalImage, '{}/{}'.format(i + 1, len(csv)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        if queue is None:
            cv2.imshow('output', finalImage)
        else:
            queue.put(finalImage,False)

        if vid is None:
            vid = vid = cv2.VideoWriter(os.path.join(params['output'], "plot.avi"),
                                        cv2.VideoWriter_fourcc(*'MPEG'), 40, (finalImage.shape[1], finalImage.shape[0]))
        vid.write(finalImage)
        if gCap.Q.qsize() < 5 or rCap.Q.qsize() < 5:
            time.sleep(0.001)
        print("Complete:",i,':', time.time() * 1000 - start)
        delay = frameTime - (time.time() * 1000) + start
        delay = 0 if delay < 0 else delay
        if queue is None:
            if cv2.waitKey(1) == ord('q'):
                break
    queue.put(None)
        # plt.pause(delay / 2000)


def applyRigidStruct(csv, params, nanDataPoints=None):
    if nanDataPoints is None:
        nanDataPoints = pickle.load(open(os.path.join(params['output'], 'nanDataPoints.pkl'), 'rb'))
    counts = {}
    for c in nanDataPoints.keys():
        counts[c] = len(nanDataPoints[c])
    total = np.sum(list(counts.values()))
    count = 0
    order = sorted(counts.keys(), key=lambda item: counts[item], reverse=True)
    for column in order:
        index = 0
        while index < len(nanDataPoints[column]):
            print('\rRigid Body Algorithm ', round(count / total * 100), "% ", count)
            cluster = []
            try:
                for j, candidate in enumerate(nanDataPoints[column][index:]):
                    if candidate == nanDataPoints[column][index] + j:
                        cluster.append(candidate)
                    else:
                        break
            except:
                pass
            clusterLen = len(cluster)
            begin = []
            i = cluster[0] - 1
            while i > 0 and i > cluster[0] - 5:
                point = convert_to_numpy(csv[column][i])
                if np.isnan(point).any():
                    break
                else:
                    begin.insert(0, [point])
                i -= 1
            candidate = []
            for i in range(cluster[0] - len(begin), cluster[0]):
                for c2 in csv.columns:
                    if c2 == column:
                        continue
                    point = convert_to_numpy(csv[c2][i])
                    if not np.isnan(point).any() and c2 not in candidate:
                        candidate.append(c2)
            if len(candidate) > 2:
                for c in candidate:
                    for i in range(len(begin)):
                        begin[i].append(convert_to_numpy(csv[c][cluster[0] - len(begin) + i]))
                begin = np.mean(begin, axis=0) if len(begin) > 0 else None
                if begin is not None:
                    temp = {column: begin[0]}
                    for i in range(len(candidate)):
                        temp[candidate[i]] = begin[i + 1]
                    begin = temp
                if begin is not None:
                    for k in range(0, len(cluster), 3):
                        if begin is not None and len(cluster) > 0:
                            localCandidates = [c for c in candidate if
                                               c != column and not np.isnan(convert_to_numpy(csv[c][cluster[k]])).any()]
                            if len(localCandidates) > 2:
                                referenceDistances = [distance(begin[column], begin[c]) for c in localCandidates]
                                referencePoints = [convert_to_numpy(csv[c][cluster[k]]) for c in localCandidates]
                                val, loss = estimate(begin[column], referencePoints, referenceDistances)
                                if loss < 5:
                                    csv[column][cluster[k]] = val
                                    nanDataPoints[column].remove(cluster[k])
            index += len(cluster)
            count += len(cluster)
    pickle.dump(nanDataPoints, open(os.path.join(params['output'], 'nanDataPoints.pkl'), 'wb'))
    return csv

def applyKalmanFilter(csv: pd.DataFrame, params):
    class KalmanThread(threading.Thread):
        def __init__(self,csv,col,dt):
            super().__init__()
            self.csv=csv
            self.col=csv[col]
            self.name=col
            self.dt=dt
            self.tracker=None
            self.progress=0
            self.data=[]
        def getProgress(self):
            return self.progress
        def run(self):
            for index,row in self.col.items():
                point=convert_to_numpy(row)
                if -4668 in point:
                    self.tracker=None
                    self.data.append(None)
                else:
                    if self.tracker is None:
                        self.tracker= Tracker(point,self.dt)
                        self.data.append(point.tolist())
                    else:
                        self.data.append(self.tracker.update(point).tolist())
                self.progress+=1
            self.csv[self.name]=self.data
    dt = 1 / params['framerate']
    threads=[]
    total=len(csv)
    string=""
    for col in csv.columns:
        string+=col+" {}/"+str(total)+" "
        t=KalmanThread(csv,col,dt)
        t.start()
        threads.append(t)
    while any([t.is_alive() for t in threads]):
        print('\r'+string.format(*[t.getProgress() for t in threads]),end='')
        time.sleep(0.4)
    return csv


def applyConstraints(csv, params):
    columns = list(params['multiply'].keys())
    total = len(csv) * len(columns)
    count = 0
    for column in columns:
        for i, val in csv[column].items():
            print('\rApplying Constraints:{}% total processed: {}'.format(round(count / total * 100), count),end='')
            point = convert_to_numpy(val)
            if -4668 in point:
                count += 1
                continue
            else:
                csv[column][i] = (point * params['multiply'][column]).tolist()
            count += 1
    return csv

def applySmoothing(csv,params):
    class AvgThread(threading.Thread):
        def __init__(self, csv, col):
            super().__init__()
            self.csv = csv
            self.col = csv[col]
            self.name = col
            self.avg =[]
            self.progress = 0
            self.data = []

        def run(self):
            for index,row in self.col.items():
                point=convert_to_numpy(row)
                if -4668 in point:
                    self.avg.clear()
                    self.data.append(None)
                else:
                    self.avg.append(point)
                    if len(self.avg)>5:
                        self.avg.pop(0)
                    point=np.mean(self.avg,axis=0)
                    self.data.append(point.tolist())
                self.progress+=1
            self.csv[self.name]=self.data

        def getProgress(self):
            return self.progress
    threads = []
    total = len(csv)
    string = ""
    for col in csv.columns:
        string += col + " {}/" + str(total) + " "
        t = AvgThread(csv, col)
        t.start()
        threads.append(t)
    while any([t.is_alive() for t in threads]):
        print('\r' + string.format(*[t.getProgress() for t in threads]), end='')
        time.sleep(0.4)
def generateKinematicsData(csv, params):
    #Deprecated
    accDataPoints = pickle.load(open(os.path.join(params['output'], 'accDataPoints.pkl'), 'rb'))
    writer = CSV.writer(open(os.path.join(params['output'], 'kinematicsData.csv'), 'w'))
    header = ['index']
    header.extend(list(params['Trajectory'].keys()))
    header.extend(list(params['bodyVectors'].keys()))
    header.extend(['A_HB', 'A_BD', 'A_MD', 'MD_HB', 'MD_BD', 'V_MD'])
    writer.writerow(header)
    alias = params['alias']
    averageNFrames = 30
    dt = 1 / params['framerate']
    for point in accDataPoints:
        rows = []
        tracker = None
        avgMD = []
        for i in range(point['begin'], point['end'] + 1):
            row = [i]
            # Does not support multiple trajectory keys for now because used local variable (ex mdVector)
            for key in params['Trajectory'].keys():
                if i != point['begin']:
                    curr = utils.evaluateFunction(params['Trajectory'][key]["fn"],
                                                  [convert_to_numpy(csv[alias[p]][i]) for p in
                                                   params['Trajectory'][key]["param"]])
                    prev = utils.evaluateFunction(params['Trajectory'][key]["fn"],
                                                  [convert_to_numpy(csv[alias[p]][i - 1]) for p in
                                                   params['Trajectory'][key]["param"]])
                    mdVector = curr - prev
                    avgMD.append(mdVector)
                    if len(avgMD) > averageNFrames:
                        avgMD.pop(0)
                    mdVector = np.mean(avgMD, axis=0)
                    if tracker is None:
                        tracker = Tracker(mdVector, dt)
                    else:
                        mdVector = tracker.update(mdVector)
                    row.append(mdVector)
                else:
                    row.append('')
            for key in params['bodyVectors'].keys():
                vectors = params['bodyVectors'][key]
                row.append(convert_to_numpy(csv[alias[vectors[0]]][i]) - convert_to_numpy(csv[alias[vectors[1]]][i]))
            row.append(utils.vectorYawPitch(row[2])[0])
            row.append(utils.vectorYawPitch(row[3])[0])
            if type(row[1]) != str:
                row.append(utils.vectorYawPitch(row[1])[0])
                row.append(utils.getCosineAngle(row[1], row[2]))
                row.append(utils.getCosineAngle(row[1], row[3]))
                row.append(utils.magnitude(row[1]) / dt)
            else:
                row.extend(['', '', '', ''])

            rows.append(row)
        writer.writerows(rows)


import random


def generateGraphics(csv, params, targetColumns=None):
    accDataPoints = pickle.load(open(os.path.join(params['output'], 'accDataPoints.pkl'), 'rb'))
    angles = {}
    velocity = []
    index = random.randint(0, len(accDataPoints) - 1)
    while (accDataPoints[index]['end'] - accDataPoints[index]['begin'] < 100):
        index = random.randint(0, len(accDataPoints))
    targetColumns = list(csv.columns) if targetColumns is None else targetColumns
    for column in targetColumns:
        angles[column] = []
    t = []
    for index2, row in csv.loc[accDataPoints[index]['begin']:accDataPoints[index]['end']].iterrows():
        if row.isna().any():
            continue
        velocity.append(float(row["V_MD"]))
        for column in targetColumns:
            angles[column].append(float(row[column]))
    plt.figure(1, (10, 10))
    plt.subplot(311)
    x = list(range(accDataPoints[index]['begin'] + 1, accDataPoints[index]['end'] + 1))
    plt.plot(x, angles['A_HB'], linestyle='-.', color='b', label='HeadDirection')
    plt.plot(x, angles['A_MD'], linestyle='--', color='r', label='MovementDirection')
    plt.xlabel("Frame number")
    plt.ylabel("Angle °")
    plt.legend()
    plt.subplot(312)
    plt.plot(x, velocity, color='r', label='Movement Direction speed')
    plt.xlabel("Frame number")
    plt.ylabel("mm/s")
    plt.legend()
    plt.subplot(313)
    plt.plot(x, angles['A_BD'], linestyle=':', color='g', label='BodyDirection')
    plt.plot(x, angles['A_MD'], linestyle='--', color='r', label='MovementDirection')
    plt.xlabel("Frame number")
    plt.ylabel("Angle °")
    plt.legend()
    plt.savefig('plot{}-{}.png'.format(accDataPoints[index]['begin'], accDataPoints[index]['end']), dpi=600)
    plt.show()

def sanityCheckRecon(csv,params,thresholdDistance=20):
    for column in ['tailTip','tailMid','tailBase']:
        for index, row in csv[column].items():
            point = convert_to_numpy(row)
            if (-4668 not in point) and (point[0]<-100 or point[0]>1100 or point[1]<-100 or point[1]>1100 or point[2]<-20 or point[2]>100):
                print('\rSetting index',column, index, 'val', csv[column][index], 'to None',end='')
                csv[column][index]=None
    for column in csv.columns:
        if column in ['tailTip','tailMid','tailBase']:
            continue
        for index, row in csv[column].items():
            point = convert_to_numpy(row)
            if (-4668 not in point) and (point[0]<-100 or point[0]>1100 or point[1]<-100 or point[1]>1100 or point[2]<-100 or point[2]>450):
                print('\rSetting index',column, index, 'val', csv[column][index], 'to None',end='')
                csv[column][index]=None


    return csv
if __name__ == "__main__":
    Settings('cfg.ini')
    params = Settings.params
    csv = pandas.read_csv(os.path.join(params['output'], "Kalman.csv"),sep=';')
    # printVectors(csv,[['leftEar','snout'],['rightEar','snout']])
    # applySmoothing(csv,params)
    # sanityCheckRecon(csv,params)
    # csv.to_csv(os.path.join(params['output'], "smooth.csv"), index=False,sep=';')
    # analyze3DReconData(csv,params)
    # runModel(csv,params)
    # csv,nanDataPoints=analyze3DReconData(csv,params)
    # csv=applyKalmanFilter(csv,params)
    # csv.to_csv(os.path.join(params['output'], "Kalman.csv"), index=False,sep=';')
    # interpolateDataPoints(csv,params)
    # csv.to_csv(os.path.join(params['output'], "Interpolated.csv"), index=False,sep=';')
    # analyze3DReconData(csv, params)
    # csv = applyRigidStruct(csv, params)
    # csv.to_csv(os.path.join(params['output'], "processed.csv"), index=False)
    # csv=applyConstraints(csv,params)
    # csv.to_csv(os.path.join(params['output'], "Constraints.csv"), index=False, sep=';')
    # csv.to_csv(os.path.join(params['output'], "processed.csv"), index=False)
    # csv = pandas.read_csv(os.path.join(params['output'], "processed.csv"))
    # csv = pandas.read_csv(os.path.join(params['output'], "naiveModelInference.csv"), sep=';')
    plot(csv, params)
    # csv = pandas.read_csv(os.path.join(params['output'], "Interpolated.csv"))
    # generateKinematicsData(csv,params)
    # csv = pandas.read_csv(os.path.join(params['output'], "kinematicsData.csv"), index_col=0)
    # generateGraphics(csv,params,"A_HB,A_BD,A_MD,MD_HB,MD_BD".split(','))
