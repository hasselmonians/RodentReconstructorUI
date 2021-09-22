import argparse
import configparser
import csv
import datetime
import json
import math
import os
import random
import sys
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R

import Estimator
from DLT import DLTrecon, DLTdecon
from KalmanFilter import Tracker
from Rodent import *
from settings import Settings

def readCSV(path):
    return list(csv.reader(open(path)))


def drawUI(gImg, rImg, params, g: Rodent, r: Rodent, color, angleMap):
    global bodyVectors
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
    length=len(pastPoints)
    init = pastPoints[0]*(1/length)
    for weight,p in enumerate(pastPoints[1:]):
        init = init + ((weight+2)/length)*p
    return init * (length/sum(list(range(1,length+1))))


def formatExportData(data):
    for i in range(len(data)):
        data[i] = round(data[i], 2)
    return str(list(data))


def on_key(event):
    global exitFlag
    sys.stdout.flush()
    if event.key == 'q':
        exitFlag = True


def alphaBetaFilter(measurement, prevState, dt, velocity=0,acceleration=0, a=0.7, b=0.85,g=0.8):
    estimate = prevState + velocity * dt + 0.5*(dt**2)*acceleration
    velocity = velocity + acceleration * dt
    residual = measurement - estimate
    estimate = estimate + residual * a
    velocity = (residual) * (b / dt) + velocity
    acceleration = acceleration + g*residual
    return estimate,velocity,acceleration


def angleBVectors(v1, v2, isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    dot = np.dot(v2[:2], v1[:2])
    det = (v2[0] * v1[1]) - (v2[1] * v1[0])
    yaw = multiplier * math.atan2(det, dot)
    pitch = multiplier * math.acos(v2[2]/sqrt(np.sum(np.square(v2))))
    return [round(yaw, 2), round(pitch, 2)]

def getVectorMatrix(parts,rodent):
    for i,p in enumerate(parts[:-1]):
        for j,p2 in enumerate(parts[i+1:]):
            print(p,'-',p2,':',angleBVectors(rodent[p2],rodent[p]),end=' ')
        print(' ')
    print('-------------------')
if __name__ == "__main__":
    # Begin configuration
    Settings('config.ini')
    params=Settings.params
    distanceMat=lil_matrix((len(params['parts']),len(params['parts'])))
    angleMatrix=lil_matrix((len(params['parts']),len(params['parts'])))
    alias=params['alias']
    gcsv = readCSV(params['gcsv'])
    rcsv = readCSV(params['rcsv'])
    dltCoeff = params['dltCoeff']
    showPlot = params['showPlots']
    showVid = params['showVid']
    isScaled = params['isScaled']
    gCap = cv2.VideoCapture(params['gvid'])
    rCap = cv2.VideoCapture(params['rvid'])
    length = len(gcsv) if len(gcsv) <= len(rcsv) else len(rcsv)
    origin2D = params['fixedPoint']
    target2D = params['targetPoints']
    vectors = params['vectors']
    colors = []
    exitFlag = False
    for i in vectors:
        colors.append([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)])
    if showPlot:
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca(projection='3d')
        fig.canvas.mpl_connect('key_press_event', on_key)
    nor = Part((DLTrecon(3, 2, dltCoeff, origin2D)), "origin", 1)
    ntr = Part((DLTrecon(3, 2, dltCoeff, target2D[0])), "topRight", 1)
    nbl = Part((DLTrecon(3, 2, dltCoeff, target2D[1])), "bottomLeft", 1)

    rotation=R.align_vectors([nbl-nor,ntr-nor],[[1,0,0],[0,1,0]])[0]
    rotation = rotation.as_matrix()
    invRotation = np.linalg.inv(rotation)

    def rotate(p, isInv=False):
        global rotation, invRotation
        if not isInv:
            op = np.matmul(p, rotation)
            op*= np.array([1, 1, -1])
        else:
            op = np.matmul(p* np.array([1,1,-1]), invRotation)
        return op


    origin = Part(rotate(DLTrecon(3, 2, dltCoeff, origin2D)), "origin", 1)
    tr = Part(rotate(DLTrecon(3, 2, dltCoeff, target2D[0])), "topRight", 1)
    bl = Part(rotate(DLTrecon(3, 2, dltCoeff, target2D[1])), "bottomLeft", 1)
    # trans_mat = np.asarray([0., 0., 0.]) - np.asarray(
    #     [(origin[0] + bl[0]) / 2., (origin[1] + tr[1]) / 2., (origin[2] + bl[2] + tr[2]) / 3.])
    trans_mat=-origin
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.scatter([origin[0]], [origin[1]],
    #            [origin[2]], color='#F00')
    # ax.scatter([tr[0]], [tr[1]], [tr[2]], color='#0F0',label="TopRight")
    # ax.scatter([bl[0]], [bl[1]], [bl[2]], color='#00F',label="BottomLeft")
    # ax.plot([tr[0],origin[0],bl[0]],[tr[1],origin[1],bl[1]],[tr[2],origin[2],bl[2]])
    # ax.plot([ntr[0],nor[0],nbl[0]],[ntr[1],nor[1],nbl[1]],[ntr[2],nor[2],nbl[2]])
    # ax.scatter([nor[0]], [nor[1]],
    #            [nor[2]], color='#700')
    # ax.scatter([ntr[0]], [ntr[1]], [ntr[2]], color='#070',label="oTopRight")
    # ax.scatter([nbl[0]], [nbl[1]], [nbl[2]], color='#007',label='oBottomLeft')
    # ax.scatter(0+origin[0],0+origin[1],0+origin[2],color='c',label="origin")
    # ax.scatter(0.5+origin[0],0+origin[1],0+origin[2],color='m',label="x-axis")
    # ax.scatter(0+origin[0],0.5+origin[1],0+origin[2],color='y',label="Y-axis")
    # ax.scatter(0+origin[0],0+origin[1],0.5+origin[2],color='k',label="Z-axis")
    # ax.scatter(0,0,0,color='c')
    # ax.scatter(0.5,0,0,color='m')
    # ax.scatter(0,0.5,0,color='y')
    # ax.scatter(0,0,0.5,color='k')
    # plt.legend()
    # plt.show()
    # exit()
    scale = [params['distanceFixedTarget'] / origin.distance(bl), params['distanceFixedTarget'] / origin.distance(tr),
             1] if isScaled else 1.0
    print(scale)

    scale[2] = (scale[1] + scale[0]) / 2.0
    scale = np.array(scale)
    frameTime = 1 / params['framerate']
    # End Configuration
    # Begin write CSV header
    filename = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    file = open(params['output'] + filename + ".csv",
                "w")  # create a new file at location defined by "output" in the configuration.
    if showPlot or showVid:
        vid = None
    csvWriter = csv.writer(file)
    bodyVectors = params['bodyVectors']
    if params[
        'showMetaData']:  # if true, prints origin, top right, bottom left, and scale factor to convert to desired unit
        csvWriter.writerow(["Scale", scale])
        for k in bodyVectors.keys():
            csvWriter.writerow([k + " : " + str(bodyVectors[k])])
    exportList = params['exportDataList']  # List of data to be exported into CSV file
    header = []
    angleMap = {}
    for p in exportList:
        if p.startswith("A_"):
            angleMap[p[2:]] = []
        header.append(p)
    csvWriter.writerow(header)

    # Main Loop
    prevTrodent = []
    velocityRodent = None
    vectorSubtraction = None
    distanceMatrix = {}
    finalAccurateRepr=None
    prevGRodent=None
    velocityGRodent=None
    accelGRodent=accelRRodent=accelRodent=None
    prevRRodent=None
    velocityRRodent=None
    trackerG2D=None
    trackerR2D = None
    tracker3D=None
    aggregatePrevTRodent=None
    tRodentTracker=None
    for p in params['parts']:
        distanceMatrix[p] = {}
        for p2 in params['parts']:
            distanceMatrix[p][p2] = 0
    for i in range(0,length):
        if exitFlag:
            break
        start = (time.time() * 1000)
        # create rodent objects from CSV
        gRodent = Rodent(params['parts'], gcsv[i])
        rRodent = Rodent(params['parts'], rcsv[i])
        #2D Kalman Tracker
        # if trackerG2D is None:
        #     trackerG2D = Tracker(gRodent,frameTime)
        # else:
        #    gRodent=trackerG2D.update(gRodent)
        #
        # if trackerR2D is None:
        #    trackerR2D = Tracker(rRodent, frameTime)
        # else:
        #    rRodent = trackerR2D.update(rRodent)

        # if prevGRodent is not None:
        #     if velocityGRodent is not None:
        #         gRodent,velocityGRodent,accelGRodent= alphaBetaFilter(gRodent, prevGRodent, frameTime, a=0.45, b=0.5, )
        #     else:
        #         gRodent,velocityGRodent,accelGRodent = alphaBetaFilter(gRodent, prevGRodent, frameTime, velocityGRodent,accelGRodent, a=0.45, j=0.5)
        # if prevRRodent is not None:
        #     if velocityRRodent is not None:
        #         rRodent,velocityRRodent, accelRodent = alphaBetaFilter(rRodent, prevRRodent, frameTime, a=0.9, b=0.9)
        #     else:
        #         rRodent,velocityRRodent, accelRodent = alphaBetaFilter(rRodent, prevRRodent, frameTime, velocityRRodent,accelRRodent,a=0.9, b=0.9)
        # # Read n future location from CSV
        # fgRodent = [Rodent(params['parts'], gcsv[index]) for index in range(i + 1, i + 1 + params['nPastPoints']) if
        #             index < length]
        # frRodent = [Rodent(params['parts'], rcsv[index]) for index in range(i + 1, i + 1 + params['nPastPoints']) if
        #             index < length]

        # Read one frame from video files
        gImg = gCap.read()[1]
        rImg = rCap.read()[1]
        # 3D reconstruction
        reconData = {}
        probData = {}
        # fReconData = [{} for index in range(len(fgRodent))]  # stores reconstructed data of future points
        # fProbData = [{} for index in range(len(fgRodent))]
        for name in params['parts']:
            reconData[name] = rotate(DLTrecon(3, 2, dltCoeff, [gRodent[name], rRodent[name]]))
            probData[name] = min(gRodent.partsLikelihood[name], rRodent.partsLikelihood[name])
            # for index in range(len(fgRodent)):
            #     fReconData[index][name] = rotate(
            #         DLTrecon(3, 2, dltCoeff, [fgRodent[index][name], frRodent[index][name]]))
            #     fProbData[index][name]=min(fgRodent[index].partsLikelihood[name],frRodent[index].partsLikelihood[name])
        tRodent = Rodent(params['parts'], None, reconData, probData) * scale
        # ftRodent = [Rodent(params['parts'], None, fReconData[index],fProbData[index]) * scale for index in range(len(fgRodent))]
        if params['isFixedPointShifted']:
            for part in tRodent.parts:
                tRodent[part] = tRodent[part] + scale * trans_mat
            # for futureRodent in ftRodent:
            #     for part in futureRodent.parts:
            #         futureRodent[part] = futureRodent[part] - scale * origin
        # dRodent = ftRodent[0]
        # for index in range(1, len(ftRodent)):
        #     dRodent += ftRodent[index]
        # dRodent = dRodent * (1 / len(ftRodent))
        for v in bodyVectors.keys():
            # dRodent[v]=dRodent[bodyVectors[v][0]]-dRodent[bodyVectors[v][1]]
            # dRodent.partsLikelihood[v]=min(dRodent.partsLikelihood[bodyVectors[v][0]],dRodent.partsLikelihood[bodyVectors[v][1]])
            tRodent[v] = tRodent[bodyVectors[v][0]] - tRodent[bodyVectors[v][1]]
            tRodent.partsLikelihood[v] = min(tRodent.partsLikelihood[bodyVectors[v][0]],
                                             tRodent.partsLikelihood[bodyVectors[v][1]])
        lowLikelihoodParts = []
        highLikelihoodParts = list(tRodent.parts.keys())
        for p in tRodent.parts:
            if tRodent.partsLikelihood[p] < 0.9 and tRodentTracker is not None:
                lowLikelihoodParts.append(p)
                highLikelihoodParts.remove(p)
            elif p not in params['parts']:
                highLikelihoodParts.remove(p)
        if tRodentTracker is not None:
            # if velocityRodent is None:
            #     tRodent,velocityRodent,accelRodent = alphaBetaFilter(tRodent, aggregatePrevTRodent, frameTime)
            # else:
            #     tRodent,velocityRodent,accelRodent = alphaBetaFilter(tRodent, aggregatePrevTRodent, frameTime, velocityRodent,accelRodent)
            if prevTrodent is not None:
                finalAccurateRepr=aggregatePastPoints(prevTrodent)
                for p in lowLikelihoodParts:
                    if p not in params['parts']:
                        continue
                    if len(highLikelihoodParts)>=4:
                        #length based prediction code
                        dT = [Estimator.distance(finalAccurateRepr[p], finalAccurateRepr[part]) for part in highLikelihoodParts]
                        print("distance ",tRodent[p]," to",end=' ')
                        dT.append(10)
                        refPoints=[tRodent[part] for part in highLikelihoodParts]
                        refPoints.append(prevTrodent[-1][p])
                        tRodent[p] = Estimator.estimate(finalAccurateRepr[p], refPoints, dT)
                        print(tRodent[p])

                    # # 2D reprojection code
                    # if p in params['parts']:
                    #     if params['isFixedPointShifted']:
                    #         scaledDown = tRodent[p] - scale * trans_mat
                    #         scaledDown *= (1 / scale)
                    #     else:
                    #         scaledDown = tRodent[p] * (1 / scale)
                    #     scaledDown = rotate(scaledDown, True)
                    #     decon = DLTdecon(dltCoeff, scaledDown.reshape(1, 3))
                    #     print(p,"Changed ",gRodent[p],rRodent[p],end=' ')
                    #     gRodent[p] = decon[0][:2]
                    #     rRodent[p] = decon[0][2:]
                    #     print("to ",gRodent[p],rRodent[p])
            tRodent = tRodentTracker.update(tRodent)
            vectorSubtraction = tRodent - aggregatePastPoints(prevTrodent)
        else:
            tRodentTracker=Tracker(tRodent, frameTime)
        prevTrodent.append(deepcopy(tRodent))
        # tRodent.translate_z()
        getVectorMatrix(params['parts'],tRodent)
        if len(prevTrodent) > params['nPastPoints']:
            prevTrodent.pop(0)
        # for v in bodyVectors.keys():
        #     vectorSubtraction[v]=dRodent[v]-tRodent[v]
        # Export Data
        exportData = []
        for p in exportList:
            # if p.startswith("V_"):
            #     if len(ftRodent) == 0:
            #         exportData.append([0, 0, 0])
            #     else:
            #         if "-" not in p:
            #             exportData.append(
            #                 list(vectorSubtraction[p[2:]] / (frameTime * params['nPastPoints'])))
            #         else:
            #             parts = p[2:].split('-')
            #             dVector = vectorSubtraction[parts[0]] - vectorSubtraction[parts[1]]
            #             exportData.append(
            #                 list(dVector / (frameTime * params['nPastPoints'])))
            # else:
            if p.startswith("A_"):
                parts = p[2:].split("-")
                angles = angleBVectors(tRodent[parts[0]], tRodent[parts[1]], True)
                angleMap[p[2:]] = angles
                exportData.append(formatExportData(angles))
            elif p in tRodent.parts.keys():
                exportData.append(formatExportData(tRodent[p]))
            else:
                exportData.append("")
        csvWriter.writerow(exportData)

        # draw ui on image and resize
        if showVid:
            gImg, rImg = drawUI(gImg, rImg, params, gRodent, rRodent, colors, angleMap)
            dims = (gImg.shape[0]//2, gImg.shape[1]//2)
            gImg = cv2.resize(gImg, dims)
            rImg = cv2.resize(rImg, dims)
            merged = np.concatenate((gImg, rImg), axis=0)
            cv2.imshow('Video', merged)
        if showPlot:
            # plot 3D reconstruction data

            ax.clear()
            # ax.set_xlim3d(500,1024)
            # ax.set_ylim3d(0, 1024)
            # ax.set_zlim3d(30, 80)
            for p in exportList:
                # if p.startswith("V_"):
                #     if len(ftRodent) == 0:
                #         exportData.append([0, 0, 0])
                #     else:
                #         if "-" not in p:
                #             directionV = tRodent[p[2:]] + 10 * (
                #                     vectorSubtraction[p[2:]] / (vectorSubtraction[p[2:]].magnitude()))
                #             try:
                #                 directionV = [int(directionV[ix]) for ix in range(len(directionV))]
                #             except:
                #                 directionV = tRodent[p[2:]]
                #             px = np.array([int(tRodent[p[2:]][0]), directionV[0]])
                #             py = np.array([int(tRodent[p[2:]][1]), directionV[1]])
                #             pz = np.array([int(tRodent[p[2:]][2]), directionV[2]])
                #         else:
                #             parts = p[2:].split('-')
                #             dVector = vectorSubtraction[parts[0]] - vectorSubtraction[parts[1]]
                #             midPoint = (tRodent[parts[0]] + tRodent[parts[1]]) / 2.0
                #             directionV = midPoint + 20 * (dVector / dVector.magnitude())
                #             try:
                #                 # directionV = [int(directionV[ix]) for ix in range(len(directionV))]
                #                 pass
                #             except:
                #                 directionV = tRodent[parts[0]] - tRodent[parts[1]]
                #             px = np.array([midPoint[0], directionV[0]])
                #             py = np.array([midPoint[1], directionV[1]])
                #             pz = np.array([midPoint[2], directionV[2]])
                #         ax.plot(px, py, pz)
                if vectorSubtraction is not None and not p.startswith("A_") and p not in params['parts']:
                    dVector = vectorSubtraction[p]
                    midPoint = (tRodent[bodyVectors[p][0]] + tRodent[bodyVectors[p][1]]) / 2.0
                    magnitude = sqrt(np.sum(np.square(dVector)))
                    if magnitude < 1e-4:
                        continue
                    directionV = midPoint + 20 * (dVector / magnitude)
                    px = np.array([midPoint[0], directionV[0]])
                    py = np.array([midPoint[1], directionV[1]])
                    pz = np.array([midPoint[2], midPoint[2]])
                    ax.plot(px, py, pz)
            for vector in params['vectors']:
                px = np.array([tRodent[vector[0]][0], tRodent[vector[1]][0]])
                py = np.array([tRodent[vector[0]][1], tRodent[vector[1]][1]])
                pz = np.array([tRodent[vector[0]][2], tRodent[vector[1]][2]])
                ax.plot(px, py, pz)
                # ax.plot(origin[0]*scale,origin[1]*scale,origin[2]*scale)
            xlim3d, ylim3d, zlim3d = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
            ax.set_xlim3d(0, 1000)
            ax.set_ylim3d(0, 1000)
            # ax.set_zlim3d(0, 225)
            ax.set_zlim3d(min(zlim3d), min(zlim3d)+150)
            plt.draw()
            # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            if showVid:
                xDim = merged.shape[0]
                padding = np.full(((xDim - data.shape[0]), data.shape[1], 3), 255)
                data = np.vstack((data, padding))
                merged = np.concatenate((data, merged), axis=1).astype(np.uint8)
            else:
                merged = data.astype(np.uint8)
        if not showVid and not showPlot:
            print("Frame " + str(i - 2) + "/" + str(length - 2))
        else:
            if vid is None:
                vid = vid = cv2.VideoWriter(params['output'] + filename + ".avi",
                                            cv2.VideoWriter_fourcc(*'MPEG'), 120, (merged.shape[1], merged.shape[0]))
            vid.write(merged)
        # Delay code for visualization
        delay = frameTime - (time.time() * 1000) + start
        if params["DEBUG"]:
            print(delay)
        delay = 1 if delay < 0 else delay
        if showVid and showPlot:
            if cv2.waitKey(int(delay / 2) + 1) == ord('q'):
                break
            plt.pause(delay / 2000)
            plt.cla()
        elif showVid:
            if cv2.waitKey(int(delay)) == ord('q'):
                break
        elif showPlot:
            plt.pause(delay / 1000)
            plt.cla()
