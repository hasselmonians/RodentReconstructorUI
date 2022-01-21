import numpy as np
import math
from math import sqrt
import json

def magnitude(vector):
    return sqrt(np.sum(np.square(vector)))+1e-5

def evaluateFunction(fn,params):
    if fn.strip().lower()=="avg":
        return np.mean(params,axis=0)

def alphaBetaFilter(measurement, prevState, dt, velocity=0,acceleration=0, a=0.7, b=0.85,g=0.8):
    estimate = prevState + velocity * dt + 0.5*(dt**2)*acceleration
    velocity = velocity + acceleration * dt
    residual = measurement - estimate
    estimate = estimate + residual * a
    velocity = (residual) * (b / dt) + velocity
    acceleration = acceleration + g*residual
    return estimate,velocity,acceleration

def vectorYawPitch(v1,isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    yaw = multiplier * math.atan2(v1[1],v1[0])
    # yaw = yaw + (multiplier *2 *math.pi) if yaw<0 else yaw
    return np.array([yaw,0])
def getCosineAngle(v1,v2,isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    v1=v1[:2]
    v2=v2[:2]
    return math.acos(np.dot(v1,v2)/(magnitude(v1)*magnitude(v2))) * multiplier
def angleBVectors(v1, v2, isDegrees=True):
    multiplier = 57.2958 if isDegrees else 1
    #Yaw Calculations
    y1=multiplier * math.atan2(v1[1],v1[0])
    y2=multiplier * math.atan2(v2[1],v2[0])
    yaw = y2-y1
    # pitch = multiplier * math.acos((v2[2]-v1[2])/magnitude(v2))
    return np.array([yaw, 0])

def getVectorMatrix(parts,rodent):
    for i,p in enumerate(parts[:-1]):
        for j,p2 in enumerate(parts[i+1:]):
            print(p,'-',p2,':',angleBVectors(rodent[p2],rodent[p]),end=' ')
        print(' ')
    print('-------------------')

def convert_to_list(inp):
    if type(inp)!=str and type(inp)!=list and math.isnan(inp):
        t= np.array([-4668,-4668,-4668],dtype=np.int32)
    else:
        if type(inp)!=str:
            t=np.array(inp,dtype=np.int32)
        else:
            if ',' not in inp:
                inp=inp.replace(' ',',')
            t=np.array(json.loads(inp),dtype=np.int32)
    return t

def convert_to_numpy(input):
    if type(input) == np.ndarray and (input.dtype == np.int32 or input.dtype == np.float32):
        return input
    if type(input) == np.ndarray:
        input = np.array(list(map(convert_to_numpy, input)), dtype=np.int32)
    elif type(input) == str:
        split=' '
        if ',' in input:
            split=','
        input = np.array(input[1:-1].strip().split(split)).astype(np.float32)
    else:
        input = np.array([-4668,-4668,-4668], dtype=np.int32)
    return input

def buildInputData(csv,index,seqLen=60):
    PAD=np.array([[-23,-23,-23]]*10)
    section = csv[index:min(index + seqLen, len(csv) - 1)].applymap(convert_to_list)
    data = list(map(convert_to_numpy, section.to_numpy()))
    while len(data) != seqLen:
        data.append(PAD)
    return np.array([data],dtype='float32')

def buildBatchInputData(csv,index,batch=12,seqLen=60):
    PAD=np.array([[-23,-23,-23]]*10)
    indices=[]
    totalLen=len(csv)
    for i in range(index,min(index+seqLen*batch,totalLen),seqLen):
        indices.append(i)
        if i==index:
            inp = buildInputData(csv,i)
        else:
            inp=np.concatenate((inp,buildInputData(csv,i)),axis=0)
    return indices,inp