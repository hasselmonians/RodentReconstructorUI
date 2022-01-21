import configparser
import json
import os
import numpy as np
import re
import numpy as np
class Settings:

    binaryFunc="([A-Za-z\s]*)\(([\s\w,]*)\)"
    params={}

    def __init__(self,cfg_path=None):
        if cfg_path!=None:
            self.cParser = configparser.ConfigParser()
            self.cParser.optionxform = str
            self.cParser.read(cfg_path)
            if not self.cParser.has_section('path'):
                self.cParser.add_section('path')
            path = self.cParser['path']
            path['output'] = path['output'].replace('\\', '/')
            if path['output'][-1] != '/':
                path['output'] += '/'
            for key in path.keys():
                if not os.path.exists(path[key]):
                    print("can not access: ", path[key])
                    exit()
            name=self.cParser['Meta']['name']
            parts = json.loads(self.cParser['ReconstructionParams']['parts'])
            for p in parts:
                if not p.isalnum():
                    print("Part names can't contain special characters: " + p)
                    exit()
            vectors = {}
            for v in self.cParser['Vectors'].keys():
                vjson = json.loads(self.cParser['Vectors'][v])
                vectors[v] = vjson.split("-")
            multiply={}
            trajectory={}
            for v in self.cParser['Trajectory'].keys():
                trajectory[v]=json.loads(self.cParser['Trajectory'][v])
            for v in self.cParser['multiply'].keys():
                multiply[v]=np.array(json.loads(self.cParser['multiply'][v]))
            Settings.params = {'dltCoeff': json.loads(self.cParser['ReconstructionParams']['dltCoeff']),
                      'parts': parts, 'gvid': path['gvid'],
                      'rvid': path['rvid'], 'gcsv': path['gcsv'], 'rcsv': path['rcsv'],
                      'showPlots': self.cParser['ReconstructionParams'].getboolean('showPlot'),
                      'DEBUG': self.cParser['ReconstructionParams'].getboolean('DEBUG'),
                      'showVid': self.cParser['ReconstructionParams'].getboolean('showVid'),
                      'isScaled': self.cParser['ReconstructionParams'].getboolean('isScaled'),
                      'nPastPoints': int(self.cParser['ReconstructionParams']['nPastPoints']),
                      'vectors': json.loads(self.cParser['ReconstructionParams']['vectors']),
                      'g_origin': json.loads(self.cParser['ReconstructionParams']['g_origin']),
                      'r_origin': json.loads(self.cParser['ReconstructionParams']['r_origin']),
                      'r_bottomLeft': json.loads(self.cParser['ReconstructionParams']['r_bottomLeft']),
                      'r_topRight': json.loads(self.cParser['ReconstructionParams']['r_topRight']),
                      'g_bottomLeft': json.loads(self.cParser['ReconstructionParams']['g_bottomLeft']),
                      'g_topRight': json.loads(self.cParser['ReconstructionParams']['g_topRight']),
                      'dBottomLeft': float(self.cParser['ReconstructionParams']['dBottomLeft']),
                      'dTopRight': float(self.cParser['ReconstructionParams']['dTopRight']),
                      'output': path['output'],
                      'isFixedPointShifted': self.cParser['export'].getboolean('isFixedPointShifted'),
                      'showMetaData': self.cParser['export'].getboolean('showMetaData'),
                      'framerate': float(self.cParser['ReconstructionParams']['framerate']),
                      'multiply':multiply,'Trajectory':trajectory,
                      'name':name
                               }
    def save_settings(self,path,params):
        self.cParser['ReconstructionParams']['dltCoeff']=json.dumps(params['dltCoeff'])
        self.cParser['ReconstructionParams']['parts']=json.dumps(params['parts'])
        self.cParser['path']['rvid']=params['rvid']
        self.cParser['path']['gvid'] = params['gvid']
        self.cParser['path']['gcsv'] = params['gcsv']
        self.cParser['path']['rcsv'] = params['rcsv']
        self.cParser['path']['output'] = params['output']
        self.cParser['ReconstructionParams']['showPlot']=params['showPlots']
        self.cParser['ReconstructionParams']['DEBUG'] = str(params['DEBUG'])
        self.cParser['ReconstructionParams']['showVid'] = str(params['showVid'])
        self.cParser['ReconstructionParams']['isScaled'] = str(params['isScaled'])
        self.cParser['ReconstructionParams']['nPastPoints'] = str(params['nPastPoints'])
        self.cParser['ReconstructionParams']['vectors'] = json.dumps(params['vectors'])
        self.cParser['ReconstructionParams']['g_origin'] = json.dumps(params['g_origin'])
        self.cParser['ReconstructionParams']['r_origin'] = json.dumps(params['r_origin'])
        self.cParser['ReconstructionParams']['g_bottomLeft'] = json.dumps(params['g_bottomLeft'])
        self.cParser['ReconstructionParams']['r_bottomLeft'] = json.dumps(params['r_bottomLeft'])
        self.cParser['ReconstructionParams']['g_topRight'] = json.dumps(params['g_topRight'])
        self.cParser['ReconstructionParams']['r_topRight'] = json.dumps(params['r_topRight'])
        self.cParser['ReconstructionParams']['dBottomLeft'] = str(params['dBottomLeft'])
        self.cParser['ReconstructionParams']['dBottomLeft'] = str(params['dBottomLeft'])
        self.cParser['ReconstructionParams']['framerate'] = str(params['framerate'])
        for k in params['multiply'].keys():
            self.cParser['multiply'][k]=json.dumps(params['multiply'][k].tolist())
        for k in self.cParser['Trajectory'].keys():
            self.cParser['Trajectory'][k]=json.dumps(params['Trajectory'][k])
        self.cParser['Meta']['name']=params['name']
        with open(path,'w') as f:
            self.cParser.write(f)