import json
import os
import time

import pandas

from settings import Settings
import threading
import gi
import pathlib
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from RigidBodyRecon import reconstruct3D,readCSV,applyKalmanFilter,interpolateDataPoints,plot,analyze3DReconData
from gi.repository import Gtk, Gdk,GdkPixbuf,GObject,GLib
from gi.repository import Gst
import multiprocessing
import cv2

settings = Gtk.Settings.get_default()
# settings.set_property("gtk-theme-name", "ChromeOS-dark")
# settings.set_property("gtk-application-prefer-dark-theme", True)
screen = Gdk.Screen.get_default()
provider = Gtk.CssProvider()
provider.load_from_path("style.css")
Gtk.StyleContext.add_provider_for_screen(screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
class PostProcessingThread(threading.Thread):
    def __init__(self,func,csv,path,params,arg=None):
        super(PostProcessingThread,self).__init__()
        self.funct=func
        self.csv=csv
        self.arg=arg
        self.params=params
        self.output_path=path

    def run(self) -> None:
        if self.arg is not None:
            self.funct(self.csv,self.params,None,self.arg)
        else:
            self.funct(self.csv,self.params)
        self.csv.to_csv(self.output_path, index=False,sep=';')

class PostProcessor:
    def __init__(self,cfgPath=None):
        Settings(cfgPath)
        self.params=Settings.params
        self.builder = Gtk.Builder()
        self.builder.add_from_file('PostProcessor.glade')
        self.builder.connect_signals(self)
        self.tabs=self.builder.get_object('tabs')
        self.window = self.builder.get_object("mainWindow")
        self.window.connect("destroy", Gtk.main_quit)
        self.greenThumbnailWindow = self.builder.get_object("greenImage")
        self.origin = self.builder.get_object("origin")
        self.topRight = self.builder.get_object("topRight")
        self.bottomLeft = self.builder.get_object("bottomLeft")
        self.redThumbnailWindow = self.builder.get_object("redImage")
        self.window.show()
        self.reconstructionTab=self.builder.get_object('reconstruction')
        self.postprocessingTab=self.builder.get_object('postProcessing')
        self.viewTab = self.builder.get_object('view')
        self.projectNameInput=self.builder.get_object('projectName')
        self.parts = self.builder.get_object('parts')
        self.listOfVectors = self.builder.get_object('listOfVectors')
        self.distanceTopRight = self.builder.get_object('distanceTopRight')
        self.distanceBottomLeft = self.builder.get_object('distanceBottomLeft')
        self.threshold = self.builder.get_object('threshold')
        self.reconProgress = self.builder.get_object('reconProgress')
        self.player = self.builder.get_object('player')
        self.interpolateMax = self.builder.get_object('interpolateMax')
        self.fileName=self.builder.get_object('fileName')
        self.greenFile=None
        self.redFile=None
        self.greenCSV=None
        self.redCSV=None

    def buildPixbuf(self,image):
        h, w, d = image.shape
        return GdkPixbuf.Pixbuf.new_from_data(image.tostring(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * d)


    def selectGreenFile(self,file):
        self.greenFile=file.get_filename()
        self.params['gvid']=self.greenFile
        vid=cv2.VideoCapture(self.greenFile)
        _,frame=vid.read()
        vid.release()
        self.greenThumbnail=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.greenThumbnailWindow.set_from_pixbuf(self.buildPixbuf(self.drawInformation(self.greenThumbnail,True)))

    def drawInformation(self,image,isGreen):
        out=image.copy()
        prefix='g_' if isGreen else 'r_'
        points=['origin','bottomLeft','topRight']
        color=[(255,255,255),(255,0,0),(0,0,255)]
        for point,color in zip(points,color):
            if prefix+point in self.params:
                out=cv2.putText(out,'({},{})'.format(*self.params[prefix+point]),(self.params[prefix+point][0],self.params[prefix+point][1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)
                out=cv2.circle(out, self.params[prefix+point], 3, color, thickness=-1)
        return out

    def setPixel(self,widget,click):
        if widget.get_name()=='green':
            if self.greenFile==None:
                return
            prefix='g_'
        else:
            if self.redFile==None:
                return
            prefix='r_'
        x=int(click.x)
        y=int(click.y)
        if self.origin.get_active():
            self.params[prefix+'origin']=[x,y]
        if self.topRight.get_active():
            self.params[prefix + 'topRight'] = [x, y]
        if self.bottomLeft.get_active():
            self.params[prefix + 'bottomLeft'] = [x, y]
        if widget.get_name()=='green':
            image=self.greenThumbnail
            out=self.drawInformation(image,True)
            self.greenThumbnailWindow.set_from_pixbuf(self.buildPixbuf(out))
        else:
            image=self.redThumbnail
            out = self.drawInformation(image, False)
            self.redThumbnailWindow.set_from_pixbuf(self.buildPixbuf(out))

    def selectRedFile(self,file):
        self.redFile = file.get_filename()
        self.params['rvid']=self.redFile
        vid = cv2.VideoCapture(self.redFile)
        _, frame = vid.read()
        vid.release()
        self.redThumbnail = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.redThumbnailWindow.set_from_pixbuf(self.buildPixbuf(self.drawInformation(self.redThumbnail,False)))

    def selectGreenCSV(self,file):
        self.params['gcsv']=file.get_filename()

    def selectRedCSV(self,file):
        self.params['rcsv'] = file.get_filename()

    def verifyInformation(self,btn):
        requiredInfo=['rcsv','gcsv','rvid','gvid','g_origin','g_topRight','g_bottomLeft','r_origin','r_topRight','r_bottomLeft']
        for info in requiredInfo:
            if info not in self.params:
                return
        if self.projectNameInput.get_text().strip()=='':
            return
        self.params['output']='./output/'+self.projectNameInput.get_text().strip()
        os.makedirs(self.params['output'],exist_ok=True)
        self.reconstructionTab.set_visible(True)
        self.postprocessingTab.set_visible(True)
        self.viewTab.set_visible(True)
        self.tabs.next_page()

    def startReconstruction(self,btn):
        if self.parts.get_text().strip()=='' or self.listOfVectors.get_text().strip()=='' or self.distanceTopRight.get_text().strip()=='' or self.distanceBottomLeft.get_text().strip()=='' or self.threshold.get_text().strip()=='':
            return
        self.params['parts']=json.loads(self.parts.get_text().strip())
        self.params['vectors']=json.loads(self.listOfVectors.get_text().strip())
        self.params['dBottomLeft']=int(self.distanceBottomLeft.get_text().strip())
        self.params['dTopRight']=int(self.distanceTopRight.get_text().strip())
        threshold = float(self.threshold.get_text().strip())
        t = threading.Thread(target=reconstruct3D,args=(readCSV(self.params['gcsv']),readCSV(self.params['rcsv']),self.params,threshold))
        t.start()

    def interpolateFrames(self,btn):
        if self.interpolateMax.get_text().strip()!='':
            csv=pandas.read_csv(os.path.join(self.params['output'], "3DRecon.csv"),sep=';')
            postprocessingThread=PostProcessingThread(interpolateDataPoints,csv,os.path.join(self.params['output'],'processed.csv'),self.params,int(self.interpolateMax.get_text().strip()))
            postprocessingThread.start()

    def applyKalmanFilter(self,btn):
        csv=pandas.read_csv(os.path.join(self.params['output'], "3DRecon.csv"),sep=';')
        thread=PostProcessingThread(applyKalmanFilter,csv,os.path.join(self.params['output'],'processed.csv'),self.params)
        thread.start()

    def renderThread(self,queue):
        img=queue.get()
        while(img is not None):
            cv2.cvtColor(img,cv2.COLOR_BGR2RGB,dst=img)
            GLib.idle_add(self.player.set_from_pixbuf, self.buildPixbuf(img))
            img=queue.get()

    def playVideo(self,btn):
        file = 'processed.csv' if os.path.exists(os.path.join(self.params['output'],'processed.csv')) else '3DRecon.csv'
        csv = pandas.read_csv(os.path.join(self.params['output'], file), sep=';')
        queue=multiprocessing.Queue()
        t=threading.Thread(target=plot,args=(csv,self.params,queue))
        # t=threading.Thread(target=plot,args=(csv,self.params))
        t.daemon=True
        t.start()
        t1 = threading.Thread(target=self.renderThread,args=(queue,))
        t1.daemon=True
        t1.start()


    def analyzeReconFile(self,btn):
        if self.fileName.get_text().strip()=='':
            return
        csv = pandas.read_csv(os.path.join(self.params['output'], self.fileName.get_text().strip()), sep=';')
        t=threading.Thread(target=analyze3DReconData,args=(csv,self.params))
        t.start()

if __name__ == "__main__":
    postProcessor=PostProcessor('../cfg.ini')
    Gtk.main()
