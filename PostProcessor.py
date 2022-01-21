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
    def __init__(self):
        self.cfgFile=None
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
        self.reconstructionTab=self.builder.get_object('reconstruction')
        self.postprocessingTab=self.builder.get_object('postProcessing')
        self.viewTab = self.builder.get_object('view')
        self.projectNameInput=self.builder.get_object('projectName')
        self.newLandmark = self.builder.get_object('newLandmark')
        self.newLandmark.set_activates_default(True)
        self.addBtn=self.builder.get_object('addBtn')
        self.addBtn.set_can_default(True)
        self.addBtn.grab_default()
        self.grid = self.builder.get_object('bodyVectorsGrid')
        self.distanceTopRight = self.builder.get_object('distanceTopRight')
        self.distanceBottomLeft = self.builder.get_object('distanceBottomLeft')
        self.threshold = self.builder.get_object('threshold')
        self.reconProgress = self.builder.get_object('reconProgress')
        self.player = self.builder.get_object('player')
        self.interpolateMax = self.builder.get_object('interpolateMax')
        self.fileName=self.builder.get_object('fileName')
        self.gFileChooser=self.builder.get_object('gFileChooser')
        self.gCSVChooser = self.builder.get_object('gCSVChooser')
        self.rFileChooser = self.builder.get_object('rFileChooser')
        self.rCSVChooser = self.builder.get_object('rCSVChooser')
        self.greenFile=None
        self.redFile=None
        self.greenCSV=None
        self.redCSV=None
        self.window.show_all()
        self.reconstructionTab.set_visible(False)
        self.postprocessingTab.set_visible(False)
        self.viewTab.set_visible(False)

    def buildPixbuf(self,image):
        h, w, d = image.shape
        return GdkPixbuf.Pixbuf.new_from_data(image.tostring(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * d)

    def rebuildGrid(self):
        parts = self.params.get('parts', [])
        for i, p in enumerate(parts):
            self.grid.attach(self.getLabelBox(p), 0, i + 1, 1, 1)
            self.grid.attach(self.getLabelBox(p), i + 1, 0, 1, 1)
        for i in range(len(parts)):
            for j in range(i+1,len(parts)):
                self.grid.attach(Gtk.ToggleButton(),i+1,j+1,1,1)
                self.grid.attach(Gtk.ToggleButton(), j + 1, i + 1, 1, 1)
        vectors=self.params.get('vectors',[])
        for v in vectors:
            idx=[parts.index(v[0]),parts.index(v[1])]
            idx.sort(reverse=True)
            self.grid.get_child_at(idx[0]+1,idx[1]+1).set_active(True)
        self.grid.show_all()

    def addLandmark(self,btn):
        if self.newLandmark.get_text().strip()=='' or self.newLandmark.get_text().strip() in self.params['parts']:
            return
        length=len(self.params['parts'])
        if length:
            for i in range(length,0,-1):
                self.grid.remove_row(i)
                self.grid.remove_column(i)
        self.params['parts'].append(self.newLandmark.get_text().strip())
        self.rebuildGrid()

    def getLabelBox(self,name):
        box=Gtk.Box(spacing=5)
        box.set_name(name)
        btn = Gtk.Button(label='x')
        btn.set_border_width(5)
        btn.connect("clicked", self.removeLandmark,name)
        label = Gtk.Label(label=name)
        box.pack_start(label,True,True,0)
        box.pack_start(btn, False, False, 0)
        box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL),False,False,0)
        return box

    def removeLandmark(self,btn,name):
        index = self.params['parts'].index(name)
        self.params['parts'].pop(index)
        removeIndex=[]
        for i,v in enumerate(self.params['vectors']):
            if name in v:
                removeIndex.append(i)
        removeIndex.reverse()
        for i in removeIndex:
            self.params['vectors'].pop(i)
        self.grid.remove_row(index+1)
        self.grid.remove_column(index+1)
        self.grid.show_all()


    def selectGreenFile(self,file):
        self.greenFile=file.get_filename()
        self.params['gvid'] = self.greenFile
        self.setGreenFile()

    def setGreenFile(self):
        vid = cv2.VideoCapture(self.greenFile)
        _, frame = vid.read()
        vid.release()
        self.greenThumbnail = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.greenThumbnailWindow.set_from_pixbuf(self.buildPixbuf(self.drawInformation(self.greenThumbnail, True)))

    def drawInformation(self,image,isGreen):
        out=image.copy()
        prefix='g_' if isGreen else 'r_'
        points=['origin','bottomLeft','topRight']
        color=[(255,255,255),(255,0,0),(0,255,255)]
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
    def openCFG(self,btn):
        dialog = Gtk.FileChooserDialog(
            title="Please choose a file", parent=self.window, action=Gtk.FileChooserAction.OPEN
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )
        filter_text = Gtk.FileFilter()
        filter_text.set_name("ini files")
        filter_text.add_pattern("*.ini")
        dialog.add_filter(filter_text)
        resp=dialog.run()
        if resp == Gtk.ResponseType.OK:
            self.cfgFile=dialog.get_filename()
            self.settings=Settings(self.cfgFile)
            self.params = Settings.params
            self.distanceBottomLeft.set_text(str(self.params.get('dBottomLeft', '')))
            self.distanceTopRight.set_text(str(self.params.get('dTopRight', '')))
            self.rebuildGrid()
            self.redFile=self.params['rvid']
            self.greenFile=self.params['gvid']
            self.setRedFile()
            self.setGreenFile()
            self.drawInformation(self.greenThumbnail,True)
            self.drawInformation(self.redThumbnail,False)
            self.gFileChooser.set_filename(self.greenFile)
            self.rFileChooser.set_filename(self.redFile)
            self.gCSVChooser.set_filename(self.params['gcsv'])
            self.rCSVChooser.set_filename(self.params['rcsv'])
            self.projectNameInput.set_text(self.params['name'])
        dialog.destroy()

    def saveCFG(self,btn):
        dialog = Gtk.FileChooserDialog(
            title="Save As", parent=self.window, action=Gtk.FileChooserAction.SAVE
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE,
            Gtk.ResponseType.OK,
        )
        filter_text = Gtk.FileFilter()
        filter_text.set_name("ini files")
        filter_text.add_pattern("*.ini")
        dialog.set_do_overwrite_confirmation(True)
        dialog.add_filter(filter_text)
        resp = dialog.run()
        if resp == Gtk.ResponseType.OK:
            self.cfgFile=dialog.get_filename()
            self.updateCFG(None)
        dialog.destroy()

    def updateCFG(self,btn):
        self.settings.save_settings(self.cfgFile,self.params)

    def selectRedFile(self,file):
        self.redFile = file.get_filename()
        self.params['rvid'] = self.redFile
        self.setRedFile()

    def setRedFile(self):

        vid = cv2.VideoCapture(self.redFile)
        _, frame = vid.read()
        vid.release()
        self.redThumbnail = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.redThumbnailWindow.set_from_pixbuf(self.buildPixbuf(self.drawInformation(self.redThumbnail, False)))

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
        self.params['name']=self.projectNameInput.get_text().strip()
        self.params['output']='./output/'+self.projectNameInput.get_text().strip()
        os.makedirs(self.params['output'],exist_ok=True)
        self.reconstructionTab.set_visible(True)
        self.postprocessingTab.set_visible(True)
        self.viewTab.set_visible(True)
        print('here')
        self.tabs.next_page()

    def startReconstruction(self,btn):
        if self.distanceTopRight.get_text().strip()=='' or self.distanceBottomLeft.get_text().strip()=='' or self.threshold.get_text().strip()=='':
            return
        vectors=[]
        for i in range(len(self.params['parts'])):
            for j in range(i+1,len(self.params['vectors'])):
                if self.grid.get_child_at(i+1,j+1)!=None:
                    if self.grid.get_child_at(i+1,j+1).get_active() or self.grid.get_child_at(j+1,i+1).get_active():
                        vectors.append([self.params['parts'][i],self.params['parts'][j]])
        self.params['vectors']=vectors
        self.params['dBottomLeft']=float(self.distanceBottomLeft.get_text().strip())
        self.params['dTopRight']=float(self.distanceTopRight.get_text().strip())
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
    postProcessor=PostProcessor()
    Gtk.main()
