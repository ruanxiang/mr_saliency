####################################################################
## Author:
##       Xiang Ruan
##       http://ruanxiang.net
##       ruanxiang@gmail.com
## License:
##       GPL 2.0
## Version:
##       1.0
##
## ----------------------------------------------------------------
## A simple live demo showing how to use manifold ranking saliency
## Usage:
##      # python live_demo
##
## I didn't put much effort in this program, it is just a very simple
## example to show how to use MR saliency.
## As you can see the implement of using wxpython is not very prefessional, 
## if you find any bug please feel free to contact me, I will refine the
## program accordingly


import wx
import cv2
import MR
import scipy as sp
import sys

capture = cv2.VideoCapture(0)

if MR.cv_ver >= 3: 
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
else:
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,320)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,240)

capture2 = cv2.VideoCapture(0)
if MR.cv_ver >= 3:
    capture2.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    capture2.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
else:
    capture2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,320)
    capture2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,240)

global mr_sal
mr_sal = MR.MR_saliency()
        
class ButtonPanel(wx.Panel):
    def OnClose(self,event):
        capture.release()
        capture2.release()
        sys.exit()

    def __init__(self,parent):
        wx.Panel.__init__(self,parent)
        #self.save = wx.Button(self,label="Save")
        #self.close = wx.Button(self,label="Close")
        self.close = wx.Button(self,wx.ID_CLOSE)
        self.close.Bind(wx.EVT_BUTTON, self.OnClose)


        self.sizer = wx.GridBagSizer(1,2)
        self.sizer.Add(self.close,(1,1))
        #self.sizer.Add(self.close,(1,2,))
  
class ShowSaliency(wx.Panel):
    def __init__(self, parent, capture, fps=24):
        wx.Panel.__init__(self, parent)
                
        self.capture = capture2
        ret, frame = self.capture.read()


        sal = mr_sal.saliency(frame)
        sal = cv2.resize(sal,(320,240)).astype(sp.uint8)
        sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX)
        outsal = cv2.applyColorMap(sal,cv2.COLORMAP_HSV)
        self.bmp = wx.BitmapFromBuffer(320,240, outsal.astype(sp.uint8))

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)


    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)
        # self.display(sal,style('image'))

    def NextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sal = mr_sal.saliency(frame).astype(sp.uint8)
            sal = cv2.resize(sal,(320,240))
            outsal = cv2.applyColorMap(sal,cv2.COLORMAP_JET)
            self.bmp.CopyFromBuffer(outsal)
            self.Refresh()


class ShowCapture(wx.Panel):
    def __init__(self, parent, capture, fps=24):
        wx.Panel.__init__(self, parent)

        self.SetDoubleBuffered(True)
        
        self.capture = capture
        ret, frame = self.capture.read()
        
        height, width = frame.shape[:2]
        parent.SetSize((width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.bmp = wx.BitmapFromBuffer(width, height, frame)

        self.timer = wx.Timer(self)
        self.timer.Start(1000./fps)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)


    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def NextFrame(self, event):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh()


class Main_Window(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title = title, size = (640,300))

        # Set variable panels
        self.main_splitter = wx.SplitterWindow(self)
        self.up_splitter = wx.SplitterWindow(self.main_splitter)
        self.buttompanel = ButtonPanel(self.main_splitter)
        self.saliencypanel = ShowSaliency(self.up_splitter,capture2)
        self.oripanel = ShowCapture(self.up_splitter,capture)
        self.up_splitter.SplitVertically(self.saliencypanel,self.oripanel,sashPosition=320)
        self.main_splitter.SplitHorizontally(self.up_splitter,self.buttompanel,sashPosition=240)

def main():
    app = wx.App(False)
    frame = Main_Window(None, "Saliency")
    frame.Show()
    app.MainLoop()

if __name__ == "__main__" :
    main()
