# 2020-02-18. Leonardo Molina.
# 2021-08-06. Last modified.

import time
import cv2 as cv
from .events import Publisher
from .scheduler import Scheduler

class VideoCapture(cv.VideoCapture):
    def __init__(self, *args, **kwargs):
        self.__time = 0
        self.__index = 0
        self.__start = time.time()
        self.__isVideoFile = False
        self.__publisher = Publisher()
        self.__scheduler = Scheduler()
        self.__scheduler.subscribe(lambda scheduler : self.__update())
        super().__init__()
        if len(args) > 0 or len(kwargs) > 0:
            self.open(*args, **kwargs)
    
    def __update(self):
        if super().isOpened() and super().grab():
            self.__publisher.invoke("grab", self)
            if not self.__isVideoFile:
                self.__index += 1
    
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exc_value, traceback):
        self.release()
    
    @property
    def playing(self):
        return self.__scheduler.running
    
    @property
    def time(self):
        if self.__isVideoFile:
            value = super().get(cv.CAP_PROP_POS_FRAMES) / super().get(cv.CAP_PROP_FPS)
        else:
            value = (self.__time + time.time() - self.__start) if self.playing else self.__time
        # print("__time:%.2f time:%.2f start:%.2f" % (self.__time, time.time(), self.__start))
        return value
    
    @time.setter
    def time(self, value):
        if self.__isVideoFile:
            fps = super().get(cv.CAP_PROP_FPS)
            self.index = round(value * fps)
    
    @property
    def index(self):
        return int(super().get(cv.CAP_PROP_POS_FRAMES)) if self.__isVideoFile else self.__index
    
    @index.setter
    def index(self, value):
        # For video file:
        #   Will seek manually when CV API does not work by reopening the file if value has past.
        # For webcam:
        #   Will ignore request.
        if self.__isVideoFile:
            playing = self.playing
            self.stop()
            self.join()
            current = super().get(cv.CAP_PROP_POS_FRAMES)
            total = super().get(cv.CAP_PROP_FRAME_COUNT)
            value = min(max(int(value), 1), total)
            super().set(cv.CAP_PROP_POS_FRAMES, value)
            if current == super().get(cv.CAP_PROP_POS_FRAMES) and current < total:
                if value < super().get(cv.CAP_PROP_POS_FRAMES):
                    self.release()
                    self.open()
                while super().get(cv.CAP_PROP_POS_FRAMES) < value and super().grab():
                    super().read()
            if playing:
                self.start()
    
    def open(self, *args, **kwargs):
        self.__isVideoFile = (len(args) > 0 or len(kwargs) > 0) and ("filename" in kwargs.keys() or (len(args) > 0 and isinstance(args[0], str)))
        result = super().open(*args, **kwargs)
        return result
    
    @property
    def resolution(self):
        return (int(super().get(cv.CAP_PROP_FRAME_WIDTH)), int(super().get(cv.CAP_PROP_FRAME_HEIGHT)))
    
    @resolution.setter
    def resolution(self, dimensions):
        super().set(cv.CAP_PROP_FRAME_WIDTH, dimensions[0])
        super().set(cv.CAP_PROP_FRAME_HEIGHT, dimensions[1])
    
    def release(self, *args, **kwargs):
        self.stop()
        super().release(*args, **kwargs)
    
    def subscribe(self, event, callback):
        return self.__publisher.subscribe(callback, event)
    
    def start(self):
        self.__start = time.time()
        self.__scheduler.repeat(delay=0, interval=0, repetitions=-1, start=True)
    
    def stop(self):
        if self.playing:
            self.__time += time.time() - self.__start
            self.__scheduler.stop()
    
    def join(self):
        self.__scheduler.join()
        
if __name__ == "__main__":
    import numpy as np
    import time
    from videoCapture import VideoCapture
    
    test = 2
    
    if test == 1:
        def onGrab(stream):
            global frame
            stream.retrieve(frame)
        
        # Test 1: Async playback.
        # source = 0
        source = "C:/Users/molina/Documents/public/HALO/data/Tracking/DS97.avi"
        stream = VideoCapture(source)
        _, frame = stream.read()
        stream.subscribe("grab", onGrab)
        stream.start()
        while True:
            cv.imshow("Test", frame)
            key = cv.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if stream.playing:
                    stream.stop()
                else:
                    stream.start()
        stream.release()
        cv.destroyAllWindows()
    
    elif test == 2:
        # Test 2: Sync playback.
        source = "C:/Users/molina/Documents/public/HALO/data/Tracking/DS97.avi"
        stream = VideoCapture(source)
        _, frame = stream.read()
        cv.imshow("Test", frame)
        print("Frame:%i" % stream.index)
        key = cv.waitKey(0) & 0xFF
        # stream.index = 2
        _, frame = stream.read()
        print("Frame:%i" % stream.index)
        cv.imshow("Test", frame)
        key = cv.waitKey(0) & 0xFF
        stream.release()
        cv.destroyAllWindows()