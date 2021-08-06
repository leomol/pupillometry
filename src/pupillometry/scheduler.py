# 2020-02-18. Leonardo Molina.
# 2021-08-06. Last modified.

import threading
import time
from .events import Publisher, Subscription

class Scheduler:
    def __init__(self, *args, **kwargs):
        self.fixed = False
        self.loop = None
        self.executor = None
        
        self.__sleep = 0
        self.__delay = 0
        self.__interval = 0
        self.__repetitions = -1
        self.__count = 0
        self.__running = False
        self.__event = threading.Event()
        self.__publisher = Publisher()
    
    @property
    def running(self):
        return self.__running
    
    @property
    def count(self):
        return self.__count
    
    def subscribe(self, callback):
        s = self.__publisher.subscribe(callback, "tick")
        return s
    
    def start(self):
        if self.__running:
            self.__event.set()
        self.__start()
    
    def __start(self):
        self.__running = True
        self.__thread = threading.Thread(target=self.__update, args=())
        # !! self.__thread.daemon = True
        self.__thread.start()
    
    def stop(self):
        if self.__running:
            self.__event.set()
    
    def join(self):
        if self.__running:
            self.__thread.join()
    
    def delay(self, delay, start = True):
        self.repeat(delay=delay, interval=0, repetitions=1, start=start)
    
    def repeat(self, delay = None, interval = None, repetitions = -1, start = True):
        # repeat(interval, repetitions)
        # repeat(delay, interval, repetitions)
        if start:
            if self.__running:
                self.__event.set()
        if delay is None:
            delay = interval
        if interval is None:
            interval = delay
        self.__sleep = delay
        self.__delay = delay
        self.__interval = interval
        self.__repetitions = repetitions
        self.__count = 0
        if start:
            self.__start()
    
    def __update(self):
        start = time.time()
        while (self.__repetitions == -1 or self.__count < self.__repetitions) and not self.__event.wait(self.__sleep):
            self.__count += 1
            if self.fixed:
                sleep = self.__interval
            else:
                lag = time.time() - start - self.__delay - (self.__count - 1) * self.__interval
                sleep = self.__interval - lag
            self.__sleep = sleep
            self.__publisher.invoke("tick", self)
        self.__count = 0
        self.__event.clear()
        self.__running = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exc_value, traceback):
        if self.__running:
            self.__event.set()
        
if __name__ == "__main__":
    def update(scheduler):
        print(scheduler.count)
    
    step = 0.050
    s1 = Scheduler()
    s1.subscribe(update)
    print("Start background loop.")
    s1.repeat(delay=step, interval=step, repetitions=20, start=True)
    s2 = Scheduler()
    s2.subscribe(lambda scheduler : s1.stop())
    print("Schedule early stop.")
    s2.delay(delay=10.5 * step, start=True)
    # Wait for s2 to finish.
    s2.join()
    # Wait for s1 to finish.
    s1.join()
    if s1.running:
        print("Thread 1 should not be running.")
    else:
        print("Thread 1 stopped successfully.")
        
    print("Wait for 2 seconds.")
    s2.delay(delay=2.00, start=True)
    s2.join()
    
    print("Restart thread 1. Do not stop.")
    s1.start()
    s1.join()
    
    print("Done")