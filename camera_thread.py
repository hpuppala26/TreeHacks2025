import cv2
import queue
from threading import Thread

class ThreadedCamera:
    def __init__(self, src=0, width=480, height=360):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Initialize the queue used to store frames
        self.q = queue.Queue(maxsize=2)
        
        # Start frame thread
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
                
            # If the queue is full, remove one frame
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            self.q.put(frame)

    def read(self):
        return True, self.q.get()

    def release(self):
        self.capture.release()