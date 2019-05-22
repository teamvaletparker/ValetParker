from PIL import ImageGrab
import numpy
import time
import cv2
import threading
import sys

count = 0

def screenshot():
        image = ImageGrab.grab(bbox=(86 ,177, 1280, 720))
        printScreen = numpy.array(image)
        return  printScreen

def get_sctime():
        timer = threading.Timer(1,get_sctime)
        #timer.daemon = True
        timer.start()
        if count == 0:
                cv2.imshow('Capture monitor', cv2.cvtColor(screenshot(), cv2.COLOR_BGR2RGB))
                print('캡쳐')
        if cv2.waitKey(0) & 0xFF == ord('q'):
                timer.cancel()
                cv2.destroyAllWindows()


get_sctime()
print("Capture is over")
count = 1
sys.exit


