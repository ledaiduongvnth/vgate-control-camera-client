import datetime
import Jetson.GPIO as GPIO
import time

GPIO.cleanup()
output_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)


class Door(object):
    def __init__(self, delta):
        self.opening_time = datetime.datetime.now()
        self.delta = delta

    def open_door(self):
        if (datetime.datetime.now() - self.opening_time).seconds > self.delta:
            print("opening the door ............................................")
            GPIO.output(output_pin, GPIO.HIGH)
            self.opening_time = datetime.datetime.now()
            time.sleep(0.05)
            GPIO.output(output_pin, GPIO.LOW)

