#!/usr/bin/env python

'''
Requires numpy, scipy

'''

import neopixel
#import unicornhat as unicorn
import time, scipy.constants, scipy.special
import scipy
import numpy as np
import sys
import traceback
import os
import datetime
import subprocess
import uuid
import matplotlib.image
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import picamera
import cv2
import io

def ProcessCommandLine():

	"""Create an argparse parser for the command line options."""
	import argparse

	parser = argparse.ArgumentParser(description=__doc__.strip())

	#parser.add_argument('--crossing-blobs', action='store_true', default=False,
	#		    help='Include crossing blobs animation')
 	#parser.add_argument('--duration-sec', metavar='SECONDS', type=float, default=2,
	#		    help='Duration of each mode before cycling')
	parser.add_argument('--debug', action='store_true', default=False,
			    help='Print back trace in event of exception')


	return parser.parse_args()

class NeopixelStrip:
        def __init__(self,
                     LED_COUNT      = 150,     # Number of LED pixels.
                     LED_PIN        = 18,      # GPIO pin connected to the pixels (must support PWM!).
                     LED_FREQ_HZ    = 800000,  # LED signal frequency in hertz (usually 800khz)
                     LED_DMA        = 5,       # DMA channel to use for generating signal (try 5)
                     LED_BRIGHTNESS = 255,     # Set to 0 for darkest and 255 for brightest
                     LED_INVERT     = False,   # True to invert the signal (when using NPN transistor level shift)
             ):

                self._strip=neopixel.Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
                # Intialize the library (must be called once before other functions).
                self._strip.begin()
                self._ind=np.array(range(LED_COUNT))

        def shuffle(self, seed=None):
                np.random.seed(seed)
		np.random.shuffle(self._ind)

        def clear(self):
                """Set all off"""
                for i in range(self._strip.numPixels()):
                        self._strip.setPixelColor(i, neopixel.Color(0, 0, 0))
                self._strip.show()

        def set_element(self, i, (r, g, b)):
                self._strip.setPixelColor(self._ind[i], neopixel.Color(int(r), int(g), int(b)))
		self._strip.show()

class UnicornAsStrip:
	def __init__(self):
		iy, ix=np.mgrid[0:8, 0:8]
 		self._pix_ind=zip(np.ravel(ix), np.ravel(iy))

	def shuffle(self, seed=None):
		np.random.seed(seed)
		np.random.shuffle(self._pix_ind)

	def clear(self):
		unicorn.off()

	def set_element(self, i, (r, g, b)):
		ix, iy=self._pix_ind[i]
		unicorn.set_pixel(ix, iy, int(r), int(g), int(b))
		unicorn.show()

	def scan(self, rgb=(80, 80, 80)):
		for i in range(64):
			self.set_element(i, (80, 80, 80))
			time.sleep(0.025)
			self.set_element(i, (0, 0, 0))

	def set_all(self, (r, g, b)):
		for ix in range(8):
			for iy in range(8):
				unicorn.set_pixel(ix, iy, int(r), int(g), int(b))
		unicorn.show()

def grab_image(camera):
	stream = io.BytesIO()
	camera.capture(stream, format='jpeg', use_video_port=True)
	# Construct a numpy array from the stream
	data = np.fromstring(stream.getvalue(), dtype=np.uint8)
	# "Decode" the image from the array, preserving colour
	image = cv2.imdecode(data, 1)
	# OpenCV returns an array with data in BGR order. Convert to RGB and return
	return image[:, :, ::-1]*1.

class Segmenter:
	def __init__(self):

                self._nempty=0
                self._prevempty=None
                self._totempty=None
                self._totdevempty2=None
                self._segtot=None
                self._cx=[]
                self._cy=[]
                self._pixelid=[]

        def add_empty(self, empty):

                if self._nempty==0:
                        self._totempty=empty
                        self._totdevempty2=np.zeros_like(empty)
                else:
                        self._totempty+=empty
                        self._totdevempty2+=(empty-self._prevempty)**2

                self._prevempty=empty
                self._nempty+=1

        def calc_empty_stats(self):

                if self._nempty<2:
                        raise RuntimeError("Add least two empty images required")

                # Per-pixel average empty image
                self._meanempty=self._totempty*1./self._nempty

                # Per-pixel standard deviation of empty image
                # Estimate from sq difference between subsequent frames
                self._sigempty=np.sqrt(self._totdevempty2/(self._nempty-1)/2.)

                # Set zero values to minimum non-zero value
                self._sigempty[np.where(self._sigempty==0)]=self._sigempty[np.where(self._sigempty>0)].min()

                # Save memory
                del self._totempty
                del self._totdevempty2
                #del self._totempty2

        def normdev(self, im):
		return (im-self._meanempty)/self._sigempty

	def segment(self, im, pixelid=None, nsig=3, ndevquantile=0.9995):

                # Calculate per pixel deviation from mean empty / per pixel std dev in empty
		ndev=self.normdev(im)

                # Identify thresh, the value below which a fraction ndevquantile of the values in ndev lie
                sfndev=np.sort(ndev.flatten())
                thresh=sfndev[int(np.floor(ndevquantile*len(sfndev)))]

                # Identify which pixels in ndev are greater than thresh and nsig
		on=np.where((ndev>thresh) & (ndev>nsig))

                if len(on[0]):

                        # Create an image which is 1 for points in 'on', 0 elsewhere
                        segim=np.zeros_like(im)
                        segim[on]=1.

                        # Calculate the centroid of the image and store it
                        cx, cy=centroid(segim)
                        self._cx.append(cx)
                        self._cy.append(cy)
                        self._pixelid.append(pixelid)

                        # Add it to the total segmented image
                        if self._segtot is None:
                                self._segtot=segim
                        else:
                                self._segtot+=segim
                        return segim

        def meanempty(self):
                return self._meanempty

        def sigempty(self):
                return self._sigempty

        def segtot(self):
                return self._segtot

        def x(self):
                return self._cx

        def y(self):
                return self._cy

        def pixelid(self):
                return self._pixelid

def centroid(image):
        imh, imw=image.shape
	iy, ix=np.mgrid[0:imh, 0:imw]
	x=ix+0.5
	y=iy+0.5
        ixmean=np.sum(ix*image)/np.sum(image)
        iymean=np.sum(iy*image)/np.sum(image)
        return ixmean, iymean

def Run(args):

        print("Initialising strip...")
	#s=UnicornAsStrip()
	s=NeopixelStrip()

	rgb=50, 50, 50

        print("Initialising camera...")
	camera=picamera.PiCamera()
	camera.hflip=True
	camera.vflip=True

        # Adjust setting while showing a spot then fix them
        camera.resolution = (640, 480)
        camera.start_preview()
        s.clear()
        s.set_element(59, rgb)
        fps=10
        camera.framerate=fps*2
        camera.shutter_speed=int(1./fps*1000000)
        #camera.analog_gain=
        #camera.digital_gain=
        time.sleep(2)
        camera.exposure_mode='off'
        camera.awb_mode='off'
        camera.awb_gains=(1., 1.)
        camera.iso=100
        camera.stop_preview()

        nempty=10
        nspots=10 # 115

        segger=Segmenter()

        # Grab empty (no lights on) stack
        s.clear()
        for i in range(nempty):
                print("Empty %d/%d..." % (i+1, nempty))
                segger.add_empty(np.mean(grab_image(camera), -1))
        s.clear()

        segger.calc_empty_stats()

        # Grab and segment images with each light on
        for i in range(nspots):
                print("Image %d/%d..." % (i+1, nspots))
                s.clear()
                pixelid=i+30
                s.set_element(pixelid, rgb)
                segger.segment(np.mean(grab_image(camera), -1), pixelid=pixelid)
        s.clear()

        print("Creating calibration plot...") 
        fig=plt.figure(figsize=(21./2.54, 29.7/2.54))

        x, y, ids=segger.x(), segger.y(), segger.pixelid()

        ax1=fig.add_subplot(3, 1, 1)
        ax1.imshow(segger.meanempty(), cmap='gray')
        ax1.set_title("Mean empty [%g to %g]" % (segger.meanempty().min(), segger.meanempty().max()))
        ax1.plot(x, y, lw=0, marker='x', color='red')
        ax1.set_xlim([0, camera.resolution[0]])
        ax1.set_ylim([camera.resolution[1], 0])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.plot(x, y, color='green')
        ax1.plot(x, y, lw=0, marker='x', color='yellow')
        for i in range(len(x)):
                ax1.text(x[i], y[i], ids[i], fontsize=5, color='magenta')

        ax2=fig.add_subplot(3, 1, 2)
        ax2.imshow(segger.sigempty(), cmap='gray')
        ax2.set_title("Std dev empty [%g to %g]" % (segger.sigempty().min(), segger.sigempty().max()))
        ax2.set_xlim([0, camera.resolution[0]])
        ax2.set_ylim([camera.resolution[1], 0])
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3=fig.add_subplot(3, 1, 3)
        ax3.imshow(segger.segtot(), cmap='gray')
        ax3.set_title("Segmented")
        ax3.set_xlim([0, camera.resolution[0]])
        ax3.set_ylim([camera.resolution[1], 0])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.plot(x, y, color='green')
        ax3.plot(x, y, lw=0, marker='x', color='yellow')
        for i in range(len(x)):
                ax3.text(x[i], y[i], ids[i], fontsize=5, color='magenta')

        pdfname='led_positions.pdf'
        plt.savefig(pdfname)
        print("Wrote '%s'" % pdfname)

if __name__ == "__main__":

	args=ProcessCommandLine()

	try:

		Run(args)

	except Exception, err:

		print("Fatal error: "+str(err))
		if args.debug:
			print "\n--- Failure -------------------------------------"
			traceback.print_exc(file=sys.stdout)
			print "-------------------------------------------------------"
