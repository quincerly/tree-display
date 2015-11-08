#!/usr/bin/env python

'''
Requires numpy, scipy

'''

import unicornhat as unicorn
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
	#parser.add_argument('--debug', action='store_true', default=False,
	#		    help='Print back trace in event of exception')


	return parser.parse_args()

class UnicornAsStrip:
	def __init__(self, seed=None):
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
	def __init__(self, emptystack):
		nempty=len(emptystack)
                #print("segger nempty %d" % nempty)

		# Calculate average empty image
		meanempty=emptystack[0]*0.
		for i in range(nempty):
                        #print("segger empty %d %g to %g" % (i, emptystack[i].min(), emptystack[i].max()))
			meanempty+=emptystack[i]
		meanempty=meanempty*1./nempty
                #print("segger meanempty %g to %g" % (meanempty.min(), meanempty.max()))
		self._meanempty=meanempty

		# Calculate emptystack std dev image
		sigempty=emptystack[0]*0.
		for i in range(nempty):
			dev=emptystack[i]-meanempty
			sigempty+=dev*dev
		sigempty=np.sqrt(sigempty/nempty)
		sigempty[np.where(sigempty==0)]=sigempty[np.where(sigempty>0)].min() # Set zero values to minimum non-zero value
                #print("segger sigempty %g to %g" % (sigempty.min(), sigempty.max()))
		self._sigempty=sigempty

        def meanempty(self):
                return self._meanempty

        def sigempty(self):
                return self._sigempty

        def normdev(self, im):
		return (im-self._meanempty)/self._sigempty

	def segment(self, im, nsig=0.5):
		ndev=self.normdev(im)
                #print("segger normdev %g to %g" % (ndev.min(), ndev.max()))
		on=np.where(ndev>nsig)
		off=np.where(ndev<=nsig)
		segim=im*0.
		segim[on]=1.
		segim[off]=0.
		return segim

def centroid(image):
        imh, imw=image.shape
	iy, ix=np.mgrid[0:imh, 0:imw]
	x=ix+0.5
	y=iy+0.5
        ixmean=np.sum(ix*image)/np.sum(image)
        iymean=np.sum(iy*image)/np.sum(image)
        return ixmean, iymean

def Run(args):

	s=UnicornAsStrip(0)
        s.shuffle()

	rgb=50, 50, 50

	camera=picamera.PiCamera()
	camera.hflip=True
	camera.vflip=True

	#s.scan()

        # Adjust setting while showing a spot then fix them
        camera.resolution = (640, 480)
        camera.start_preview()
        s.clear()
        s.set_element(0, rgb)
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

        nempty=20
        nspots=64

        # Grab empty (no lights on) stack
        s.clear()
        emptystack=[]
        for i in range(nempty):
                print("Empty %d/%d..." % (i+1, nempty))
                empty=np.mean(grab_image(camera), -1)
                emptystack.append(empty)
        s.clear()

        # Grab images with each light on
        ims=[]
        for i in range(nspots):
                print("Image %d/%d..." % (i+1, nspots))
                s.clear()
                s.set_element(i, rgb)
                ims.append(np.mean(grab_image(camera), -1))

        nims=len(ims)
        #nrows=int(np.ceil(np.sqrt(nims+1)))
        #ncols=int(np.ceil((nims+1)*1./nrows))

        segger=Segmenter(emptystack)
        meanempty=segger.meanempty()
        sigempty=segger.sigempty()

        fig=plt.figure()

        #ax=fig.add_subplot(nrows, ncols, 1)
        #ax.set_title("Mean empty [%g to %g]" % (meanempty.min(), meanempty.max()))
        #ax.imshow(meanempty, cmap='hot')
        #ax.set_xticks([])
        #ax.set_yticks([])

        #ax=fig.add_subplot(nrows, ncols, 2)
        #ax.set_title("Std dev empty [%g to %g]" % (sigempty.min(), sigempty.max()))
        #ax.imshow(sigempty, cmap='hot')
        #ax.set_xticks([])
        #ax.set_yticks([])

        segtot=ims[0]*0.
        cxs=[]
        cys=[]
        for i in range(nims):
                print("Segment %d of %d" % (i+1, nims))
                segim=segger.segment(ims[i], nsig=50.)
                segtot+=segim
                #ax=fig.add_subplot(nrows, ncols, i+1)
                #ax.set_title("Segmented")
                #ax.imshow(segim, cmap='hot')
                #ax.set_xticks([])
                #ax.set_yticks([])
                cx, cy=centroid(segim)
                cxs.append(cx)
                cys.append(cy)

        #ax=fig.add_subplot(nrows, ncols, nims+1)
        ax=fig.add_subplot(1, 1, 1)
        ax.imshow(segtot, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(cxs, cys, lw=0, marker='x', color='red')

        plt.savefig('led_positions.pdf')

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
