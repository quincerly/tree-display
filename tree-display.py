#!/usr/bin/env python

'''
Neopixel strip location registration and control
'''

from __future__ import print_function

import neopixel
#import unicornhat as unicorn
import time
import argparse
import numpy as np
import sys
import traceback
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import picamera
import cv2
import io
import json

import treebuttons

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

        def clear(self, show=True):
                """Set all off"""
                for i in range(self._strip.numPixels()):
                        self._strip.setPixelColor(i, neopixel.Color(0, 0, 0))
                if show: self._strip.show()

        def show(self):
                self._strip.show()

        def set_element(self, i, (r, g, b)):
                self._strip.setPixelColor(self._ind[i], neopixel.Color(int(r), int(g), int(b)))

        def numpixels(self):
                return self._strip.numPixels()

class UnicornAsStrip:
	def __init__(self):
		iy, ix=np.mgrid[0:8, 0:8]
 		self._pix_ind=zip(np.ravel(ix), np.ravel(iy))

	def shuffle(self, seed=None):
		np.random.seed(seed)
		np.random.shuffle(self._pix_ind)

	def clear(self):
		unicorn.off()

        def show(self):
                unicorn.show()

	def set_element(self, i, (r, g, b)):
		ix, iy=self._pix_ind[i]
		unicorn.set_pixel(ix, iy, int(r), int(g), int(b))

	def scan(self, rgb=(80, 80, 80)):
		for i in range(64):
			self.set_element(i, (80, 80, 80))
			time.sleep(0.025)
			self.set_element(i, (0, 0, 0))

	def set_all(self, (r, g, b)):
		for ix in range(8):
			for iy in range(8):
				unicorn.set_pixel(ix, iy, int(r), int(g), int(b))

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
                return np.array(self._cx)

        def y(self):
                return np.array(self._cy)

        def pixelid(self):
                return np.array(self._pixelid)

def centroid(image):
        imh, imw=image.shape
	iy, ix=np.mgrid[0:imh, 0:imw]
	x=ix+0.5
	y=iy+0.5
        ixmean=np.sum(ix*image)/np.sum(image)
        iymean=np.sum(iy*image)/np.sum(image)
        return ixmean, iymean

class Calibration:
        def __init__(self):
                self._x, self._y, self._pixelid=None, None, None

        def create(self, s, filename):

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
                s.show()
                fps=10
                camera.framerate=fps*2
                camera.shutter_speed=int(1./fps*1000000)
                time.sleep(2)
                camera.exposure_mode='off'
                camera.awb_mode='off'
                camera.awb_gains=(1., 1.)
                camera.iso=100
                camera.stop_preview()

                nempty=10
                firstpixelid=0
                lastpixelid=149

                segger=Segmenter()

                # Grab empty (no lights on) stack
                s.clear()
                for i in range(nempty):
                        print("Empty %d/%d..." % (i+1, nempty))
                        segger.add_empty(np.mean(grab_image(camera), -1))
                s.clear()

                segger.calc_empty_stats()

                # Grab and segment images with each light on
                for pixelid in range(firstpixelid, lastpixelid+1):
                        print("Image %d/%d..." % (pixelid-firstpixelid+1, lastpixelid-firstpixelid+1))
                        s.clear()
                        s.set_element(pixelid, rgb)
                        s.show()
                        segger.segment(np.mean(grab_image(camera), -1), pixelid=pixelid)
                s.clear()

                imx, imy, self._pixelid=segger.x(), segger.y(), segger.pixelid()
                self._x=(imx-imx.min())/(imx.max()-imx.min())
                self._y=(imy.max()-imy)/(imx.max()-imx.min())

                self._write(filename)

                print("Creating calibration plot...") 
                fig=plt.figure(figsize=(21./2.54, 29.7/2.54))

                ax1=fig.add_subplot(3, 1, 1)
                ax1.imshow(segger.meanempty(), cmap='gray')
                ax1.set_title("Mean empty [%g to %g]" % (segger.meanempty().min(), segger.meanempty().max()))
                ax1.plot(imx, imy, lw=0, marker='x', color='red')
                ax1.set_xlim([0, camera.resolution[0]])
                ax1.set_ylim([camera.resolution[1], 0])
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.plot(imx, imy, color='green')
                ax1.plot(imx, imy, lw=0, marker='x', color='yellow')
                for i in range(len(imx)):
                        ax1.text(imx[i], imy[i], self._pixelid[i], fontsize=5, color='magenta')

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
                ax3.plot(imx, imy, color='green')
                ax3.plot(imx, imy, lw=0, marker='x', color='yellow')
                for i in range(len(imx)):
                        ax3.text(imx[i], imy[i], self._pixelid[i], fontsize=5, color='magenta')

                pdfname=filename+'_full.pdf'
                plt.savefig(pdfname)
                print("Wrote '%s'" % pdfname)

        def _write(self, filename):
                obj={'x': self._x.tolist(),
                     'y': self._y.tolist(),
                     'pixelid': self._pixelid.tolist()}
                with io.open(filename, 'w', encoding='utf-8') as f:
                        f.write(unicode(json.dumps(obj, ensure_ascii=False)))
                        print("Wrote calibration to '%s'" % filename)

        def read(self, filename):
                with io.open(filename, encoding='utf-8') as f:
                        obj=json.loads(f.read())
                self._x=np.array(obj['x'])
                self._y=np.array(obj['y'])
                self._pixelid=np.array(obj['pixelid'])
                print("Read calibration from '%s'" % filename)

        def plot(self, pdfname):
                fig=plt.figure(figsize=(21./2.54, 29.7/2.54))
                ax=fig.add_subplot(1, 1, 1)
                ax.set_aspect('equal')
                ax.set_title("NeoPixel positions")
                ax.plot(self._x, self._y, color='green')
                ax.plot(self._x, self._y, lw=0, marker='x', color='yellow')
                for i in range(len(self._x)):
                        ax.text(self._x[i], self._y[i], self._pixelid[i], fontsize=5, color='magenta')
                ax.set_xlim([self._x.min(), self._x.max()])
                ax.set_ylim([self._y.min(), self._y.max()])

                plt.savefig(pdfname)
                print("Wrote '%s'" % pdfname)

        def get_data(self):
                return self._x, self._y, self._pixelid

def ProcessCommandLine():

	"""Create an argparse parser for the command line options."""
	parser = argparse.ArgumentParser(description=__doc__.strip())

	parser.add_argument('--calibrate', action='store_true', default=False,
                            help='Create pixel position calibration using camera')
 	parser.add_argument('--calibration-name', metavar='FILENAME', type=str, default=None,
                            help='Pixel calibration name to create/read')
	parser.add_argument('--plot', action='store_true', default=False,
                            help='Plot the calibration')
	parser.add_argument('--clear', action='store_true', default=False,
                            help='Switch of all pixels')
	parser.add_argument('--debug', action='store_true', default=False,
			    help='Print back trace in event of exception')


	return parser.parse_args()

class Renderer:
        def __init__(self, strip, cal):
                self._strip=strip
                self._x, self._y, self._pixelid=cal.get_data()

        def square(self, x1, x2, y1, y2, (r, g, b), show=True):
                for i in range(len(self._x)):
                        x=self._x[i]
                        y=self._y[i]
                        if x>=x1 and x<x2 and y>=y1 and y<y2:
                                self._strip.set_element(self._pixelid[i], (r, g, b))
                if show: self._strip.show()

        def chase(self, colours, wait_ms=50, iterations=10):
                """Movie theater light style chaser animation."""
                for it in range(iterations):
                        for icol in range(len(colours)):
                                for i in range(self._strip.numpixels()):
                                        self._strip.set_element(i, colours[(i+icol)%len(colours)])
                                self._strip.show()
                                time.sleep(wait_ms/1000.0)

        def random(self, rgbs, show=True):
                for i in range(len(self._x)):
                        self._strip.set_element(self._pixelid[i], rgbs[np.random.randint(0, len(rgbs))])
                if show: self._strip.show()

        def apply_rgb_fn_xy(self, rgb_fn, show=True):
                rgbs=rgb_fn(self._x, self._y)
                for i in range(len(self._x)):
                        self._strip.set_element(self._pixelid[i], rgbs[i])
                #        self._strip.set_element(self._pixelid[i], rgb_fn(self._x[i], self._y[i]))
                if show: self._strip.show()

def Demo1(r, s):

        rgbmax=50

        r.chase([(  0  , 0, rgbmax),
                 (0, 0, 0),
                 (0, 0, 0)],
                iterations=20)
        r.chase([(rgbmax, rgbmax, rgbmax),
                 (0, 0, 0),
                 (0, 0, 0)],
                iterations=20)
        r.chase([(rgbmax,  0  , 0),
                 (0, 0, 0),
                 (0, 0, 0)],
                iterations=20)

        nx=20
        wait_ms=50
        for ix in range(nx):
                x0=ix/(nx-1.)*0.9-0.9
                r.square(x0+0.10, x0+0.25, 0., 0.9, (  0  , 0, rgbmax))
                r.square(x0+0.25, x0+0.40, 0., 0.9, (rgbmax, rgbmax, rgbmax))
                r.square(x0+0.40, x0+0.55, 0., 0.9, (rgbmax,   0,   0))
                r.square(x0+0.55, x0+0.70, 0., 0.9, (  0  , 0, rgbmax))
                r.square(x0+0.70, x0+0.85, 0., 0.9, (rgbmax, rgbmax, rgbmax))
                r.square(x0+0.85, x0+1.00, 0., 0.9, (rgbmax,   0,   0))
                time.sleep(wait_ms/1000.0)

        s.clear()

        nx=20
        wait_ms=50
        for ix in range(nx):
                x0=ix/(nx-1.)*0.9-0.9
                r.square(x0+0.10, x0+0.40, 0., 0.9, (  0  , 0, rgbmax))
                r.square(x0+0.40, x0+0.70, 0., 0.9, (rgbmax, rgbmax, rgbmax))
                r.square(x0+0.70, x0+1.00, 0., 0.9, (rgbmax,   0,   0))
                time.sleep(wait_ms/1000.0)

        time.sleep(5000/1000.0)

        # Fade
        #wait_ms=10
        #for i in range(0, rgbmax+1, 2):
        #        r.square(0.10, 0.40, 0., 0.9, (    0,     0, rgbmax-i))
        #        r.square(0.40, 0.70, 0., 0.9, (rgbmax-i, rgbmax-i, rgbmax-i))
        #        r.square(0.70, 1.00, 0., 0.9, (rgbmax-i  ,   0,     0))
        #        time.sleep(wait_ms/1000.0)

        #s.clear()

def Demo2(r, s):

        buttons=treebuttons.TreeHatButtons()

        def handle_buttons():
                event=buttons.check_state()
                if event is not None:
                    if event['type']=='release' and buttons.all_on(event['prevcode']):
                        print("Quitting")
                        s.clear()
                        exit()
                    elif event['type']=='release':
                        if event['tsec']>1:
                            print('long ', end='')
                        print('press '+buttons.state_string(event['prevcode']-event['code'], present_only=True))

        rgbmax=50

        xmin=0.
        xmax=1.00
        ymin=0.
        ymax=0.65
        w=xmax-xmin
        h=ymax-ymin

        def foldphase(phase):
                return phase-np.floor(phase)

        rgbs=[]
        rgbs.append((1, 0, 0))
        #rgbs.append((0, 0, 0))
        rgbs.append((0, 1, 0))
        #rgbs.append((0, 0, 0))
        rgbs.append((0, 1, 1))
        #rgbs.append((0, 0, 0))
        rgbs.append((1, 1, 0))
        #rgbs.append((0, 0, 0))
        rgbs.append((1, 0, 1))
        #rgbs.append((0, 0, 0))

        cphase=np.linspace(0, len(rgbs)-1, len(rgbs))/len(rgbs)
        rs=[]
        gs=[]
        bs=[]
        for rgb in rgbs:
                rs.append(rgb[0])
                gs.append(rgb[1])
                bs.append(rgb[2])

        class RGBFn:
                def __init__(self, phase=0., direction='y'):
                        self._phase=phase
                        self._direction=direction
                def set_direction(self, direction):
                        self._direction=direction
                def set_phase(self, phase):
                        self._phase=phase
                def rgb_fn_phase(self, phase):
                        return (np.round(np.interp(foldphase(phase), cphase, rs)*rgbmax),
                                np.round(np.interp(foldphase(phase), cphase, gs)*rgbmax),
                                np.round(np.interp(foldphase(phase), cphase, bs)*rgbmax))
                def __call__(self, x, y):
                        if self._direction=='y':
                                return zip(*self.rgb_fn_phase((y-ymin)/(ymax-ymin)+self._phase))
                        elif self._direction=='x':
                                return zip(*self.rgb_fn_phase((x-xmin)/(xmax-xmin)+self._phase))
                        else:
                                raise RuntimeError("Unknown direction '%s'" % self._direction)

        randrgbs=map(lambda rgb: (rgb[0]*rgbmax, rgb[1]*rgbmax, rgb[2]*rgbmax), rgbs)
        wait_ms=50.
        for i in range(50):
                r.random(randrgbs)
                time.sleep(wait_ms/1000.0)

        nt=20
        wait_ms=25.
        rgbfn=RGBFn()
        nx=5
        ny=5
        while True:
                rgbfn.set_direction('y')
                for i in range(ny):
                        for it in range(nt):
                                handle_buttons()
                                rgbfn.set_phase(it*1./nt)
                                r.apply_rgb_fn_xy(rgbfn)
                                time.sleep(wait_ms/1000.0)
                rgbfn.set_direction('x')
                for i in range(nx):
                        for it in range(nt):
                                handle_buttons()
                                rgbfn.set_phase(it*1./nt)
                                r.apply_rgb_fn_xy(rgbfn)
                                time.sleep(wait_ms/1000.0)

        s.clear()

        #time.sleep(wait_ms/1000.0)

def Run(args):

        print("Initialising strip...")
        #s=UnicornAsStrip()
        s=NeopixelStrip()

        calibration_name=args.calibration_name
        if calibration_name is None:
                user=os.environ.get('SUDO_USER')
                if user is None:
                        user=os.environ.get('USER')
                if user is not None:
                        calibration_name=os.path.join('/home', user, '.pixel_strip')
        if calibration_name is None:
                calibration_name='.pixel_strip'

        cal=Calibration()
        if args.calibrate:
                cal.create(s, calibration_name)
        else:
                cal.read(calibration_name)
        if args.plot:
                cal.plot(calibration_name+'_locations.pdf')
        if args.clear:
                s.clear()
        else:
                r=Renderer(s, cal)
                #Demo1(r, s)
                Demo2(r, s)

if __name__ == "__main__":

	args=ProcessCommandLine()

	try:

		Run(args)

	except Exception, err:

		print("Fatal error: "+str(err))
		if args.debug:
			print("\n--- Failure -------------------------------------")
			traceback.print_exc(file=sys.stdout)
			print("-------------------------------------------------------")
