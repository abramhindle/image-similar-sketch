#!/bin/env python3
import skimage
import skimage.io
from skimage.color import rgb2hsv, hsv2rgb
import skimage.transform
import matplotlib.pyplot as plt
import numpy
import skimage.draw
import random
import argparse

parser = argparse.ArgumentParser(description = "Produce similar pictures via search")
parser.add_argument('-output', default="output/", help='Directory for output')
parser.add_argument('-input', default="inputs/IMG_20180718_193840.jpg", help='File to emulate')
parser.add_argument('-init',default=None, help='Load this file to boostrap the process')
args = parser.parse_args()
output = args.output
imgo = skimage.io.imread(args.input)
img = skimage.transform.resize(imgo, (imgo.shape[0] // 4, imgo.shape[1] // 4), anti_aliasing=True)
plt.imshow(img)
hsvimg = rgb2hsv(img)
canvas = numpy.zeros(hsvimg.shape)
if not args.init is None:
    print("Initing from previous image: %s" % args.init)
    canvas[0:canvas.shape[0],0:canvas.shape[1]] = rgb2hsv(skimage.io.imread(args.init))

def random_point(canvas):
    s = canvas.shape
    w = s[1]
    h = s[0]
    return (random.randint(0,h-1), random.randint(0,w-1))

def random_hsv():
    return (random.random(),random.random(),random.random())

def paletted_random_hsv():
    p = random_point(hsvimg)
    return hsvimg[p[0],p[1]]

def add_random_line(canvas,color=None):    
    rr, cc = skimage.draw.line(*(random_point(canvas) + random_point(canvas)))
    if color is None:
        color = paletted_random_hsv()
    canvas[rr, cc] = color

def random_line_operator(canvas,n=10):
    canvas = numpy.copy(canvas)
    for i in range(n):
        add_random_line(canvas)
    return canvas

def add_random_circle(canvas,color=None):
    pt = random_point(canvas)
    rr, cc = skimage.draw.circle(pt[0],pt[1], random.randint(25,50), canvas.shape)
    if color is None:
        color = paletted_random_hsv()
    canvas[rr, cc] = color    
    
def random_circle_operator(canvas,n=3):
    canvas = numpy.copy(canvas)
    for i in range(n):
        add_random_circle(canvas)
    return canvas

def mse(x, y):
    return numpy.linalg.norm(x - y)

def mse_random_circle(canvas,color=None):
    """ conditionally draws a circle """
    pt = random_point(canvas)
    rr, cc = skimage.draw.circle(pt[0],pt[1], random.randint(3,50), canvas.shape)    
    if color is None:
        color = paletted_random_hsv()
    omse = mse(canvas[rr, cc],hsvimg[rr,cc])
    nmse = mse((canvas[rr, cc]*0 + 1)*color,hsvimg[rr,cc])
    if (nmse < omse):
        canvas[rr, cc] = color
    
    
def mse_circle_operator(canvas,n=300):
    #canvas = numpy.copy(canvas)
    for i in range(n):
        mse_random_circle(canvas)
    return canvas

def mse(x, y):
    return numpy.linalg.norm(x - y)

import skimage.measure

def isim(x,y):
    return skimage.measure.compare_ssim(x,y,multichannel=True) # dynamic range?

def distance(c1,c2):
    m = mse(c1,c2)
    #s = isim(c1,c2)
    m = 1.0/(1+m)
    return m
    #return (s + m)/2.0

operator = mse_circle_operator

oldd = distance(canvas, hsvimg)
lastd = oldd
for i in range(0,5000):
    new_canvas = operator(canvas)
    d = distance(new_canvas, hsvimg)
    if d > oldd:        
        print("Replace! %s - %s" % (i,d))
        oldd = d
        canvas = new_canvas
        ratio = (d+0.000001)/(lastd+0.000001)
        if ratio > 1.01:
            lastd = d
            fname = "%s/%s.png" % (output,d)
            rgbcanvas = hsv2rgb(canvas)
            skimage.io.imsave(fname,rgbcanvas)

fname = "%s/%s.png" % (output,d)
rgbcanvas = hsv2rgb(canvas)
skimage.io.imsave(fname,rgbcanvas)

