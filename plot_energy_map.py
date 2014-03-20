#! /usr/bin/env python

import numpy
import scipy
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot
from numpy import random
from mpl_toolkits.mplot3d import Axes3D

def gauss_kern(size, width):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    x, y = scipy.mgrid[0:size, 0:size]
    g = numpy.exp(-(x**2/float(width)+y**2/float(width)))
    return g / g.sum()



def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(im.shape[0], n)
    from numpy.fft import fft2, ifft2
    fftimage = fft2(im)*fft2(g)

    new_im = ifft2(fftimage).real

    new_im = new_im - new_im.min()

    return new_im

seed = random.seed(123088)

total_min = -10
total_max = 10
resolution = 0.5
delta = total_max - total_min

x_ax = numpy.arange(total_min,total_max,resolution)
y_ax = numpy.arange(total_min,total_max,resolution)

minim = []
minim.append( (total_min+delta*0.3,total_min+delta*0.7) )
minim.append( (total_min+delta*0.6,total_min+delta*0.24))
minim.append((total_min+delta*0.3,total_min+delta*0.3))
minim.append((total_min+delta*0.8,total_min+delta*0.4))

random_min = []
for i in range(100):
    x_data = random.uniform(total_min,total_max)
    y_data = random.uniform(total_min,total_max)
    random_min.append( (x_data,y_data) )

#sample broadly
x = (random.sample(100000) - 0.5) * 20
y = (random.sample(100000) - 0.5) * 20

#sample locally
for ax,ay in random_min:
    wide = random.rand() * 5 
    xx = ax + ( (random.sample(1000) - 0.5 ) * wide ) 
    wide = random.rand() * 5 
    yy = ay + ( (random.sample(1000) - 0.5 ) * wide )
    x = numpy.hstack((x,xx))
    y = numpy.hstack((y,yy))

#sample even more in the global minima
for ax,ay in minim:
    wide = random.rand() * 7 
    xx = ax + ( (random.sample(5000) - 0.5 ) * wide )
    wide = random.rand() * 7 
    yy = ay + ( (random.sample(5000) - 0.5 ) * wide )
    x = numpy.hstack((x,xx))
    y = numpy.hstack((y,yy))



#Data acquisition complete
#Create the X,Y,Z array that we will need to plot the data

#Z, xedges, yedges = numpy.histogram2d(y,x,bins=(x_ax,y_ax))
Z, xedges, yedges = numpy.histogram2d(y,x,bins=20,normed=True)
Z = blur_image(Z,2)
Z = Z * (-1)
print xedges
print Z
X,Y = numpy.meshgrid(xedges,yedges)
print Z.shape
print X.shape
print Y.shape

pyplot.figure()
im = pyplot.imshow(Z,interpolation='nearest',origin='low',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
pyplot.show()

X = X[0:20,0:20]
Y = Y[0:20,0:20]
print X.shape

pyplot.figure()
pyplot.contourf(X, Y, Z,20)
#,cmap=pyplot.cm.bone)
pyplot.show()

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=pyplot.cm.jet,shade=True,linewidth=0, antialiased=False)
#ax.set_xlim( (-9,9) )
#ax.set_ylim( (-9,9) )
ax.set_axis_off()
cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.007)
pyplot.show()


###Now make the plots by adding quadratically to Z in a grid basis approach:
x_not = []
y_not = []
for x,y in minim:
    x_not.append(x)
    y_not.append(y)
x_not = numpy.array(x_not)
y_not = numpy.array(y_not)
ZZ = []
tot_min =  numpy.amin(Z)
tot_max =  numpy.amax(Z)
increment = (tot_max - tot_min)/5
for x,y,z in zip(X,Y,Z):
    ZZ_temp = []
    for xx,yy,zz in zip(x,y,z):
        x_o = min(abs(x_not-xx))-2
        y_o = min(abs(y_not-yy))-2
        incr = (x_o**2 + y_o**2 )* increment
        ZZ_temp.append(min(zz+incr,tot_max)) 
        #ZZ_temp.append(zz+incr)
    ZZ.append(ZZ_temp)

ZZ = numpy.array(ZZ)
        


fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,ZZ,rstride=1,cstride=1,cmap=pyplot.cm.jet,shade=True,linewidth=0, antialiased=False)
#ax.set_xlim( (-9,9) )
#ax.set_ylim( (-9,9) )
ax.set_axis_off()
cset = ax.contourf(X, Y, ZZ, zdir='z', offset=-0.007)
pyplot.show()


