import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

n=100 #particle numbers
nstep_per_image = 1
t=0
dt= 0.01
xdata=[]
ydata=[]
zdata=[]

data= open("Data.txt")
data.readline()
lines = data.readlines()
size=len(lines)
for line in lines:
    line= line.strip().split("\t")
    xdata.append(line[0])
    ydata.append(line[1])
    zdata.append(line[2])

xdata=np.array(xdata)
ydata=np.array(ydata)
zdata=np.array(zdata)
xdata=xdata.astype(float)
ydata=ydata.astype(float)
zdata=zdata.astype(float)
endtime = len(xdata)/n*dt




fig = plt.figure()
ax = p3.Axes3D(fig)


init = [[4, 6],
       [ 4, 6],
       [ 4, 6]]




datax = np.reshape(xdata, (11, 100))
datay = np.reshape(ydata, (11, 100))
dataz = np.reshape(zdata, (11, 100))
print(len(datax[0]))

x=datax[0]
y=datay[0]
z=dataz[0]


points, = ax.plot(x, y, z, '*')
txt = fig.suptitle('')



def update_points(num, x, y, z, points):
    txt.set_text('num={:d}'.format(num)) # for debug purposes

    # calculate the new sets of coordinates here. The resulting arrays should have the same shape
    # as the original x,y,z
    new_x = datax[num]#datax[num]
    new_y = datay[num]
    new_z = dataz[num]


    # update properties
    points.set_data(new_x,new_y)
    points.set_3d_properties(new_z, 'z')

    # return modified artists
    return points,txt

ax.set_xlim3d([-64, 64])
ax.set_ylim3d([-64, 64])
ax.set_zlim3d([-64, 64])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ani=animation.FuncAnimation(fig, update_points, frames=11, interval = 500, fargs=(x, y, z, points))

plt.show()
















