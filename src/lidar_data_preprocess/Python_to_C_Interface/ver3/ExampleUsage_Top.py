import numpy as np
import math

# -------------------- 0. Python Lidar preprocess, used for comparison ---------
def lidar_to_top(lidar):
    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
#    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
#    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qxs=((pxs-TOP_X_MIN)/TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)/TOP_Y_DIVISION).astype(np.int32)
    #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs=(pzs-TOP_Z_MIN)/TOP_Z_DIVISION
    quantized = np.dstack((qxs,qys,qzs,prs)).squeeze()

#    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
#    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
#    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)

    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 2

    #print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    #if 1:  #new method
    for x in range(Xn):
        ix  = np.where(quantized[:,0]==x)
        quantized_x = quantized[ix]
        if len(quantized_x) == 0 : continue
        yy = -x

        for y in range(Yn):
            iy  = np.where(quantized_x[:,1]==y)
            quantized_xy = quantized_x[iy]
            count = len(quantized_xy)
            if  count==0 : continue
            xx = -y

            top[yy,xx,Zn+1] = min(1, np.log(count+1)/math.log(32))
            max_height_point = np.argmax(quantized_xy[:,2])
            top[yy,xx,Zn]=quantized_xy[max_height_point, 3]

            for z in range(Zn):
#                iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
                iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<z+1))
                quantized_xyz = quantized_xy[iz]
                if len(quantized_xyz) == 0 : continue
                zz = z

                #height per slice
                max_height = max(0,np.max(quantized_xyz[:,2])-z)
                top[yy,xx,zz]=max_height
    return top

# -------------------- 1. SETTING PARAMETERS HERE !!! ----------------------------
# initial setting for  Kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000004.bin
TOP_X_MIN =0  
TOP_X_MAX =40
TOP_Y_MIN =-20  
TOP_Y_MAX =20
TOP_Z_MIN =-2
TOP_Z_MAX = 1.0
TOP_X_DIVISION = 0.1
TOP_Y_DIVISION = 0.1
TOP_Z_DIVISION = 0.4

# setting for DiDi from xuefung
TOP_X_MIN =-45   #-15
TOP_X_MAX =45	#45
TOP_Y_MIN =-10  #-30
TOP_Y_MAX =10	#30
TOP_Z_MIN =-3
TOP_Z_MAX =1.0
TOP_X_DIVISION = 0.2 #0.2
TOP_Y_DIVISION = 0.2 #0.2
TOP_Z_DIVISION = 1 #0.5

# setting for DiDi from xuefung  (weird slit appears in original Python version)
#TOP_X_MIN =-45  
#TOP_X_MAX =45	
#TOP_Y_MIN =-10 
#TOP_Y_MAX =10	
#TOP_Z_MIN =-3
#TOP_Z_MAX =1.0
#TOP_X_DIVISION = 5 
#TOP_Y_DIVISION = 5 
#TOP_Z_DIVISION = 1 


# Calculate map size and pack parameters for top view and front view map (DON'T CHANGE THIS !)
Xn = int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)
Yn = int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)
Zn = int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)  

top_paras = (TOP_X_MIN, TOP_X_MAX, TOP_Y_MIN, TOP_Y_MAX, TOP_Z_MIN, TOP_Z_MAX, TOP_X_DIVISION, TOP_Y_DIVISION, TOP_Z_DIVISION, Xn, Yn, Zn)
#------------------------------------------------------------------------------


#------------------- 2. SET SOURCE RAW DATA TO BE PROCESSED --------------------
# load lidar raw data  (presumed raw data dimension : num x 4)    
#raw = np.load("raw_kitti_2011_09_26_0005_0000000004.npy")
raw = np.load("raw_0.npy")
raw = np.load("./lidar_dumps/00000.npy")
num = raw.shape[0]  # DON'T CHANGE THIS !
# num : number of points in one lidar frame
# 4 : total channels of single point (x, y, z, intensity)


# top view, front view data structure for saving processed maps (initialized with 1's)
#top_flip = np.ones((Xn, Yn, Zn+2), dtype = np.double) 	# DON'T CHANGE THIS !
top_flip = np.ones((Xn, Yn, Zn+2), dtype = np.float32) 	# DON'T CHANGE THIS !
# top-view maps : Zn height maps + 1 density map + 1 intensity map
# top-view map size : Xn * Yn
# ------------------------------------------------------------------------------


#------------------- 3. CREATE TOP VIEW AND FRONT VIEW MAPS --------------------
#import time
#tStart = time.time()

from createTopMaps import *
createTopMaps(raw, num, top_flip, top_paras, './LidarTopPreprocess.so')
top = np.flipud(np.fliplr(top_flip))

#tEnd = time.time()
#print("It takes %f sec" % (tEnd-tStart))
# ------------------------------------------------------------------------------


#------------------- 4. SHOW TOP VIEW MAPS (optional) -----------
import matplotlib.pyplot as plt
# SHOW TOP VIEW MAPS OF C VERSION (optional)
map_num = len(top[0][0])
plt.figure('C version - top view')
for i in range(map_num):
	plt.subplot(1, map_num, i+1)
	plt.imshow(top[:,:,i])
	plt.gray()

# ------------------ 5. Simple value range comparison between Python version and Python_To_C version (optional) ---
python_top=lidar_to_top(raw)

# SHOW TOP VIEW MAPS OF PYTHON VERSION (optional)
map_num = len(python_top[0][0])
plt.figure('Python version - top view')
for i in range(map_num):
	plt.subplot(1, map_num, i+1)
	plt.imshow(python_top[:,:,i])
	plt.gray()
plt.show()

for i in range(map_num-2):
	print('Top view layer ', i,':')
	print('- max in    C   ver:',max(top[:,:,i].flatten()) )
	print('- max in Python ver:',max(python_top[:,:,i].flatten()) )
	print('- min in    C   ver:',min(top[:,:,i].flatten()) )
	print('- min in Python ver:',min(python_top[:,:,i].flatten()) )

print('Top view layer ', map_num-2,' (density map) :')
print('- max in    C   ver:',max(top[:,:,map_num-2].flatten()) )
print('- max in Python ver:',max(python_top[:,:,map_num-2].flatten()) )
print('- min in    C   ver:',min(top[:,:,map_num-2].flatten()) )
print('- min in Python ver:',min(python_top[:,:,map_num-2].flatten()) )
print('Top view layer ', map_num-1,' (intensity map) :')
print('- max in    C   ver:',max(top[:,:,map_num-1].flatten()) )
print('- max in Python ver:',max(python_top[:,:,map_num-1].flatten()) )
print('- min in    C   ver:',min(top[:,:,map_num-1].flatten()) )
print('- min in Python ver:',min(python_top[:,:,map_num-1].flatten()) )

# PRINT DIFFERENT VALUES BETWEEN PYTHON AND C VERSION (optional)
#map_num = len(python_top[0][0])
#for i in range(map_num):
#	for j in zip(top[:,:,i].flatten(),python_top[:,:,i].flatten()):
#		if (abs(j[0]-j[1])>0.01):
#			print(j[0],',',j[1])






