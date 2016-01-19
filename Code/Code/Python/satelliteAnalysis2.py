import numpy
import gdal
from sklearn import svm
from collections import OrderedDict
import copy

driver = gdal.GetDriverByName('MEM')
dataset1 = driver.CreateCopy('', gdal.Open("/Users/rellerkmann/Desktop/Bachelorarbeit/Bachelorarbeit/BachelorThesis/Code/Data/gms_sample/stack3.vrt"))
cube1 = dataset1.ReadAsArray()
geotransform = dataset1.GetGeoTransform()
if not geotransform is None:
    print geotransform

numberOfDatasets = 1

print ("Cube read successfully")
print ("This is the shape:"),cube1.shape
print ("This is the shape of the band:", cube1[0].shape)

blockSize = 200
offSet = 2000

#Build the empty pixelTimeSeries

for bandCube in cube1:
    print (len(bandCube))
    print (bandCube[0])
    print (len(bandCube[0]))
    print (bandCube[0][0])
    print ("++++++++++++")
