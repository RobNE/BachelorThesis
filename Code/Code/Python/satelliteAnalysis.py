import numpy
import gdal
from sklearn import svm
from collections import OrderedDict
import copy

driver = gdal.GetDriverByName('MEM')
dataset1 = driver.CreateCopy('', gdal.Open("/Users/rellerkmann/Desktop/Bachelorarbeit/Bachelorarbeit/BachelorThesis/Code/Data/gms_sample/stack1.vrt"))
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

listOfBandTimeSeries = []

for band in xrange (0, cube1.shape[0]):
    allPixelTimeSeries = {}
    for row in xrange (offSet, offSet + blockSize):#cube.shape[1]):
        for col in range (offSet, offSet + blockSize):#cube.shape[2]):
            pixelTimeSeries = OrderedDict()
            pixelCoordinate = row, col
            allPixelTimeSeries[pixelCoordinate] = pixelTimeSeries
    listOfBandTimeSeries.append(allPixelTimeSeries)

xValues = [[] for i in range(cube1.shape[0])]
print("The empty timeSeries Creation is finished")
print("Fill the timeSeries")

for band in xrange (0, numberOfDatasets):
    allPixelTimeSeries = listOfBandTimeSeries[band]
    for sceneIndex in xrange (0, cube1.shape[0]):
        scene = cube1[sceneIndex]
        xValues[sceneIndex].append(sceneIndex)
        print ("The length of the scene (aka the count of rows):", len(scene))
        print ("The length of the scene content (aka the count of cols):", len(scene[sceneIndex]))
        for row in xrange(offSet, offSet + blockSize):
            #print ("This is the row:",scene[row])
            for col in xrange(offSet, offSet + blockSize):
                #print ("This is the col and the row: ", col, row)
                #print ("This is the value for x, y:",scene[row][col])
                pixelValue =  scene[row][col]
        
                currentPixelTimeSeries = allPixelTimeSeries[(row, col)]
                currentPixelTimeSeries[sceneIndex] = pixelValue

#print allPixelTimeSeries
print ("The number of scenes:", len(listOfBandTimeSeries))
print ("The number of pixelTimeSeriesPerScene:",len(listOfBandTimeSeries[1]))
print ("The xValues:",xValues)

print ("The analysis starts")

for band in xrange (0, numberOfDatasets):
    allPixelTimeSeries = listOfBandTimeSeries[band]
    for scene in xrange (0, cube1.shape[0]):
        for row in xrange(offSet, offSet + blockSize):
            for col in xrange(offSet, offSet + blockSize):
                #print ("This is the pos x, y:",(row, col))
                currentPixelTimeSeries = allPixelTimeSeries[(row, col)]
                #print ("The current time series:", currentPixelTimeSeries)
                yValues = []
                tempXValues = copy.deepcopy(xValues)
                for key, value in currentPixelTimeSeries.iteritems():
                    print ("This is the key:", key)
                    if (value < 16000 and value > -9999):
                        yValues.append(value)
                        print("A valid value appeared")
                    else:
                        tempXValues[key][0] = -1
                for index in xrange(len(tempXValues)-1, -1, -1):
                    if (tempXValues[index][0] == -1):
                        print("Delete date")
                        del tempXValues[index]
                #print("The yValues:",yValues)
                #print("The tempXValues:",tempXValues)
                
                if (len(tempXValues) > 0 and len(yValues) > 0):
                    svr = svm.SVR()
                    svr.fit(tempXValues, yValues)
                    print('svr-fit:')
                    print('observed:', yValues)
                    print('predicted:', svr.predict(tempXValues))
                #else:
                    #print("No sufficient number of values for a proper analysis")



print ("The analysis ended")

# SVR example
# (see also http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)


