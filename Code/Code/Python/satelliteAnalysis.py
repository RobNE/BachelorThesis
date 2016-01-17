import numpy
import gdal
from sklearn import svm
from collections import OrderedDict
import copy

driver = gdal.GetDriverByName('MEM')
dataset = driver.CreateCopy('', gdal.Open("/Users/rellerkmann/Desktop/Bachelorarbeit/Bachelorarbeit/BachelorThesis/Code/Data/gms_sample/stack1.vrt"))
cube = dataset.ReadAsArray()
print ("Cube read successfully")
print ("This is the shape:"),cube.shape
#print ("This is the cube:"),cube

blockSize = 1000

allPixelTimeSeries = {}
#Build the empty pixelTimeSeries
for row in xrange (0, blockSize):#cube.shape[1]):
    for col in range (0, blockSize):#cube.shape[2]):
        pixelTimeSeries = OrderedDict()
        pixelCoordinate = row, col
        allPixelTimeSeries[pixelCoordinate] = pixelTimeSeries

xValues = [[] for i in range(cube.shape[0])]

print("The empty timeSeries Creation is finished")
print("Fill the timeSeries")
for sceneIndex in xrange (0, cube.shape[0]):
    scene = cube[sceneIndex]
    xValues[sceneIndex].append(sceneIndex)
    print ("This is the scene:",scene)
    print ("The length of the scene (aka the count of rows):", len(scene))
    for row in xrange(0, blockSize):
        #print ("This is the row:",scene[row])
        #print ("Len(row):", len(scene[row]))
        #print ("The row:", row)
        for col in xrange(0, blockSize):
            #print ("This is the value for x, y:",scene[row][col])
            pixelValue =  scene[row][col]
            currentPixelTimeSeries = allPixelTimeSeries[(row, col)]
            currentPixelTimeSeries[sceneIndex] = pixelValue

#print allPixelTimeSeries
print ("The number of pixelTimeSeries:",len(allPixelTimeSeries))
print ("The xValues:",xValues)

print ("The analysis starts")
for row in xrange(0, blockSize):
    #print ("The row:", row)
    for col in xrange(0, blockSize):
        #print ("This is the pos x, y:",(row, col))
        currentPixelTimeSeries = allPixelTimeSeries[(row, col)]
        #print ("The current time series:", currentPixelTimeSeries)
        yValues = []
        tempXValues = copy.deepcopy(xValues)
        #print("The xValues:",xValues)
        for key, value in currentPixelTimeSeries.iteritems():
            if (value < 16000 and value > -9999):
                yValues.append(value)
                print("A valid value appeared")
            else:
                tempXValues[key][0] = -1
        for index in xrange(len(tempXValues)-1, -1, -1):
            if (tempXValues[index][0] == -1):
                #print("Delete date")
                del tempXValues[index]
        #print("The yValues:",yValues)
        #print("The tempXValues:",tempXValues)

        if (len(tempXValues) > 0):
            svr = svm.SVR()
            svr.fit(xValues, yValues)
            print('svr-fit:')
            print('observed:', yValues)
            print('predicted:', svr.predict(xValues))
        #else:

            #print("No sufficient number of values for a proper analysis")


# SVR example
# (see also http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)


