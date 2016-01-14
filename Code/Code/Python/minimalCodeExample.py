import numpy
import gdal

driver = gdal.GetDriverByName('MEM')
dataset = driver.CreateCopy('', gdal.Open("/Users/rellerkmann/Desktop/Bachelorarbeit/Bachelorarbeit/BachelorThesis/Code/Data/gms_sample/stack.vrt"))
cube = dataset.ReadAsArray()
print ("Cube read successfully")
print cube

x = numpy.array([[0.], [1.], [2.], [3.], [4.], [5.]])
y = numpy.array([1.,2.,4.,9., 7., 5.])

# SVR example
# (see also http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
from sklearn import svm
svr = svm.SVR()
svr.fit(x, y)
print('svr-fit:')
print('observed:', y)
print('predicted:', svr.predict(x))
