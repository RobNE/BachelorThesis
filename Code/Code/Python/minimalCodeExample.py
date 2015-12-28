# training data
import numpy
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

# least squares fit example
# (see also http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.curve_fit.html)
import scipy
#def(x, m, n):return m*x+n
fitpars, covmat = scipy.optimize.curve_fit(f, x.flatten(), y)
print('\nlinear-fit:')
print('observed:', y)
print('predicted:', f(x.flatten(), *fitpars))

#svr-fit:

#('observed: ', array([ 1.,  2.,  4.,  9.]))

#('predicted:', array([ 1.69573121,  2.1       ,  3.9       ,  4.30426879]))



#linear-fit:

#('observed: ', array([ 1.,  2.,  4.,  9.]))

#('predicted:', array([ 0.1,  2.7,  5.3,  7.9]))
