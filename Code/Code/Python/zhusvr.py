from numpy import abs, ones
import scipy
import matplotlib.pyplot as plt
from sklearn import svm

class svr:
    def __init__(self, x, y, sigma, C, epsilon=0.001, weights=None):
        self.svr = svm.SVR(gamma=1./(2.*(sigma/12.)**2), C=C, epsilon=epsilon)
        self.svr.fit(x.reshape([-1,1]), y, sample_weight=weights)
    def __call__(self, x):
        return self.svr.predict(x.reshape([-1,1]))

class zhu:
    def __init__(self, x, y):
        from scipy.optimize import curve_fit
        def zhu(x, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15):
            from numpy import sin, cos, pi            
            T = 1.
            return ( a0
                    +a1*cos(2*pi*x/(1*T))+b1*sin(2*pi*x/(1*T))
                    +a2*cos(2*pi*x/(2*T))+b2*sin(2*pi*x/(2*T))
                    +a3*cos(2*pi*x/(3*T))+b3*sin(2*pi*x/(3*T))
                    +a4*cos(2*pi*x/(4*T))+b4*sin(2*pi*x/(4*T))
                    +a5*cos(2*pi*x/(5*T))+b5*sin(2*pi*x/(5*T))
                    +a6*cos(2*pi*x/(6*T))+b6*sin(2*pi*x/(6*T))
                    +a7*cos(2*pi*x/(7*T))+b7*sin(2*pi*x/(7*T))
                    +a8*cos(2*pi*x/(8*T))+b8*sin(2*pi*x/(8*T))
                    +a9*cos(2*pi*x/(9*T))+b9*sin(2*pi*x/(9*T))
                    +a10*cos(2*pi*x/(10*T))+b10*sin(2*pi*x/(10*T))
                    +a11*cos(2*pi*x/(11*T))+b11*sin(2*pi*x/(11*T))
                    +a12*cos(2*pi*x/(12*T))+b12*sin(2*pi*x/(12*T))
                    +a13*cos(2*pi*x/(13*T))+b13*sin(2*pi*x/(13*T))
                    +a14*cos(2*pi*x/(14*T))+b14*sin(2*pi*x/(14*T))
                    +a15*cos(2*pi*x/(0.5*T))+b15*sin(2*pi*x/(0.5*T)))
        self.zhu = zhu
        self.params, covmat = curve_fit(self.zhu, x, y)
    def __call__(self, x):
        return self.zhu(x, *self.params)

class zhusvr:
    def __init__(self, x, y, sigma, C, epsilon=0.001, weights=None):
        self.svr = svm.SVR(gamma=1./(2.*(sigma/12.)**2), C=C, epsilon=epsilon)
        self.svr.fit(x.reshape([-1,1]), y, sample_weight=weights)
    def __call__(self, x):
        return self.svr.predict(x.reshape([-1,1]))
        
def doit(dataset, sigma1, C1, sigma2, C2, weights_factor):
    dyear = dataset.dyear
    clear = dataset.data['Mask'][:,line,sample]
    x = dataset.dyear[clear]
    y = dataset.data[band][:,line,sample][clear]/100. # TOA-reflectance in %
    
    # outlier detection #
    # fit SVR and exclude all observations with residual >= res_max
    # sigma = 3 month and C=15% TOA reflectance works well
    res_max = 15.
    fsvr = svr(x, y, sigma=sigma1, C=C1, epsilon=1.)
    valid = abs(fsvr(x)-y) <= res_max 
    
    # fit Zhu
    fzhu = zhu(x[valid], y[valid])
    
    # fit SVR on residuals of Zhu; weight residuals according to the data density
    density_kernel_size = 25.
    xdensity = scipy.ndimage.filters.convolve1d(1*clear, ones(density_kernel_size), mode='constant', cval=0) #(25-1)*8/30. = 6.4 month window
#    weights_factor = 2. # stretch range is [1..1+factor]
    weights = 1+(density_kernel_size-xdensity)/density_kernel_size*weights_factor # >10/25->weigth=1; 0/25->weight=2*factor; the rest scales in between 
    yr = y[valid]-fzhu(x[valid])
    fsvr2 = svr(x[valid], yr, sigma=sigma2, C=C2, epsilon=0.001, weights=weights[clear][valid])
    
    # Zhu+SVR ensemble
    fzhusvr = lambda x: fzhu(x)+fsvr2(x) 
    
    # plot results
#    plt.figure(figsize=[10,4])
    plt.plot(x, y, 'bo', alpha=0.3)
    plt.step(dyear, xdensity, 'k-', label='data densitiy', alpha=0.3)
    plt.plot(dyear, fzhu(dyear), 'g-', label='Zhu et al.', alpha=0.7)
    plt.plot(dyear, fsvr(dyear), 'y-', label='SVR', alpha=0.7)
    plt.plot(dyear, fzhusvr(dyear), 'r-', label='SVR+Zhu', alpha=0.7, linewidth=2)
    plt.legend()
#    plt.show()



if __name__ == '__main__':
    import numpy as np
    import igarss    
    # init variables
    dataset = igarss.dataset()
    sample = int(np.random.uniform(0,999))
    line = int(np.random.uniform(0,399))
    band = 'NIR'
    # run
    
    doit(dataset, sigma1=3, C1=15, sigma2=3, C2=2, weights_factor=2.)


# combine SVR and Zhu
#==============================================================================
# wmax = 15.
# density = np.clip(p.densitytrain,0,wmax)
# wsvr = density
# wzhu = wmax-wsvr
# wsum = wsvr+wzhu
#==============================================================================

# model Zhu residuals with SVR
#xres = x[valid]
#yres = y[valid]-zhu(x[valid], *fitpars)
#dres = d[valid]

#grid search best svr model
#==============================================================================
# param_grid =  {'C': np.arange(0.1,2,0.1), 
#                'gamma': 1./(2.*(np.array([1,2,3])/12.)**2),
#                'kernel': ['rbf']},
# 
# svr = svm.SVR(epsilon=0.01)
# grid_search = sklearn.grid_search.GridSearchCV(svr, param_grid=param_grid, cv=10)
# grid_search.fit(xres.reshape([-1,1]), yres)
# svr = grid_search.best_estimator_
# print svr.gamma, svr.C
#==============================================================================
#==============================================================================
# sigma_res = 3
# sample_weight = 1+(10.-np.clip(dres, 0,10))/10.*2 # >10/25->weigth=1; 0/25->weight=2*factor; the rest scales in between 
# svr = svm.SVR(gamma=1./(2.*(sigma_res/12.)**2), C=2, epsilon=0.0)
# svr.fit(xres.reshape([-1,1]), yres, sample_weight=sample_weight)
# def svr_res(x): return svr.predict(x.reshape([-1,1]))
# 
# fbothgrid = zhu(p.xgrid, *fitpars) + svr_res(p.xgrid)
# fbothtest = zhu(p.xtest, *fitpars) + svr_res(p.xtest)
# 
# def rmse(v1,v2): return np.sqrt(np.mean((v1-v2)**2))
# rmse_zhu = rmse(p.ytest, zhu(p.xtest, *fitpars))
# rmse_svr = rmse(p.ytest, p.ftest)
# rmse_svr_zhu = rmse(p.ytest, fbothtest)
# 
# plt.figure(figsize=[10,4])
# plt.plot(x, y, 'bo', alpha=0.3)
# plt.plot(p.xtest, p.ytest, 'yo', alpha=0.3)
# plt.plot(p.xgrid, zhu(p.xgrid, *fitpars), 'g-', label='Zhu et al.', alpha=0.7)
# plt.plot(p.xgrid, p.fgrid, 'y-', label='SVR', alpha=0.7)
# plt.plot(p.xgrid, fbothgrid, 'r-', label='SVR+Zhu'.format(rmse_svr_zhu), alpha=0.7, linewidth=2)
#==============================================================================
#plt.legend()
#plt.show()



#==============================================================================
# 
# plt.figure(figsize=[10,4])
# plt.plot(xres, yres, 'bo', alpha=0.3)
# plt.plot(p.xgrid, svr_res(p.xgrid), 'r-', alpha=0.7)
# plt.show()
#==============================================================================




#==============================================================================
# 
# a0
# N = 14
# T = 365
# 
# print 'a0'
# for i_ in range(1,15):
#     i = str(i_)
#     print r'+a'+i+r'*cos(2*pi*x/('+i+'*T))+b'+i+r'*sin(2*pi*x/('+i+'*T))'
# print r'+a15*cos(2*pi*x/(0.5*T))+b15*cos(2*pi*x/(0.5*T))'
# 
# for i_ in range(1,15):
#     i = str(i_)
#     print r'a'+i+', b'+i+','
#==============================================================================
