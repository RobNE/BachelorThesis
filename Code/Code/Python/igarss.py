import imageio
import numpy as np
import gdal
from osgeo.gdalconst import *
import scipy
import scipy.ndimage
import collections
import matplotlib.pyplot as plt
from sklearn import svm
import sklearn
from matplotlib.backends.backend_pdf import PdfPages
import os
import calendar

# %config InlineBackend.figure_format = 'svg'

badTimeFlag = (105,173)
badTimeYDOY = ('2002-116','2003-295')

class dataset:
    root = r'/Users/rellerkmann/Desktop/Bachelorarbeit/Bachelorarbeit/BachelorThesis/Code/Data/gms_sample/'
    def __init__(self):
        # read meta data        
        meta = np.loadtxt(self.root+'meta.txt', skiprows=1, unpack=True)
        self.year, self.doy, self.month, self.day = meta[0:4].astype(int)
        self.julian, self.dyear = meta[4:]
        # read timeseries data
        global data 
        if 'data' in globals():
            self.data = data
        else:
            self.data = collections.OrderedDict()
           # self.data['Blue'] = self._read_cube(self.root+'ts_band1')
           # self.data['Green'] = self._read_cube(self.root+'ts_band2')
           # self.data['Red'] = read_cube(self.root+'ts_band3')
            self.data['NIR'] = imageio.read_cube(self.root+'ts_band4')
           # self.data['SWIR1'] = read_cube(self.root+'ts_band5')
           # self.data['SWIR2'] = read_cube(self.root+'ts_band7')
            self.data['FMask'] = imageio.read_cube(self.root+'ts_mask')
            self.data['FMask'][(105,173)] = 255 # mask out bad scenes
            # convert to FMask to boolean and apply buffer of size 20
            self.data['Mask'] = self.data['FMask'] <= 1 # clear land and clear water   
            self.data['Mask20'] = scipy.ndimage.filters.minimum_filter(self.data['Mask'], footprint=np.ones((1,20,20)))
            data = self.data
        self.bands, self.lines, self.samples = self.data['FMask'].shape
        
    def tv453(self, t, showMask=False, showFMask=False):
        title = 'Observation Date: {}-{} / {}-{}'.format(self.year[t], self.doy[t], self.month[t], self.day[t])
        rgb = [self.data['NIR'][t],
               self.data['SWIR1'][t],
               self.data['Red'][t]] # list of bands
        for i in range(3): rgb[i] = scipy.misc.bytescale(rgb[i], 0, 3000)
        rgb = np.dstack(rgb) # ndarray
        if showMask: rgb[...,0][self.data['Mask'][t] == False] = 255
        if showFMask: rgb[...,0][self.data['FMask'][t] > 1] = 255
        
        plt.figure(dpi=72, figsize=(10,4))
        plt.imshow(rgb, interpolation='nearest')

        plt.title(title)
        plt.axis('off')
    def pdf453(self,trange):
        pdfFilename = self.root+'landsat453.pdf'
        pp = PdfPages(pdfFilename)
        for t in trange:
            self.tv453(t)
            pp.savefig()
        pp.close()
        os.system("start "+pdfFilename)
    def plotDataAvailabilityMonthly(self):
        for imonth in range(1,13):
            bands = self.month == imonth
            nbands = bands.sum()
            nobs = np.sum(1*self.data['Mask'][bands], axis=0)
            hist, loc = np.histogram(nobs, bins=np.arange(0,nbands,nbands/20))
            plt.step(1.*loc[:-1]/nbands*100,1.*hist/nobs.size*100)
            plt.title('Data Availability inside Image\nfor '+calendar.month_name[imonth])
            plt.xlabel('clear observations [%]')
            plt.ylabel('number of pixels [%]')
            plt.show()
        
class profile():
    def __init__(self, dataset):
        self.dataset = dataset
        self.svr = svm.SVR()
        self.xgrid = dataset.dyear #np.linspace(min(dataset.dyear),max(dataset.dyear),2000)
    def _read(self, sample, line, band, test_size=0.):
        valid = self.dataset.data['Mask'][:,line,sample]
        x = self.dataset.dyear[valid]
        y = self.dataset.data[band][:,line,sample][valid]/100. # TOA-reflectance in %
        self.sample = sample
        self.line = line
        self.band = band
        self.x = x
        self.y = y
        self.xtrain, self.xtest, self.ytrain, self.ytest, self.indextrain, self.indextest = sklearn.cross_validation.train_test_split(x, y, np.arange(self.dataset.bands)[valid], test_size=test_size)
        valid[self.indextest] = 0
        self.densitytrain = scipy.ndimage.filters.convolve1d(1*valid,np.ones(25),mode='constant',cval=0) #(25-1)*8/30. = 6.4 month window
    def _setSVRParameters(self, sigma=None, C=None):
        if C is not None: 
            self.svr.C = C
        if sigma is not None:
            self.svr.gamma = 1./(2.*(sigma/12.)**2) # sigma in month
            self.sigma = sigma
    def _getSVRParameters(self):
        return self.sigma, self.svr.C
    def _svr_fit(self): 
        self.svr.epsilon=0.5
        #self.svr.fit(self.x.reshape([-1,1]), self.y)
        self.svr.fit(self.xtrain.reshape([-1,1]), self.ytrain)
    
        # approx. the model
#        self._svr_predict()
#        self.svr.epsilon=0.5
#        self.svr.fit(self.x.reshape([-1,1]), self.f)
        
    def _svr_predict(self):
        self.ftrain = self.svr.predict(self.xtrain.reshape([-1,1]))
        self.ftest = self.svr.predict(self.xtest.reshape([-1,1]))
        self.fgrid = self.svr.predict(self.xgrid.reshape([-1,1]))
        self.f = self.svr.predict(self.x.reshape([-1,1]))
        
    def _plot_xy(self):
        plt.step(self.xgrid, 1.*self.densitytrain/25.*self.y.max(), color=[0.5,0.5,0.5] )
        plt.plot(self.xtrain, self.ytrain, 'bo', alpha=0.3)
        plt.plot(self.xtest, self.ytest, 'yo', alpha=0.3)
    def _plot_xygrid(self):
        plt.plot(self.xgrid, self.fgrid, 'r-')
        plt.plot(self.xgrid[[0,-1]],[self.svr.intercept_,self.svr.intercept_], 'r:', alpha=0.3)
    def plot_yf(self, nmin=0, nmax=25):
        stratum = (self.densitytrain[self.indextest] >= nmin) * (self.densitytrain[self.indextest] <= nmax)
        if not stratum.any(): return
        plt.figure(figsize=[4,4])  
        plt.plot(self.ytest[stratum], self.ftest[stratum], 'go', alpha=0.5)
        v = np.array((0,100))
        plt.plot(v, v, 'k-')
        #residuals = self.ftest-self.ytest
        residuals = self.ftest[stratum]-self.ytest[stratum]
        residualsDist = np.percentile(residuals,(5,25,50,75,95))
        rmse = residuals.std()
      #  r, p_r = scipy.stats.stats.pearsonr(self.ftest[stratum],self.ytest[stratum])
     #   r2 = r**2
        style = ('r:','r--','r-')
        for i in range(3):
            plt.plot(v, v+residualsDist[i], style[i], alpha=0.5)
            plt.plot(v, v+residualsDist[-i-1], style[i], alpha=0.5)
        min = self.ftest.min() #np.minimum(self.ytest, self.ftest).min()
        max = self.ftest.max() #np.maximum(self.ytest, self.ftest).max()
        plt.grid(b=True, which='major', color='k', linestyle=':')
        plt.xlim(min-5, max+5)
        plt.ylim(min-5, max+5)
        plt.title('top-of-atmosphere reflectance [%]\n'
                   'number of clear obs: [{}-{}]/25\n'
                   'RMSE={:.2f}\n\n'
                   'residuals distribution\n'                   
                   '10%    25%   median  75%   90%\n'
                   '{:.2f}    {:.2f}    {:.2f}   {:.2f}    {:.2f}'.format(nmin,nmax,rmse,*residualsDist))
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.show()
    def fit(self, sample, line, band, test_size, sigma=None, C=None, ):
        self._read(sample, line, band, test_size)
        self._setSVRParameters(sigma, C)
        self._svr_fit()
        self._svr_predict()
    def plot(self, sample, line, band, test_size, sigma=None, C=None):
        self.fit(sample, line, band, test_size, sigma, C)
        # predicted and observed vs. time     
        plt.figure(figsize=[20,4])
        self._plot_xy()
        self._plot_xygrid()
        #plt.ylim(self.fgrid.min()-5, self.fgrid.max()+5)
        plt.title('Landsat {}, Pixel: {} {} \n'.format(band, sample+1, line+1)+
              r'SVR $\sigma$={} [month] $C$={} [% reflectance]'.format(sigma, C))
        plt.ylabel('top-of-atmosphere reflectance [%]')
        plt.show()
        # predicted vs. observed        
        self.plot_yf(0,5)
        self.plot_yf(6,10)
        self.plot_yf(11,25)
        
def sensitivityAnalysis(p):
    runs = 100
    n = np.array([])
    y = np.array([])
    f = np.array([])
    band = 'NIR'
    sigma = 3
    C = 15
    for i in range(runs):
        sample, line = int(np.random.uniform(0,ds.samples))-1, int(np.random.uniform(0,ds.lines))-1
        p.fit(sample, line, band, sigma, C)     
        f = np.append(f, p.ftest)
        y = np.append(y, p.ytest)
        n = np.append(n, p.densitytrain[p.indextest]) 
    
    if 1: # 
        rmses = list()
        ns = list()
        
        for nmin,nmax in zip(range(26),range(26)):#[0,5,10,15],[4,9,14,25]):

            stratum = (n >= nmin) * (n <= nmax)
            if stratum.sum() < 5: continue
            plt.figure(figsize=[8,8])
            plt.plot(y[stratum], f[stratum], 'yo', alpha=0.1)
            v = np.array((0,100))
            plt.plot(v, v, 'k-')
            #residuals = self.ftest-self.ytest
            residuals = f[stratum]-y[stratum]
            residualsDist = np.percentile(residuals,(5,25,50,75,95))
            rmse = residuals.std()
            rmses.append(rmse)
            ns.append(nmin)
          #  r, p_r = scipy.stats.stats.pearsonr(self.ftest[stratum],self.ytest[stratum])
         #   r2 = r**2
            style = ('r:','r--','r-')
            for i in range(3):
                plt.plot(v, v+residualsDist[i], style[i], alpha=0.5)
                plt.plot(v, v+residualsDist[-i-1], style[i], alpha=0.5)
            min = f.min() #np.minimum(self.ytest, self.ftest).min()
            max = f.max() #np.maximum(self.ytest, self.ftest).max()
            plt.grid(b=True, which='major', color='k', linestyle=':')
            plt.xlim(min-5, max+5)
            plt.ylim(min-5, max+5)
            plt.title('top-of-atmosphere reflectance [%]\n'
                       'number of clear obs: [{}-{}]/25\n'
                       'RMSE={:.2f}\n\n'
                       'n={}\n'
                       'residuals distribution\n'                   
                       '10%    25%   median  75%   90%\n'
                       '{:.2f}    {:.2f}    {:.2f}   {:.2f}    {:.2f}'.format(nmin,nmax,rmse,(stratum*1).sum(),*residualsDist))
            plt.xlabel('observed')
            plt.ylabel('predicted')
            plt.show()
        print 'RMSEs = ', rmses
        plt.figure(figsize=[8,8])
        plt.plot(ns, rmses, 'b-', alpha=1)
        plt.title('RMSE learning curve')        
        plt.xlabel('number of observations out of 25')
        plt.ylabel('RMSE')

def predictImage(p):
    band = 'NIR'
    sigma = 3
    C = 15
    cube = np.zeros((30,30,p.dataset.bands))
    for line in range(30):
        for sample in range(30):
          p.fit(sample, line, band, sigma, C)
          cube[line,sample,]

#ds = dataset() # create instance for global usage (not so nice, better way to do it?)
#p = profile(ds)
#p.plot(669,374,'NIR', test_size=0., sigma=3, C=15)
#sa = sensitivityAnalysis(p)
#print sa
#np.random.seed(1)
#for i in range(10):
#    x, y = int(np.random.uniform(0,ds.samples))-1, int(np.random.uniform(0,ds.lines))-1
#    p.plot(x,y,'NIR',3,15)

#ds.tv453(1,showMask=False, showFMask=True)
#ds.plotDataAvailabilityMonthly()


