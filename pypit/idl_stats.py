#!/usr/bin/env python
# encoding: utf-8
"""
idl_stats.py
This package reproduces some of the more useful
IDL statistical routines in python.
Created by Jonathan Foster on 2010-12-03.
"""

import sys
import os
import numpy as np
import scipy.signal

# from Djs_IterStat import *
# FMean, FSig, FMedian, FMask = Djs_Iterstat(10+cos(frange(10000)/1000),RejVal=-1, SigRej=1.3)
# mplot.plot(FMask)
#
############################################################


def histOutline(dataIn, binsIn=None):
    """
    Make a histogram that can be plotted with plot() so that
    the histogram just has the outline rather than bars as it
    usually does.
    """
    if (binsIn == None):
        (en, eb) = np.histogram(dataIn)
        binsIn = eb
    else:
        (en, eb) = np.histogram(dataIn, bins=binsIn)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(eb) * 2 + 2, dtype=float)
    data = np.zeros(len(eb) * 2 + 2, dtype=float)
    for bb in range(len(binsIn) - 1):
        bins[2 * bb + 1] = binsIn[bb]
        bins[2 * bb + 2] = binsIn[bb] + stepSize
        data[2 * bb + 1] = en[bb]
        data[2 * bb + 2] = en[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)


def djs_iterstat(InputArr, SigRej=2.0, MaxIter=10, Mask=0, \
                 Max='', Min='', RejVal='', BinData=0):
    NGood = InputArr.size
    ArrShape = InputArr.shape
    if NGood == 0:
        print
        'No data points given'
        return 0, 0, 0, 0, 0
    if NGood == 1:
        print
        'Only one data point; cannot compute stats'
        return 0, 0, 0, 0, 0

    # Determine Max and Min
    if Max == '':
        Max = InputArr.max()
    if Min == '':
        Min = InputArr.min()

    if np.unique(InputArr).size == 1:
        return 0, 0, 0, 0, 0

    Mask = np.zeros(ArrShape, dtype='byte') + 1
    # Reject those above Max and those below Min
    Mask[InputArr > Max] = 0
    Mask[InputArr < Min] = 0
    if RejVal != '':  Mask[InputArr == RejVal] = 0
    FMean = np.sum(1. * InputArr * Mask) / NGood
    FSig = np.sqrt(np.sum((1. * InputArr - FMean) ** 2 * Mask) / (NGood - 1))

    NLast = -1
    Iter = 0
    NGood = np.sum(Mask)
    if NGood < 2:
        return -1, -1, -1, -1, -1

    while (Iter < MaxIter) and (NLast != NGood) and (NGood >= 2):

        LoVal = FMean - SigRej * FSig
        HiVal = FMean + SigRej * FSig
        NLast = NGood

        Mask[InputArr < LoVal] = 0
        Mask[InputArr > HiVal] = 0
        NGood = np.sum(Mask)

        if NGood >= 2:
            FMean = np.sum(1. * InputArr * Mask) / NGood
            FSig = np.sqrt(np.sum((1. * InputArr - FMean) ** 2 * Mask) / (NGood - 1))
            SaveMask = Mask.copy()
        else:
            SaveMask = Mask.copy()

        Iter = Iter + 1
    if np.sum(SaveMask) > 2:
        FMedian = np.median(InputArr[SaveMask == 1])
        if BinData == 1:
            HRange = InputArr[SaveMask == 1].max() - InputArr[SaveMask == 1].min()
            bins_In = np.arange(HRange) + InputArr[SaveMask == 1].min()
            Bins, N = histOutline(InputArr[SaveMask == 1], binsIn=bins_In)
            FMode = Bins[(np.where(N == N.max()))[0]].mean()
        else:
            FMode = 0
    else:
        FMedian = FMean
        FMode = FMean

    return FMean, FSig, FMedian, FMode, SaveMask


# This is another version of iterstat I found online. The other one looks superior
def iterstat(image,sigrej = 3.0,maxiter=3):
    """ Adapted to python from djs_iterstat.pro by Schlegel,Hogg,
    Eisenstein and Rosolowsky.
    Computes the mean and sigma of data with iterative sigma.
    Useful for excluding bad edges on sigma and also real signal.
    Verified consistent with IDL code on 2010-12-06
    IDL> djs_iterstat,[0,2,3,4,4,4,4,4,4,4,4,4,5000],mean=mean,sigma=sigma
    IDL> print,mean,sigma
          3.41667      1.24011
    In [38]: idl_stats.iterstat(np.array([0,2,3,4,4,4,4,4,4,4,4,4,5000]))
    Out[38]: (3.4166666666666665, 1.2401124093721456)
    """

    ngood = image.size
    mask  = np.ones(ngood)
    fmean = np.sum(image*mask)/float(ngood)
    fsig  = np.sqrt(np.sum((image-fmean)**2*mask)/(ngood-1))

    iiter = 1
    nlast = -1
    while ((iiter < maxiter) and (nlast != ngood) and (ngood >= 2)):
        loval = fmean - sigrej * fsig
        hival = fmean + sigrej * fsig
        nlast = ngood

        mask = np.where ((image > loval) & (image < hival),mask,0)
        ngood = np.sum(mask)

        if (ngood >= 2):
            fmean = np.sum(image*mask)/float(ngood)
            fsig = np.sqrt(np.sum((image-fmean)**2*mask) / (ngood-1))
        iiter += iiter
    return(fmean,fsig)

def wt_moment(data,wt,errors=None,oversample=1.):
    """Adapted to python from wt_moment.pro by Erik Rosolowksy
    Verified to be consistent with that code on 2010-12-03
    In [33]: wm.wt_moment([1,2,3,4],[2,1,1,1],errors=[0.5,0.5,0.25,0.25],oversample=2.)
    Out[33]:
    {'errmn': 0.22135943621178658,
     'errsd': 0.10076910123994302,
     'mean': 2.2000000000000002,
     'stdev': 1.16619037896906}
    IDL> ya = wt_moment([1,2,3,4],[2,1,1,1],errors=[0.5,0.5,0.25,0.25],oversample=2)
    IDL> help,ya,/struct
    ** Structure <1db9344>, 4 tags, length=16, data length=16, refs=1:
       MEAN            FLOAT           2.20000
       STDEV           FLOAT           1.16619
       ERRMN           FLOAT          0.221359
       ERRSD           FLOAT          0.100769
            """

    wt = np.array(wt,dtype='float64')
    data = np.array(data,dtype='float64')
    errors = np.array(errors,dtype='float64')

    osf = np.sqrt(oversample) #Note this doesn't change anything for osf=1
    tot = np.sum(wt)
    mean = np.sum(wt*data)/tot
    stdev = np.sqrt(np.abs(np.sum((data-mean)**2*wt)/tot))
    if errors.any():
        mean_err = np.sqrt(np.sum(((tot*data-np.sum(wt*data))/
                                (tot**2))**2*errors**2))*osf
        sig2err = np.sqrt(np.sum(((tot*(data-mean)**2-
                                np.sum(wt*(data-mean)**2))/tot**2)**2*errors**2)+
                                (2*np.sum(wt*(data-mean))/tot)**2*mean_err**2)
        sd_err = 1./(2*stdev)*sig2err*osf
        return({"mean":mean,"stdev":stdev,"errmn":mean_err,"errsd":sd_err})
    else:
        return({"mean":mean,"stdev":stdev})


def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
            size n. The optional keyword argument ny allows for a different
            size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im,g, mode='same')
    return(improc)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal."""

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError('Window is not of flat hanning hamming bartlett or blackman')


    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
                                            #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]



def main():
    pass

if __name__ == '__main__':
    main()