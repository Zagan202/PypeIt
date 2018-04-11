from __future__ import (print_function, absolute_import, division, unicode_literals)

import time
import inspect
import copy
from collections import Counter

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, font_manager

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Gaussian1DKernel

from pypit import msgs
from pypit import arqa
from pypit import arplot
from pypit import ararc
from pypit import arutils
from pypit import arpca
from pypit import arparse as settings
from pypit.filter import BoxcarFilter
from pypit import arspecobj
from pypit import ardebug as debugger
from pypit import ginga

from pypit import arcyarc
from pypit import arcytrace

try:
    ustr = unicode
except NameError:
    ustr = str


def assign_slits(binarr, edgearr, ednum=100000, lor=-1):
    """This routine will trace the locations of the slit edges

    Parameters
    ----------
    binarr : numpy ndarray
      Calibration frame that will be used to identify slit traces (in most cases, the slit edge)
    edgearr : numpy ndarray
      An array of negative/positive numbers (left/right edges respectively) and zeros (no edge)
    ednum : int
      A dummy number given to define slit edges
    lor : int (-1 or +1)
      A flag that indicates if the left edge (-1) or right edge (+1) should be assigned

    Returns
    -------
    edgearr : numpy ndarray
      An array of negative/positive numbers (left/right edges respectively) and zeros (no edge)
    """

    if lor == -1:
        lortxt = "left"
    else:
        lortxt = "right"
    outitnm = 1
    oldedgearr = edgearr.copy()
    prevedgearr = edgearr.copy()
    while True:
        msgs.prindent("Outer {0:s} edge loop, Iteration {1:d}".format(lortxt, outitnm))
        labnum = lor*ednum
        itnm = 0
        nslit = 0
        cmnold = None
        firstpass = True
        while True:
            edgehist = np.zeros(binarr.shape[1]*2, dtype=np.int)
            itnm += 1
            # Locate edges relative to the most common edge
            if lor == -1:
                wl = np.where(edgearr <= -2*ednum)
            else:
                wl = np.where(edgearr >= 2*ednum)
            if wl[0].size == 0:
                break
            cl = Counter(edg for edg in edgearr[wl])
            comml = cl.most_common(1)
            ww = np.where(edgearr == comml[0][0])
            if not firstpass:
                if (cmnold[0] == comml[0][0]) and (cmnold[1] == comml[0][1]):
                    # Nothing has changed since the previous iteration, so end the loop
                    break
                if comml[0][1] < binarr.shape[0]/100.0:
                    # Now considering an edge that spans less than 1 per cent of the detector ---> insignificant
                    break
            cmnold = comml[0]
            # Extract just these elements
            tedgearr = edgearr[ww[0], :]
            # Calculate the offset
            offs = binarr.shape[1]
            # Add these into edgehist
            edgehist[offs] = np.sum(binarr[ww])#ww[0].size
            # And a fudge to give this edge detection some width (for peak finding, below)
            edgehist[offs-1] = 1 + ww[0].size/2
            edgehist[offs+1] = 1 + ww[0].size/2
            # Find the difference between unknown edges
            if lor == -1:
                www = np.where(tedgearr <= -2*ednum)
            else:
                www = np.where(tedgearr >= 2*ednum)
            if www[0].size == 0:
                break
            shft = www[1] - ww[1][www[0]]  # Calculate the shift between right edges
            shft += offs  # Apply the offset to the edgehist arr
            #np.add.at(edgehist, shft, 1)
            np.add.at(edgehist, shft, binarr[ww[0], :][www])
            # Smooth the histogram with a Gaussian of standard deviation 1 pixel to reduce noise
            smedgehist = ndimage.uniform_filter1d(edgehist, 3)
            # Identify peaks (which indicate the locations of the right slit edges)
            arrlfr = smedgehist[0:-4]
            arrlft = smedgehist[1:-3]
            arrcen = smedgehist[2:-2]
            arrrgt = smedgehist[3:-1]
            arrrfr = smedgehist[4:]
            wpk = np.where((arrcen >= arrlft) & (arrcen > arrrgt) &  # Exactly one of these should be >=
                           ((arrlft > arrlfr) | (arrrgt > arrrfr)))[0]
            if wpk.size == 0:
                # No more peaks
                break
            if wpk.size != 1:
                wpkmsk = prune_peaks(smedgehist, wpk, np.where(wpk+2 == offs)[0][0])
                wpk = wpk[np.where(wpkmsk == 1)]
            if wpk.size == 0:
                # After pruning, there are no more peaks
                break
            pks = wpk+2  # Shifted by 2 because of the peak finding algorithm above
#            print('calling find_peak_limits')
#            t = time.clock()
#            _pedges = arcytrace.find_peak_limits(smedgehist, pks)
#            print('Old find_peak_limits: {0} seconds'.format(time.clock() - t))
#            t = time.clock()
            pedges = new_find_peak_limits(smedgehist, pks)
#            print('New find_peak_limits: {0} seconds'.format(time.clock() - t))
#            assert np.sum(_pedges != pedges) == 0, \
#                    'Difference between old and new find_peak_limits'

            if np.all(pedges[:, 1]-pedges[:, 0] == 0):
                # Remaining peaks have no width
                break
            if msgs._debug['trace']:
                plt.clf()
                plt.plot(arrcen, 'k-', drawstyle='steps')
                plt.plot(wpk, np.zeros(wpk.size), 'ro')
                plt.show()
                debugger.set_trace()
            # Label all edge ids (in the original edgearr) that are located in each peak with the same number
            for ii in range(pks.size):
                shbad = np.zeros(edgearr.shape)
                wp = np.where((shft >= pedges[ii, 0]) & (shft <= pedges[ii, 1]))
                vals = np.unique(tedgearr[(www[0][wp], www[1][wp])])
                # Fit the edge detections in this edge and calculate the offsets
                strev = "np.where("
                for vv in vals:
                    strev += "(edgearr=={0:d})|".format(vv)
                strev = strev[:-1] + ")"
                widx = eval(strev)
                if widx[0].size < 2*settings.argflag['trace']['slits']['polyorder']:
                    continue
                badmsk, fitcof = arutils.robust_polyfit(widx[0], widx[1],
                                                        settings.argflag['trace']['slits']['polyorder'],
                                                        function=settings.argflag['trace']['slits']['function'],
                                                        minv=0, maxv=binarr.shape[0]-1)
                shbad[widx] = badmsk
                smallhist = np.zeros(101, dtype=np.int)
                meddiff = np.zeros(vals.size)
                for vv in range(vals.size):
                    widx = np.where((edgearr == vals[vv]) & (shbad == 0))
                    if widx[0].size == 0:
                        # These pixels were deemed to be bad
                        continue
                    diff = widx[1] - arutils.func_val(fitcof, widx[0],
                                                      settings.argflag['trace']['slits']['function'],
                                                      minv=0, maxv=binarr.shape[0]-1)
                    diff = 50 + np.round(diff).astype(np.int)
                    np.add.at(smallhist, diff, 1)
                    meddiff[vv] = np.median(diff)
                # Find the peaks of this distribution
                wspk = np.where((smallhist[1:-1] >= smallhist[2:]) & (smallhist[1:-1] > smallhist[:-2]))[0]
                wspk += 1  # Add one here to account for peak finding
                if msgs._debug['trace'] and False:
                    plt.clf()
                    plt.plot(smallhist, 'k-', drawstyle='steps')
                    plt.show()

                for pp in range(wspk.size):  # For all small peaks identified
                    for vv in range(vals.size):
                        if lor == -1 and vals[vv] > -2*ednum:
                            continue
                        elif lor == 1 and vals[vv] < 2*ednum:
                            continue
                        # Make sure this value is within 1 pixel of the peak
                        if meddiff[vv] < wspk[pp]-1:
                            continue
                        if meddiff[vv] > wspk[pp]+1:
                            continue
                        edgearr[np.where(edgearr == vals[vv])] = labnum
                        meddiff[vv] = -1  # Flag this val as done
                    labnum += lor*1
                # Find any vals that weren't applied
                for vv in range(vals.size):
                    if meddiff[vv] == -1:
                        continue
                    edgearr[np.where(edgearr == vals[vv])] = 0
            nslit += pks.size
            msgs.prindent("  Inner loop, Iteration {0:d}, {1:d} {2:s} edges assigned ({3:d} total)".format(itnm, pks.size, lortxt, nslit))
            firstpass = False
        outitnm += 1
        if lor == -1:
            edgearr[np.where(edgearr <= -2*ednum)] = 0
        else:
            edgearr[np.where(edgearr >= 2*ednum)] = 0
        if np.array_equal(edgearr, oldedgearr) or np.array_equal(edgearr, prevedgearr):
            break
        elif outitnm > 10:
            msgs.warn("Edge assignment may not have converged")
            msgs.info("Please check the slit edge traces")
            #debugger.set_trace()
            break
        else:
            oldedgearr = prevedgearr.copy()
            prevedgearr = edgearr.copy()
            if lor == -1:
                edgearr[np.where(edgearr <= -ednum)] -= ednum
            else:
                edgearr[np.where(edgearr >= ednum)] += ednum
    # Ignore any order detections that weren't identified in the loop
    if lor == -1:
        edgearr[np.where(edgearr <= -2*ednum)] = 0
    else:
        edgearr[np.where(edgearr >= 2*ednum)] = 0
    # Sort vals by increasing spatial position on the detector
    # First, determine the model for the most common slit edge
    if lor == -1:
        wcm = np.where(edgearr <= -ednum)
    else:
        wcm = np.where(edgearr >= ednum)
    if wcm[0].size != 0:
        cntr = Counter(edg for edg in edgearr[wcm])
        commn = cntr.most_common(1)
        wedx, wedy = np.where(edgearr == commn[0][0])
        msk, cf = arutils.robust_polyfit(wedx, wedy,
                                         settings.argflag['trace']['slits']['polyorder'],
                                         function=settings.argflag['trace']['slits']['function'],
                                         minv=0, maxv=binarr.shape[0]-1)
        cenmodl = arutils.func_val(cf, np.arange(binarr.shape[0]),
                                   settings.argflag['trace']['slits']['function'],
                                   minv=0, maxv=binarr.shape[0]-1)
        if lor == -1:
            vals = np.unique(edgearr[np.where(edgearr < 0)])
        else:
            vals = np.unique(edgearr[np.where(edgearr > 0)])
        diffarr = np.zeros(vals.size)
        diffstd = 0.0
        for jj in range(vals.size):
            wedx, wedy = np.where(edgearr == vals[jj])
            diffarr[jj] = np.mean(wedy-cenmodl[wedx])
            diffstd += np.std(wedy-cenmodl[wedx])
        diffstd /= vals.size
        dasrt = np.argsort(diffarr)
        # Relabel the edges from left to right
        edgearr[wcm] += lor*ednum
        labnum = lor*ednum
        diffarrsrt = diffarr[dasrt]
        diffs = diffarrsrt[1:] - diffarrsrt[:-1]
        for jj in range(vals.size):
            wrplc = np.where(edgearr == lor*ednum + vals[dasrt[jj]])
            edgearr[wrplc] = labnum
            if jj != vals.size-1:
                if diffs[jj] > 3.0*diffstd:
                    # The next edge must be a different edge
                    labnum += lor*1
    return


def expand_slits(slf, mstrace, det, ordcen, extord):
    """
    This routine will traces the locations of the slit edges

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    mstrace: numpy ndarray
      Calibration frame that will be used to identify slit edges
    det : int
      Index of the detector
    ordcen : ndarray
      An array providing the physical pixel locations corresponding to the slit centres
    extord : ndarray
      A boolean mask indicating if an order was extrapolated (True = extrapolated)

    Returns
    -------
    lordloc : ndarray
      Locations of the left slit edges (in physical pixel coordinates)
    rordloc : ndarray
      Locations of the right slit edges (in physical pixel coordinates)
    """

    # Calculate the pixel locations of th eorder edges
    pixcen = phys_to_pix(ordcen, slf._pixlocn[det - 1], 1)
    msgs.info("Expanding slit traces to slit edges")
#    exit()
    mordwid, pordwid = arcytrace.expand_slits(mstrace, pixcen, extord.astype(np.int))
    # Fit a function for the difference between left edge and the centre trace
    ldiff_coeff, ldiff_fit = arutils.polyfitter2d(mordwid, mask=-1,
                                                  order=settings.argflag['trace']['slits']['diffpolyorder'])
    # Fit a function for the difference between left edge and the centre trace
    rdiff_coeff, rdiff_fit = arutils.polyfitter2d(pordwid, mask=-1,
                                                  order=settings.argflag['trace']['slits']['diffpolyorder'])
    lordloc = ordcen - ldiff_fit.T
    rordloc = ordcen + rdiff_fit.T
    return lordloc, rordloc


def trace_objbg_image(slf, det, sciframe, slitn, objreg, bgreg, trim=2, triml=None, trimr=None):
    """ Creates an image with weights corresponding to object or background pixels.

    Each weight can take any floating point value from 0 to 1 (inclusive). For the
    rec_obj_img, a weight of 1 means that the pixel is fully contained within the
    object region, and 0 means that the pixel is fully contained within the
    background region. The opposite is true for the rec_bg_img array. A pixel that
    is on the border of object/background is assigned a value between 0 and 1, based
    on the percentage overlap with the object/background regions.

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    det : int
      Index of the detector
    sciframe: numpy ndarray
      Science frame
    slitn : int
      Slit (or order) number
    objreg : list
      A two element list containing arrays that indicate the left and right
      pixels of the object locations for each object.
    bgreg : list
      A two element list containing arrays that indicate the background regions
      around each object (i.e. [bgleft, bgright]). Each array is 2D and contains
      a list of pixels for each object. The size of each array is [npix, nobj].
    trim : int (optional)
      Number of pixels to trim from the left and right slit edges.
      To separately specify how many pixels to trim from the left
      and right slit edges, use triml and trimr.
    triml : int (optional)
      Number of pixels to trim from the left slit edge
    trimr : int (optional)
      Number of pixels to trim from the right slit edge

    Returns
    -------
    rec_obj_img : ndarray
      An image containing weights to be used for the object
    rec_bg_img : ndarray
      An image containing weights to be used for the background
    """
    # Get the number of objects in the slit
    nobj = len(objreg[0])
    if triml is None:
        triml = trim
    if trimr is None:
        trimr = trim
    npix = int(slf._pixwid[det-1][slitn] - triml - trimr)
    # Make an image of pixel weights for each object
    xint = np.linspace(0.0, 1.0, sciframe.shape[0])
    yint = np.linspace(0.0, 1.0, npix)
    yint = np.append(-yint[1], np.append(yint, 2.0-yint[-2]))
    msgs.info("Creating an image weighted by object pixels")
    rec_obj_img = np.zeros((sciframe.shape[0], sciframe.shape[1], nobj))
    for o in range(nobj):
        #msgs.info("obj {:d}".format(o))
        obj = np.zeros(npix)
        obj[objreg[0][o]:objreg[1][o]+1] = 1
        scitmp = np.append(0.0, np.append(obj, 0.0))
        objframe = scitmp.reshape(1, -1).repeat(sciframe.shape[0], axis=0)
        objspl = interp.RectBivariateSpline(xint, yint, objframe, bbox=[0.0, 1.0, yint.min(), yint.max()], kx=1, ky=1, s=0)
        xx, yy = np.meshgrid(np.linspace(0, 1.0, sciframe.shape[0]), np.arange(0, sciframe.shape[1]), indexing='ij')
        lo = (slf._lordloc[det-1][:, slitn]+triml).reshape((-1, 1))
        vv = ((yy-lo)/(npix-1.0)).flatten()
        xx = xx.flatten()
        wf = np.where((vv >= yint[0]) & (vv <= yint[-1]))
        rec_obj_arr = objspl.ev(xx[wf], vv[wf])
        idxarr = np.zeros(sciframe.shape).flatten()
        idxarr[wf] = 1
        idxarr = idxarr.reshape(sciframe.shape)
        rec_img = np.zeros_like(sciframe)
        rec_img[np.where(idxarr == 1)] = rec_obj_arr
        rec_obj_img[:, :, o] = rec_img.copy()
    # Make an image of pixel weights for the background region of each object
    msgs.info("Creating an image weighted by background pixels")
    rec_bg_img = np.zeros((sciframe.shape[0], sciframe.shape[1], nobj))
    for o in range(nobj):
        #msgs.info("bkg {:d}".format(o))
        backtmp = np.append(0.0, np.append(bgreg[0][:, o] + bgreg[1][:, o], 0.0))
        bckframe = backtmp.reshape(1, -1).repeat(sciframe.shape[0], axis=0)
        bckspl = interp.RectBivariateSpline(xint, yint, bckframe, bbox=[0.0, 1.0, yint.min(), yint.max()], kx=1, ky=1, s=0)
        xx, yy = np.meshgrid(np.linspace(0, 1.0, sciframe.shape[0]), np.arange(0, sciframe.shape[1]), indexing='ij')
        lo = (slf._lordloc[det-1][:, slitn]+triml).reshape((-1, 1))
        vv = ((yy-lo)/(npix-1.0)).flatten()
        xx = xx.flatten()
        wf = np.where((vv >= yint[0]) & (vv <= yint[-1]))
        rec_bg_arr = bckspl.ev(xx[wf], vv[wf])
        idxarr = np.zeros(sciframe.shape).flatten()
        idxarr[wf] = 1
        idxarr = idxarr.reshape(sciframe.shape)
        rec_img = np.zeros_like(sciframe)
        rec_img[np.where(idxarr == 1)] = rec_bg_arr
        rec_bg_img[:, :, o] = rec_img.copy()
    return rec_obj_img, rec_bg_img


def trace_object_dict(nobj, traces, object=None, background=None, params=None, tracelist=None):
    """ Creates a list of dictionaries, which contain the object traces in each slit

    Parameters
    ----------
    nobj : int
      Number of objects in this slit
    traces : numpy ndarray
      the trace of each object in this slit
    object: numpy ndarray (optional)
      An image containing weights to be used for the object.
    background : numpy ndarray (optional)
      An image containing weights to be used for the background
    params : dict
      A dictionary containing some of the important parameters used in
      the object tracing.
    tracelist : list of dict
      A list containing a trace dictionary for each slit

    To save memory, the object and background images for multiple slits
    can be stored in a single object image and single background image.
    In this case, you must only set object and background for the zeroth
    slit (with 'None' set for all other slits). In this case, the object
    and background images must be set according to the slit locations
    assigned in the slf._slitpix variable.

    Returns
    -------
    tracelist : list of dict
      A list containing a trace dictionary for each slit
    """
    # Create a dictionary with all the properties of the object traces in this slit
    newdict = dict({})
    newdict['nobj'] = nobj
    newdict['traces'] = traces
    newdict['object'] = object
    newdict['params'] = params
    newdict['background'] = background
    if tracelist is None:
        tracelist = []
    tracelist.append(newdict)
    return tracelist


def trace_object(slf, det, sciframe, varframe, crmask, trim=2,
                 triml=None, trimr=None, sigmin=2.0, bgreg=None,
                 maskval=-999999.9, slitn=0, doqa=True,
                 xedge=0.03, tracedict=None, standard=False, debug=False):
    """ Finds objects, and traces their location on the detector

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    det : int
      Index of the detector
    sciframe: numpy ndarray
      Science frame
    varframe: numpy ndarray
      Variance frame
    crmask: numpy ndarray
      Mask or cosmic rays
    slitn : int
      Slit (or order) number
    trim : int (optional)
      Number of pixels to trim from the left and right slit edges.
      To separately specify how many pixels to trim from the left
      and right slit edges, use triml and trimr.
    triml : int (optional)
      Number of pixels to trim from the left slit edge
    trimr : int (optional)
      Number of pixels to trim from the right slit edge
    sigmin : float
      Significance threshold to eliminate CRs
    bgreg : int
      Number of pixels to use in the background region
    maskval : float
      Placeholder value used to mask pixels
    xedge : float
      Trim objects within xedge % of the slit edge
    doqa : bool
      Should QA be output?
    tracedict : list of dict
      A list containing a trace dictionary for each slit

    Returns
    -------
    tracedict : dict
      A dictionary containing the object trace information
    """
    # Find the trace of each object
    tracefunc = settings.argflag['trace']['object']['function']
    traceorder = settings.argflag['trace']['object']['order']
    if triml is None:
        triml = trim
    if trimr is None:
        trimr = trim
    npix = int(slf._pixwid[det-1][slitn] - triml - trimr)
    if bgreg is None:
        bgreg = npix
    # Setup
    if 'xedge' in settings.argflag['trace']['object'].keys():
        xedge = settings.argflag['trace']['object']['xedge']
    # Store the trace parameters
    tracepar = dict(smthby=7, rejhilo=1, bgreg=bgreg, triml=triml, trimr=trimr,
                    tracefunc=tracefunc, traceorder=traceorder, xedge=xedge)
    # Interpolate the science array onto a new grid (with constant spatial slit length)
    msgs.info("Rectifying science frame")
    xint = np.linspace(0.0, 1.0, sciframe.shape[0])
    yint = np.linspace(0.0, 1.0, sciframe.shape[1])
    scispl = interp.RectBivariateSpline(xint, yint, sciframe, bbox=[0.0, 1.0, 0.0, 1.0], kx=1, ky=1, s=0)
    varspl = interp.RectBivariateSpline(xint, yint, varframe, bbox=[0.0, 1.0, 0.0, 1.0], kx=1, ky=1, s=0)
    crmspl = interp.RectBivariateSpline(xint, yint, crmask, bbox=[0.0, 1.0, 0.0, 1.0], kx=1, ky=1, s=0)
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, sciframe.shape[0]), np.linspace(0.0, 1.0, npix), indexing='ij')
    ro = (slf._rordloc[det-1][:, slitn] - trimr).reshape((-1, 1)) / (sciframe.shape[1] - 1.0)
    lo = (slf._lordloc[det-1][:, slitn] + triml).reshape((-1, 1)) / (sciframe.shape[1] - 1.0)
    vv = (lo+(ro-lo)*yy).flatten()
    xx = xx.flatten()
    recsh = (sciframe.shape[0], npix)
    rec_sciframe = scispl.ev(xx, vv).reshape(recsh)
    rec_varframe = varspl.ev(xx, vv).reshape(recsh)
    rec_crmask   = crmspl.ev(xx, vv).reshape(recsh)
    # Update the CR mask to ensure it only contains 1's and 0's
    rec_crmask[np.where(rec_crmask > 0.2)] = 1.0
    rec_crmask[np.where(rec_crmask <= 0.2)] = 0.0
    msgs.info("Estimating object profiles")
    # Avoid any 0's in varframe
    zero_var = rec_varframe == 0.
    if np.any(zero_var):
        rec_varframe[zero_var] = 1.
        rec_crmask[zero_var] = 1.
    # Smooth the S/N frame
    smthby, rejhilo = tracepar['smthby'], tracepar['rejhilo']
#    print('calling smooth_x')
#    t = time.clock()
#    _rec_sigframe_bin = arcyutils.smooth_x(rec_sciframe/np.sqrt(rec_varframe), 1.0-rec_crmask,
#                                           smthby, rejhilo, maskval)
#    print('Old smooth_x: {0} seconds'.format(time.clock() - t))
    tmp = np.ma.MaskedArray(rec_sciframe/np.sqrt(rec_varframe), mask=rec_crmask.astype(bool))
#    # TODO: Add rejection to BoxcarFilter?
#    t = time.clock()
    rec_sigframe_bin = BoxcarFilter(smthby).smooth(tmp.T).T
#    print('BoxcarFilter: {0} seconds'.format(time.clock() - t))
#    t = time.clock()
#    rec_sigframe_bin = new_smooth_x(rec_sciframe/np.sqrt(rec_varframe), 1.0-rec_crmask, smthby,
#                                    rejhilo, maskval)
#    print('New smooth_x: {0} seconds'.format(time.clock() - t))
    # TODO: BoxcarFilter and smooth_x will provide different results.
    # Need to assess their importance.
#    if np.sum(_rec_sigframe_bin != rec_sigframe_bin) != 0:
#
#        plt.plot(tmp[:,200])
#        plt.plot(rec_sigframe_bin[:,200])
#        plt.plot(_rec_sigframe_bin[:,200])
#        plt.show()
#
#        plt.imshow( np.ma.MaskedArray(_rec_sigframe_bin, mask=rec_crmask.astype(bool)),
#                   origin='lower', interpolation='nearest', aspect='auto')
#        plt.colorbar()
#        plt.show()
#        plt.imshow(np.ma.MaskedArray(rec_sigframe_bin, mask=rec_crmask.astype(bool)),
#                   origin='lower', interpolation='nearest', aspect='auto')
#        plt.colorbar()
#        plt.show()
#        plt.imshow(np.ma.MaskedArray(_rec_sigframe_bin-rec_sigframe_bin,
#                                     mask=rec_crmask.astype(bool)),
#                   origin='lower', interpolation='nearest', aspect='auto')
#        plt.colorbar()
#        plt.show()
#        t = np.ma.divide(_rec_sigframe_bin,rec_sigframe_bin)
#        t[rec_crmask.astype(bool)] = np.ma.masked
#        plt.imshow(t,
#                   origin='lower', interpolation='nearest', aspect='auto')
#        plt.colorbar()
#        plt.show()
#    assert np.sum(_rec_sigframe_bin != rec_sigframe_bin) == 0, \
#                    'Difference between old and new smooth_x'

    #rec_varframe_bin = arcyutils.smooth_x(rec_varframe, 1.0-rec_crmask, smthby, rejhilo, maskval)
    #rec_sigframe_bin = np.sqrt(rec_varframe_bin/(smthby-2.0*rejhilo))
    #sigframe = rec_sciframe_bin*(1.0-rec_crmask)/rec_sigframe_bin
    ww = np.where(rec_crmask == 0.0)
    med, mad = arutils.robust_meanstd(rec_sigframe_bin[ww])
    ww = np.where(rec_crmask == 1.0)
    rec_sigframe_bin[ww] = maskval
    srtsig = np.sort(rec_sigframe_bin,axis=1)
    ww = np.where(srtsig[:, -2] > med + sigmin*mad)
    mask_sigframe = np.ma.array(rec_sigframe_bin, mask=rec_crmask, fill_value=maskval)
    # Clip image along spatial dimension -- May eliminate bright emission lines
    clip_image = sigma_clip(mask_sigframe[ww[0], :], axis=0, sigma=4.)
    # Collapse along the spectral direction to get object profile
    #trcprof = np.ma.mean(mask_sigframe[ww[0],:], axis=0).filled(0.0)
    trcprof = np.ma.mean(clip_image, axis=0).filled(0.0)
    trcxrng = np.arange(npix)/(npix-1.0)
    msgs.info("Identifying objects that are significantly detected")
    # Find significantly detected objects
    mskpix, coeff = arutils.robust_polyfit(trcxrng, trcprof, 1+npix//40,
                                           function='legendre', sigma=2.0, minv=0.0, maxv=1.0)
    backg = arutils.func_val(coeff, trcxrng, 'legendre', minv=0.0, maxv=1.0)
    trcprof -= backg
    wm = np.where(mskpix == 0)
    if wm[0].size == 0:
        msgs.warn("No objects found")
        return trace_object_dict(0, None)
    med, mad = arutils.robust_meanstd(trcprof[wm])
    trcprof -= med
    # Gaussian smooth
    #from scipy.ndimage.filters import gaussian_filter1d
    #smth_prof = gaussian_filter1d(trcprof, fwhm/2.35)
    # Define all 5 sigma deviations as objects (should make the 5 user-defined)
    #objl, objr, bckl, bckr = arcytrace.find_objects(smth_prof, bgreg, mad)
    if settings.argflag['trace']['object']['find'] == 'nminima':
        nsmooth = settings.argflag['trace']['object']['nsmooth']
        trcprof2 = np.mean(rec_sciframe, axis=0)
        objl, objr, bckl, bckr = find_obj_minima(trcprof2, triml=triml, trimr=trimr, nsmooth=nsmooth)
    elif settings.argflag['trace']['object']['find'] == 'standard':
#        print('calling find_objects')
#        t = time.clock()
#        _objl, _objr, _bckl, _bckr = arcytrace.find_objects(trcprof, bgreg, mad)
#        print('Old find_objects: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        objl, objr, bckl, bckr = new_find_objects(trcprof, bgreg, mad)
#        print('New find_objects: {0} seconds'.format(time.clock() - t))
#        print('objl:', objl)
#        print('_objl:', _objl)
#        print('objr:', objr)
#        print('_objr:', _objr)
#        print(_objl.shape, objl.shape)
#        print(_objr.shape, objr.shape)
#        print(np.sum(_objl != objl))
#        print(np.sum(_objr != objr))
#        assert np.sum(_objl != objl) == 0, 'Difference between old and new find_objects, objl'
#        assert np.sum(_objr != objr) == 0, 'Difference between old and new find_objects, objr'
#        assert np.sum(_bckl != bckl) == 0, 'Difference between old and new find_objects, bckl'
#        assert np.sum(_bckr != bckr) == 0, 'Difference between old and new find_objects, bckr'
#        objl, objr, bckl, bckr = new_find_objects(trcprof, bgreg, mad)

    else:
        msgs.error("Bad object identification algorithm!!")
    if msgs._debug['trace_obj']:
        debugger.set_trace()
    # Remove objects within x percent of the slit edge
    xp = (objr+objl)/float(trcprof.size)/2.
    gdobj = (xp < (1-xedge)) * (xp > xedge)
    objl = objl[gdobj]
    objr = objr[gdobj]
    bckl = bckl[:, gdobj]
    bckr = bckr[:, gdobj]
    if np.sum(~gdobj) > 0:
        msgs.warn("Removed objects near the slit edges")
    #
    nobj = objl.size

    if settings.argflag['science']['extraction']['manual01']['params'] is not None and not standard:
        msgs.info('Manual extraction desired. Rejecting all automatically detected objects for now.')
        # Work on: Instead of rejecting all objects, prepend the manual extraction object?

        if settings.argflag['science']['extraction']['manual01']['params'][0] == det:
            nobj = 1
            cent_spatial_manual = settings.argflag['science']['extraction']['manual01']['params'][1]
            # Entered into .pypit file in this format: [det, x_pixel_location, y_pixel_location,[x_range, y_range]]
            # 1 or x_pixel_location is spatial pixel; 2 or y_pixel_location is dispersion/spectral pixel
            width_spatial_manual = settings.argflag['science']['extraction']['manual01']['params'][3][0]
            objl = np.array([int(cent_spatial_manual - slf._lordloc[det - 1][cent_spatial_manual]) - width_spatial_manual])
            objr = np.array([int(cent_spatial_manual - slf._lordloc[det - 1][cent_spatial_manual]) + width_spatial_manual])
            bckl = np.zeros((trcprof.shape[0], objl.shape[0]))
            bckr = np.zeros((trcprof.shape[0], objl.shape[0]))

            for o in range(nobj):
                for x in range(1, bgreg + 1):
                    if objl[o] - x >= 0:
                        bckl[objl[o] - x, o] = 1
                    if objr[o] + x <= trcprof.shape[0] - 1:
                        bckr[objr[o] + x, o] = 1
        else:
            nobj = 0

    if nobj == 1:
        msgs.info("Found {0:d} object".format(objl.size))
        msgs.info("Tracing {0:d} object".format(objl.size))
    else:
        msgs.info("Found {0:d} objects".format(objl.size))
        msgs.info("Tracing {0:d} objects".format(objl.size))
    # Max obj
    if nobj > settings.argflag['science']['extraction']['maxnumber']:
        nobj = settings.argflag['science']['extraction']['maxnumber']
        msgs.warn("Restricting to the brightest {:d} objects found".format(nobj))
    # Trace objects
    cval = np.zeros(nobj)
    allsfit = np.array([])
    allxfit = np.array([])
    clip_image2 = sigma_clip(mask_sigframe, axis=0, sigma=4.)
    for o in range(nobj):
        xfit = np.arange(objl[o], objr[o]).reshape((1, -1))/(npix-1.0)
        #cent = np.ma.sum(mask_sigframe[:,objl[o]:objr[o]]*xfit, axis=1)
        #wght = np.ma.sum(mask_sigframe[:,objl[o]:objr[o]], axis=1)
        cent = np.ma.sum(clip_image2[:, objl[o]:objr[o]]*xfit, axis=1)
        wght = np.ma.sum(clip_image2[:, objl[o]:objr[o]], axis=1)
        cent /= wght
        centfit = cent.filled(maskval)
        specfit = np.linspace(-1.0, 1.0, sciframe.shape[0])
        w = np.where(centfit != maskval)
        specfit = specfit[w]
        centfit = centfit[w]
        mskbad, coeffs = arutils.robust_polyfit(specfit, centfit, traceorder, function=tracefunc, minv=-1.0, maxv=1.0)
        cval[o] = arutils.func_val(coeffs, np.array([0.0]), tracefunc, minv=-1.0, maxv=1.0)[0]
        w = np.where(mskbad == 0.0)
        if w[0].size != 0:
            allxfit = np.append(allxfit, specfit[w])
            allsfit = np.append(allsfit, centfit[w]-cval[o])
    if nobj == 0:
        msgs.warn("No objects detected in slit")
        return trace_object_dict(0, None)
    # Tracing
    msgs.info("Performing global trace to all objects")
    mskbad, coeffs = arutils.robust_polyfit(allxfit, allsfit, traceorder, function=tracefunc, minv=-1.0, maxv=1.0)
    trcfunc = arutils.func_val(coeffs, np.linspace(-1.0, 1.0, sciframe.shape[0]), tracefunc, minv=-1.0, maxv=1.0)
    msgs.info("Constructing a trace for all objects")
    trcfunc = trcfunc.reshape((-1, 1)).repeat(nobj, axis=1)
    trccopy = trcfunc.copy()
    for o in range(nobj):
        trcfunc[:, o] += cval[o]
    if nobj == 1:
        msgs.info("Converting object trace to detector pixels")
    else:
        msgs.info("Converting object traces to detector pixels")
    ofst = slf._lordloc[det-1][:, slitn].reshape((-1, 1)).repeat(nobj, axis=1) + triml
    diff = (slf._rordloc[det-1][:, slitn].reshape((-1, 1)).repeat(nobj, axis=1)
            - slf._lordloc[det-1][:, slitn].reshape((-1, 1)).repeat(nobj, axis=1))
    # Convert central trace
    traces = ofst + (diff-triml-trimr)*trcfunc
    # Convert left object trace
    for o in range(nobj):
        trccopy[:, o] = trcfunc[:, o] - cval[o] + objl[o]/(npix-1.0)
    trobjl = ofst + (diff-triml-trimr)*trccopy
    # Convert right object trace
    for o in range(nobj):
        trccopy[:, o] = trcfunc[:, o] - cval[o] + objr[o]/(npix-1.0)
    trobjr = ofst + (diff-triml-trimr)*trccopy
    # Generate an image of pixel weights for each object
    rec_obj_img, rec_bg_img = trace_objbg_image(slf, det, sciframe, slitn,
                                                [objl, objr], [bckl, bckr],
                                                triml=triml, trimr=trimr)
    # Check object traces in ginga
    if msgs._debug['trace_obj']:
        viewer, ch = ginga.show_image(sciframe)
        for ii in range(nobj):
            ginga.show_trace(viewer, ch, traces[:, ii], '{:d}'.format(ii), clear=(ii == 0))
        debugger.set_trace()
    # Trace dict
    tracedict = trace_object_dict(nobj, traces, object=rec_obj_img, background=rec_bg_img,
                                  params=tracepar, tracelist=tracedict)

    # Save the quality control
    if doqa: # and (not msgs._debug['no_qa']):
        objids = []
        for ii in range(nobj):
            objid, xobj = arspecobj.get_objid(slf, det, slitn, ii, tracedict)
            objids.append(objid)
#        arqa.obj_trace_qa(slf, sciframe, trobjl, trobjr, objids, det,
#                          root="object_trace", normalize=False)
        obj_trace_qa(slf, sciframe, trobjl, trobjr, objids, det, root="object_trace",
                     normalize=False)
    # Return
    return tracedict


def obj_trace_qa(slf, frame, ltrace, rtrace, objids, det,
                 root='trace', normalize=True, desc=""):
    """ Generate a QA plot for the object trace

    Parameters
    ----------
    frame : ndarray
      image
    ltrace : ndarray
      Left edge traces
    rtrace : ndarray
      Right edge traces
    objids : list
    det : int
    desc : str, optional
      Title
    root : str, optional
      Root name for generating output file, e.g. msflat_01blue_000.fits
    normalize : bool, optional
      Normalize the flat?  If not, use zscale for output
    """

    plt.rcdefaults()

    # Grab the named of the method
    method = inspect.stack()[0][3]
    # Outfile name
    outfile = arqa.set_qa_filename(slf._basename, method, det=det)
    #
    ntrc = ltrace.shape[1]
    ycen = np.arange(frame.shape[0])
    # Normalize flux in the traces
    if normalize:
        nrm_frame = np.zeros_like(frame)
        for ii in range(ntrc):
            xtrc = (ltrace[:,ii] + rtrace[:,ii])/2.
            ixtrc = np.round(xtrc).astype(int)
            # Simple 'extraction'
            dumi = np.zeros( (frame.shape[0],3) )
            for jj in range(3):
                dumi[:,jj] = frame[ycen,ixtrc-1+jj]
            trc = np.median(dumi, axis=1)
            # Find portion of the image and normalize
            for yy in ycen:
                xi = max(0, int(ltrace[yy,ii])-3)
                xe = min(frame.shape[1],int(rtrace[yy,ii])+3)
                # Fill + normalize
                nrm_frame[yy, xi:xe] = frame[yy,xi:xe] / trc[yy]
        sclmin, sclmax = 0.4, 1.1
    else:
        nrm_frame = frame.copy()
        sclmin, sclmax = arplot.zscale(nrm_frame)

    # Plot
    plt.clf()
    fig = plt.figure(dpi=1200)

    plt.rcParams['font.family'] = 'times new roman'
    ticks_font = font_manager.FontProperties(family='times new roman',
       style='normal', size=16, weight='normal', stretch='normal')
    ax = plt.gca()
    for label in ax.get_yticklabels() :
        label.set_fontproperties(ticks_font)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(ticks_font)
    cmm = cm.Greys_r
    mplt = plt.imshow(nrm_frame, origin='lower', cmap=cmm, extent=(0., frame.shape[1], 0., frame.shape[0]))
    mplt.set_clim(vmin=sclmin, vmax=sclmax)

    # Axes
    plt.xlim(0., frame.shape[1])
    plt.ylim(0., frame.shape[0])
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')

    # Traces
    iy_mid = int(frame.shape[0] / 2.)
    iy = np.linspace(iy_mid,frame.shape[0]*0.9,ntrc).astype(int)
    for ii in range(ntrc):
        # Left
        plt.plot(ltrace[:, ii]+0.5, ycen, 'r--', alpha=0.7)
        # Right
        plt.plot(rtrace[:, ii]+0.5, ycen, 'c--', alpha=0.7)
        if objids is not None:
            # Label
            # plt.text(ltrace[iy,ii], ycen[iy], '{:d}'.format(ii+1), color='red', ha='center')
            lbl = 'O{:03d}'.format(objids[ii])
            plt.text((ltrace[iy[ii], ii]+rtrace[iy[ii], ii])/2., ycen[iy[ii]],
                lbl, color='green', ha='center')
    # Title
    tstamp = arqa.gen_timestamp()
    if desc == "":
        plt.suptitle(tstamp)
    else:
        plt.suptitle(desc+'\n'+tstamp)

    plt.savefig(outfile, dpi=800)
    plt.close()

    plt.rcdefaults()


def plot_orderfits(slf, model, ydata, xdata=None, xmodl=None, textplt="Slit",
                   maxp=4, desc="", maskval=-999999.9):
    """ Generate a QA plot for the blaze function fit to each slit
    Or the arc line tilts

    Parameters
    ----------
    slf : class
      Science Exposure class
    model : ndarray
      (m x n) 2D array containing the model blaze function (m) of a flat frame for each slit (n)
    ydata : ndarray
      (m x n) 2D array containing the extracted 1D spectrum (m) of a flat frame for each slit (n)
    xdata : ndarray, optional
      x values of the data points
    xmodl : ndarry, optional
      x values of the model points
    textplt : str, optional
      A string printed above each panel
    maxp : int, (optional)
      Maximum number of panels per page
    desc : str, (optional)
      A description added to the top of each page
    maskval : float, (optional)
      Value used in arrays to indicate a masked value
    """

    plt.rcdefaults()
    plt.rcParams['font.family']= 'times new roman'

    # Outfil
    method = inspect.stack()[0][3]
    if 'Arc' in desc:
        method += '_Arc'
    elif 'Blaze' in desc:
        method += '_Blaze'
    else:
        msgs.bug("Unknown type of order fits.  Currently prepared for Arc and Blaze")
    outroot = arqa.set_qa_filename(slf.setup, method)
    #
    npix, nord = ydata.shape
    pages, npp = arqa.get_dimen(nord, maxp=maxp)
    if xdata is None: xdata = np.arange(npix).reshape((npix, 1)).repeat(nord, axis=1)
    if xmodl is None: xmodl = np.arange(model.shape[0])
    # Loop through all pages and plot the results
    ndone = 0
    axesIdx = True
    for i in range(len(pages)):
        f, axes = plt.subplots(pages[i][1], pages[i][0])
        ipx, ipy = 0, 0
        for j in range(npp[i]):
            if pages[i][0] == 1 and pages[i][1] == 1: axesIdx = False
            elif pages[i][1] == 1: ind = (ipx,)
            elif pages[i][0] == 1: ind = (ipy,)
            else: ind = (ipy, ipx)
            if axesIdx:
                axes[ind].plot(xdata[:,ndone+j], ydata[:,ndone+j], 'bx', drawstyle='steps')
                axes[ind].plot(xmodl, model[:,ndone+j], 'r-')
            else:
                axes.plot(xdata[:,ndone+j], ydata[:,ndone+j], 'bx', drawstyle='steps')
                axes.plot(xmodl, model[:,ndone+j], 'r-')
            ytmp = ydata[:,ndone+j]
            gdy = ytmp != maskval
            ytmp = ytmp[gdy]
            if ytmp.size != 0:
                amn = min(np.min(ytmp), np.min(model[gdy,ndone+j]))
            else:
                amn = np.min(model[:,ndone+j])
            if ytmp.size != 0:
                amx = max(np.max(ytmp), np.max(model[gdy,ndone+j]))
            else: amx = np.max(model[:,ndone+j])
            # Restrict to good pixels
            xtmp = xdata[:,ndone+j]
            gdx = xtmp != maskval
            xtmp = xtmp[gdx]
            if xtmp.size == 0:
                xmn = np.min(xmodl)
                xmx = np.max(xmodl)
            else:
                xmn = np.min(xtmp)
                xmx = np.max(xtmp)
                #xmn = min(np.min(xtmp), np.min(xmodl))
                #xmx = max(np.max(xtmp), np.max(xmodl))
            if axesIdx:
                axes[ind].axis([xmn, xmx, amn-1, amx+1])
                axes[ind].set_title("{0:s} {1:d}".format(textplt, ndone+j+1))
            else:
                axes.axis([xmn, xmx, amn, amx])
                axes.set_title("{0:s} {1:d}".format(textplt, ndone+j+1))
            ipx += 1
            if ipx == pages[i][0]:
                ipx = 0
                ipy += 1
        # Delete the unnecessary axes
        if axesIdx:
            for j in range(npp[i], axes.size):
                if pages[i][1] == 1: ind = (ipx,)
                elif pages[i][0] == 1: ind = (ipy,)
                else: ind = (ipy, ipx)
                f.delaxes(axes[ind])
                if ipx == pages[i][0]:
                    ipx = 0
                    ipy += 1
        ndone += npp[i]
        # Save the figure
        if axesIdx: axsz = axes.size
        else: axsz = 1.0
        if pages[i][1] == 1 or pages[i][0] == 1: ypngsiz = 11.0/axsz
        else: ypngsiz = 11.0*axes.shape[0]/axes.shape[1]
        f.set_size_inches(11.0, ypngsiz)
        if desc != "":
            pgtxt = ""
            if len(pages) != 1:
                pgtxt = ", page {0:d}/{1:d}".format(i+1, len(pages))
            f.suptitle(desc + pgtxt, y=1.02, size=16)
        f.tight_layout()
        outfile = outroot+'{:03d}.png'.format(i)
        plt.savefig(outfile, dpi=200)
        plt.close()
        f.clf()
    del f

    plt.rcdefaults()

    return


def new_find_between(edgdet, ledgem, ledgep, dirc):

    if len(edgdet.shape) != 2:
        msgs.error('Edge pixels array must be 2D.')
    if len(ledgem.shape) != 1 or len(ledgep.shape) !=1:
        msgs.error('Input must be 1D.')

    sz_x, sz_y = edgdet.shape

    # Setup the coefficient arrays
    edgbtwn = np.full(3, -1, dtype=int)

    for x in range(0,sz_x):
        rng = np.sort([ledgem[x],ledgep[x]])
        if not np.any(edgdet[x,slice(*rng)] > 0):
            continue
        e = edgdet[x,slice(*rng)][edgdet[x,slice(*rng)] > 0]
        if edgbtwn[0] == -1:
            edgbtwn[0] = e[0]
        indx = e != edgbtwn[0]
        if edgbtwn[1] == -1 and np.any(indx):
            edgbtwn[1] = e[indx][0]

    # If no right order edges were found between these two left order
    # edges, find the next right order edge
    if edgbtwn[0] == -1 and edgbtwn[1] == -1:
        for x in range(0,sz_x):
            ystrt = np.max([ledgem[x],ledgep[x]])
            if dirc == 1:
                emin = np.min(edgdet[x,ystrt:][edgdet[x,ystrt:] > 0])
                if edgbtwn[2] == -1 or emin < edgbtwn[2]:
                    edgbtwn[2] = emin
            else:
                emax = np.max(edgdet[x,:ystrt+1][edgdet[x,:ystrt+1] > 0])
                if edgbtwn[2] == -1 or emax > edgbtwn[2]:
                    edgbtwn[2] = emax

    # Now return the array
    return edgbtwn


def new_find_objects(profile, bgreg, stddev):
    """
    Find significantly detected objects in the profile array
    For all objects found, the background regions will be defined.
    """
    # Input profile must be a vector
    if len(profile.shape) != 1:
        raise ValueError('Input profile array must be 1D.')
    # Get the length of the array
    sz_x = profile.size
    # Copy it to a masked array for processing
    # TODO: Assumes input is NOT a masked array
    _profile = np.ma.MaskedArray(profile.copy())

    # Peaks are found as having flux at > 5 sigma and background is
    # where the flux is not greater than 3 sigma
    gt_5_sigma = profile > 5*stddev
    not_gt_3_sigma = np.invert(profile > 3*stddev)

    # Define the object centroids array
    objl = np.zeros(sz_x, dtype=int)
    objr = np.zeros(sz_x, dtype=int)
    has_obj = np.zeros(sz_x, dtype=bool)

    obj = 0
    while np.any(gt_5_sigma & np.invert(_profile.mask)):
        # Find next maximum flux point
        imax = np.ma.argmax(_profile)
        # Find the valid source pixels around the peak
        f = np.arange(sz_x)[np.roll(not_gt_3_sigma, -imax)]
        # TODO: the ifs below feel like kludges to match old
        # find_objects function.  In particular, should objr be treated
        # as exclusive or inclusive?
        objl[obj] = imax-sz_x+f[-1] if imax-sz_x+f[-1] > 0 else 1
        objr[obj] = f[0]+imax if f[0]+imax < sz_x else sz_x-1
#        print('object found: ', imax, f[-1], objl[obj], f[0], objr[obj], sz_x)
        # Mask source pixels and increment for next iteration
        has_obj[objl[obj]:objr[obj]+1] = True
        _profile[objl[obj]:objr[obj]+1] = np.ma.masked
        obj += 1

    # The background is the region away from sources up to the provided
    # region size.  Starting pixel for the left limit...
    s = objl[:obj]-bgreg
    s[s < 0] = 0
    # ... and ending pixel for the right limit.
    e = objr[:obj]+1+bgreg
    e[e > sz_x] = sz_x
    # Flag the possible background regions
    bgl = np.zeros((sz_x,obj), dtype=bool)
    bgr = np.zeros((sz_x,obj), dtype=bool)
    for i in range(obj):
        bgl[s[i]:objl[i],i] = True
        bgr[objl[i]+1:e[i],i] = True

    # Return source region limits and background regions that do not
    # have sources
    return objl[:obj], objr[:obj], (bgl & np.invert(has_obj)[:,None]).astype(int), \
                    (bgr & np.invert(has_obj)[:,None]).astype(int)


def new_find_peak_limits(hist, pks):
    """
    Find all values between the zeros of hist

    hist and pks are expected to be 1d vectors
    """
    if len(hist.shape) != 1 or len(pks.shape) != 1:
        msgs.error('Arrays provided to find_peak_limits must be vectors.')
    # Pixel indices in hist for each peak
    hn = np.arange(hist.shape[0])
    indx = np.ma.MaskedArray(np.array([hn]*pks.shape[0]))
    # Instantiate output
    edges = np.zeros((pks.shape[0],2), dtype=int)
    # Find the left edges
    indx.mask = (hist != 0)[None,:] | (hn[None,:] > pks[:,None])
    edges[:,0] = np.ma.amax(indx, axis=1)
    # Find the right edges
    indx.mask = (hist != 0)[None,:] | (hn[None,:] < pks[:,None])
    edges[:,1] = np.ma.amin(indx, axis=1)
    return edges


def new_find_shift(mstrace, minarr, lopos, diffarr, numsrch):

    sz_y = mstrace.shape[1]
    maxcnts = -999999.9
    shift = 0
    d = mstrace - minarr[:,None]

    for s in range(0,numsrch):
        cnts = 0.0

        ymin = lopos + s
        ymin[ymin < 0] = 0
        ymax = ymin + diffarr
        ymax[ymax > sz_y] = sz_y

        indx = ymax > ymin

        if np.sum(indx) == 0:
            continue

        cnts = np.sum([ np.sum(t[l:h]) for t,l,h in zip(d[indx], ymin[indx], ymax[indx]) ]) \
                    / np.sum(ymax[indx]-ymin[indx])
        if cnts > maxcnts:
            maxcnts = cnts
            shift = s

    return shift


def new_ignore_orders(edgdet, fracpix, lmin, lmax, rmin, rmax):
    """
    .. warning::

        edgdet is alted by the function.
    """
    sz_x, sz_y = edgdet.shape

    lsize = lmax-lmin+1
    larr = np.zeros((2,lsize), dtype=int)
    larr[0,:] = sz_x

    rsize = rmax-rmin+1
    rarr = np.zeros((2,rsize), dtype=int)
    rarr[0,:] = sz_x

    # TODO: Can I remove the loop?  Or maybe just iterate through the
    # smallest dimension of edgdet?
    for x in range(sz_x):
        indx = edgdet[x,:] < 0
        if np.any(indx):
            larr[0,-edgdet[x,indx]-lmin] = np.clip(larr[0,-edgdet[x,indx]-lmin], None, x)
            larr[1,-edgdet[x,indx]-lmin] = np.clip(larr[1,-edgdet[x,indx]-lmin], x, None)
        indx = edgdet[x,:] > 0
        if np.any(indx):
            rarr[0,edgdet[x,indx]-rmin] = np.clip(rarr[0,edgdet[x,indx]-rmin], None, x)
            rarr[1,edgdet[x,indx]-rmin] = np.clip(rarr[1,edgdet[x,indx]-rmin], x, None)

    # Go through the array once more to remove pixels that do not cover fracpix
    edgdet = edgdet.ravel()
    lt_zero = np.arange(edgdet.size)[edgdet < 0]
    if len(lt_zero) > 0:
        edgdet[lt_zero[larr[1,-edgdet[lt_zero]-lmin]-larr[0,-edgdet[lt_zero]-lmin] < fracpix]] = 0
    gt_zero = np.arange(edgdet.size)[edgdet > 0]
    if len(gt_zero) > 0:
        edgdet[gt_zero[rarr[1,edgdet[gt_zero]-rmin]-rarr[0,edgdet[gt_zero]-rmin] < fracpix]] = 0
    edgdet = edgdet.reshape(sz_x,sz_y)

    # Check if lmin, lmax, rmin, and rmax need to be changed
    lindx = np.arange(lsize)[larr[1,:]-larr[0,:] > fracpix]
    lnc = lindx[0]
    lxc = lsize-1-lindx[-1]

    rindx = np.arange(rsize)[rarr[1,:]-rarr[0,:] > fracpix]
    rnc = rindx[0]
    rxc = rsize-1-rindx[-1]

    return lnc, lxc, rnc, rxc, larr, rarr


def new_limit_yval(yc, maxv):
    yn = 0 if yc == 0 else (-yc if yc < 3 else -3)
    yx = maxv-yc if yc > maxv-4 and yc < maxv else 4
    return yn, yx


def new_match_edges(edgdet, ednum):
    """
    ednum is a large dummy number used for slit edge assignment. ednum
    should be larger than the number of edges detected

    This function alters edgdet!
    """
    # mr is the minimum number of acceptable pixels required to form the
    # detection of an order edge
    mr = 5
    mrxarr = np.zeros(mr, dtype=int)
    mryarr = np.zeros(mr, dtype=int)

    sz_x, sz_y = edgdet.shape

    lcnt = 2*ednum
    rcnt = 2*ednum
    for y in range(sz_y):
        for x in range(sz_x):
            if edgdet[x,y] != -1 and edgdet[x,y] != 1:
                continue

            anyt = 0
            left = edgdet[x,y] == -1

            # Search upwards from x,y
            xs = x + 1
            yt = y
            while xs <= sz_x-1:
                xr = 10 if xs + 10 < sz_x else sz_x - xs - 1
                yn, yx = new_limit_yval(yt, sz_y)

                suc = 0
                for s in range(xs, xs+xr):
                    suc = 0
                    for t in range(yt + yn, yt + yx):
                        if edgdet[s, t] == -1 and left:
                            edgdet[s, t] = -lcnt
                        elif edgdet[s, t] == 1 and not left:
                            edgdet[s, t] = rcnt
                        else:
                            continue

                        suc = 1
                        if anyt < mr:
                            mrxarr[anyt] = s
                            mryarr[anyt] = t
                        anyt += 1
                        yt = t
                        break

                    if suc == 1:
                        xs = s + 1
                        break
                if suc == 0: # The trace is lost!
                    break

            # Search downwards from x,y
            xs = x - 1
            yt = y
            while xs >= 0:
                xr = xs if xs-10 < 0 else 10
                yn, yx = new_limit_yval(yt, sz_y)

                suc = 0
                for s in range(0, xr):
                    suc = 0
                    for t in range(yt+yn, yt+yx):
                        if edgdet[xs-s, t] == -1 and left:
                            edgdet[xs-s, t] = -lcnt
                        elif edgdet[xs-s, t] == 1 and not left:
                            edgdet[xs-s, t] = rcnt
                        else:
                            continue

                        suc = 1
                        if anyt < mr:
                            mrxarr[anyt] = xs-s
                            mryarr[anyt] = t
                        anyt += 1
                        yt = t
                        break

                    if suc == 1:
                        xs = xs - s - 1
                        break
                if suc == 0: # The trace is lost!
                    break

            if anyt > mr and left:
                edgdet[x, y] = -lcnt
                lcnt = lcnt + 1
            elif anyt > mr and not left:
                edgdet[x, y] = rcnt
                rcnt = rcnt + 1
            else:
                edgdet[x, y] = 0
                for s in range(anyt):
                    if mrxarr[s] != 0 and mryarr[s] != 0:
                        edgdet[mrxarr[s], mryarr[s]] = 0
    return lcnt-2*ednum, rcnt-2*ednum


def new_mean_weight(array, weight, rejhilo, maskval):
    _a = array if rejhilo == 0 else np.sort(array)
    sumw = np.sum(weight[rejhilo:-rejhilo])
    sumwa = np.sum(weight[rejhilo:-rejhilo]*_a[rejhilo:-rejhilo])
    return maskval if sumw == 0.0 else sumwa/sumw


def new_minbetween(mstrace, loord, hiord):
    # TODO: Check shapes
    ymin = np.clip(loord, 0, mstrace.shape[1])
    ymax = np.clip(hiord, 0, mstrace.shape[1])
    minarr = np.zeros(mstrace.shape[0])
    indx = ymax > ymin
    minarr[indx] = np.array([ np.amin(t[l:h]) 
                                for t,l,h in zip(mstrace[indx], ymin[indx], ymax[indx]) ])
    return minarr


def new_phys_to_pix(array, diff):
    if len(array.shape) > 2:
        msgs.error('Input array must have two dimensions or less!')
    if len(diff.shape) != 1:
        msgs.error('Input difference array must be 1D!')
    _array = np.atleast_2d(array)
    doravel = len(array.shape) != 2
    pix = np.argmin(np.absolute(_array[:,:,None] - diff[None,None,:]), axis=2)
    return pix.ravel() if doravel else pix

    sz_a, sz_n = _array.shape
    sz_d = diff.size

    pix = np.zeros((sz_a,sz_n), dtype=int)
    for n in range(0,sz_n):
        for a in range(0,sz_a):
            mind = 0
            mindv = _array[a,n]-diff[0]

            for d in range(1,sz_d):
                test = _array[a,n]-diff[d]
                if test < 0.0: test *= -1.0
                if test < mindv:
                    mindv = test
                    mind = d
                if _array[a,n]-diff[d] < 0.0: break
            pix[a,n] = mind
    return pix.ravel() if doravel else pix

# Weighted boxcar smooothing with rejection
def new_smooth_x(array, weight, fact, rejhilo, maskval):
    hf = fact // 2

    sz_x, sz_y = array.shape

    smtarr = np.zeros((sz_x,sz_y), dtype=float)
    medarr = np.zeros((fact+1), dtype=float)
    wgtarr = np.zeros((fact+1), dtype=float)

    for y in range(sz_y):
        for x in range(sz_x):
            for b in range(fact+1):
                if (x+b-hf < 0) or (x+b-hf >= sz_x):
                    wgtarr[b] = 0.0
                else:
                    medarr[b] = array[x+b-hf,y]
                    wgtarr[b] = weight[x+b-hf,y]
            smtarr[x,y] = new_mean_weight(medarr, wgtarr, rejhilo, maskval)
    return smtarr


def new_tilts_image(tilts, lordloc, rordloc, pad, sz_y):
    """
    Using the tilt (assumed to be fit with a first order polynomial)
    generate an image of the tilts for each slit.

    Parameters
    ----------
    tilts : ndarray
      An (m x n) 2D array specifying the tilt (i.e. gradient) at each
      pixel along the spectral direction (m) for each slit (n).
    lordloc : ndarray
      Location of the left slit edges
    rordloc : ndarray
      Location of the right slit edges
    pad : int
      Set the tilts within each slit, and extend a number of pixels
      outside the slit edges (this number is set by pad).
    sz_y : int
      Number of detector pixels in the spatial direction.

    Returns
    -------
    tiltsimg : ndarray
      An image the same size as the science frame, containing values from 0-1.
      0/1 corresponds to the bottom/top of the detector (in the spectral direction),
      and constant wavelength is represented by a single value from 0-1.

    """
    sz_x, sz_o = tilts.shape
    dszx = (sz_x-1.0)

    tiltsimg = np.zeros((sz_x,sz_y), dtype=float)
    for o in range(sz_o):
        for x in range(sz_x):
            ow = (rordloc[x,o]-lordloc[x,o])/2.0
            oc = (rordloc[x,o]+lordloc[x,o])/2.0
            ymin = int(oc-ow) - pad
            ymax = int(oc+ow) + 1 + pad
            # Check we are in bounds
            if ymin < 0:
                ymin = 0
            elif ymax < 0:
                continue
            if ymax > sz_y-1:
                ymax = sz_y-1
            elif ymin > sz_y-1:
                continue
            # Set the tilt value at each pixel in this row
            for y in range(ymin, ymax):
                yv = (y-lordloc[x, o])/ow - 1.0
                tiltsimg[x,y] = (tilts[x,o]*yv + x)/dszx
    return tiltsimg
    

    sz_x, sz_o = tilts.shape
    dszx = (sz_x-1.0)

    ow = (rordloc-lordloc)/2.0
    oc = (rordloc+lordloc)/2.0
    
    ymin = (oc-ow).astype(int) - pad
    ymax = (oc+ow).astype(int) + 1 + pad

    indx = (ymax >= 0) & (ymin < sz_y)
    ymin[ymin < 0] = 0
    ymax[ymax > sz_y-1] = sz_y-1

    tiltsimg = np.zeros((sz_x,sz_y), dtype=float)

    xv = np.arange(sz_x).astype(int)
    for o in range(sz_o):
        if np.sum(indx[:,o]) == 0:
            continue
        for x in xv[indx[:,o]]:
            # Set the tilt value at each pixel in this row
            for y in range(ymin[x,o], ymax[x,o]):
                yv = (y-lordloc[x,o])/ow[x,o] - 1.0
                tiltsimg[x,y] = (tilts[x,o]*yv + x)/dszx

    return tiltsimg


def slit_trace_qa(slf, frame, ltrace, rtrace, extslit, desc="",
                  root='trace', normalize=True, use_slitid=None):
    """ Generate a QA plot for the slit traces

    Parameters
    ----------
    slf : class
      An instance of the Science Exposure Class
    frame : ndarray
      trace image
    ltrace : ndarray
      Left slit edge traces
    rtrace : ndarray
      Right slit edge traces
    extslit : ndarray
      Mask of extrapolated slits (True = extrapolated)
    desc : str, optional
      A description to be used as a title for the page
    root : str, optional
      Root name for generating output file, e.g. msflat_01blue_000.fits
    normalize: bool, optional
      Normalize the flat?  If not, use zscale for output
    """

    plt.rcdefaults()
    plt.rcParams['font.family']= 'times new roman'
    ticks_font = font_manager.FontProperties(family='times new roman', style='normal', size=16,
                                             weight='normal', stretch='normal')

    # Outfile
    method = inspect.stack()[0][3]
    outfile = arqa.set_qa_filename(slf.setup, method)
    # if outfil is None:
    #     if '.fits' in root: # Expecting name of msflat FITS file
    #         outfil = root.replace('.fits', '_trc.pdf')
    #         outfil = outfil.replace('MasterFrames', 'Plots')
    #     else:
    #         outfil = root+'.pdf'
    ntrc = ltrace.shape[1]
    ycen = np.arange(frame.shape[0])
    # Normalize flux in the traces
    if normalize:
        nrm_frame = np.zeros_like(frame)
        for ii in range(ntrc):
            xtrc = (ltrace[:, ii] + rtrace[:, ii])/2.
            ixtrc = np.round(xtrc).astype(int)
            # Simple 'extraction'
            dumi = np.zeros((frame.shape[0], 3))
            for jj in range(3):
                dumi[:, jj] = frame[ycen, ixtrc-1+jj]
            trc = np.median(dumi, axis=1)
            # Find portion of the image and normalize
            for yy in ycen:
                xi = max(0, int(ltrace[yy, ii])-3)
                xe = min(frame.shape[1], int(rtrace[yy, ii])+3)
                # Fill + normalize
                nrm_frame[yy, xi:xe] = frame[yy, xi:xe] / trc[yy]
        sclmin, sclmax = 0.4, 1.1
    else:
        nrm_frame = frame.copy()
        nrm_frame[frame > 0.0] = np.sqrt(nrm_frame[frame > 0.0])
        sclmin, sclmax = arplot.zscale(nrm_frame)

    # Plot
    plt.clf()

    ax = plt.gca()
#    set_fonts(ax)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)
    cmm = cm.Greys_r
    mplt = plt.imshow(nrm_frame, origin='lower', cmap=cmm, interpolation=None,
                      extent=(0., frame.shape[1], 0., frame.shape[0]))
    mplt.set_clim(vmin=sclmin, vmax=sclmax)

    # Axes
    plt.xlim(0., frame.shape[1])
    plt.ylim(0., frame.shape[0])
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                    labelbottom='off', labelleft='off')

    # Traces
    iy = int(frame.shape[0]/2.)
    for ii in range(ntrc):
        if extslit[ii] is True:
            ptyp = ':'
        else:
            ptyp = '--'
        # Left
        plt.plot(ltrace[:, ii]+0.5, ycen, 'r'+ptyp, linewidth=0.3, alpha=0.7)
        # Right
        plt.plot(rtrace[:, ii]+0.5, ycen, 'c'+ptyp, linewidth=0.3, alpha=0.7)
        # Label
        if use_slitid:
            slitid, _, _ = arspecobj.get_slitid(slf, use_slitid, ii, ypos=0.5)
            lbl = 'S{:04d}'.format(slitid)
        else:
            lbl = '{0:d}'.format(ii+1)
        plt.text(0.5*(ltrace[iy, ii]+rtrace[iy, ii]), ycen[iy], lbl, color='green', ha='center', size='small')
    # Title
    tstamp = arqa.gen_timestamp()
    if desc == "":
        plt.suptitle(tstamp)
    else:
        plt.suptitle(desc+'\n'+tstamp)

    # Write
    plt.savefig(outfile, dpi=800)
    plt.close()

    plt.rcdefaults()


def trace_slits(slf, mstrace, det, pcadesc="", maskBadRows=False, min_sqm=30.):
    """
    This routine traces the locations of the slit edges

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    mstrace : numpy ndarray
      Calibration frame that will be used to identify slit traces (in most cases, the slit edge)
    det : int
      Index of the detector
    pcadesc : str, optional
      A descriptive string of text to be annotated as a title on the QA PCA plots
    maskBadRows : bool, optional
      Mostly useful for echelle data where the slit edges are bent relative to
      the pixel columns. Do not set this keyword to True if slit edges are
      almost aligned with the pixel columns.
    min_sqm : float, optional
      Minimum error used when detecting a slit edge

    Returns
    -------
    lcenint : ndarray
      Locations of the left slit edges (in physical pixel coordinates)
    rcenint : ndarray
      Locations of the right slit edges (in physical pixel coordinates)
    extrapord : ndarray
      A boolean mask indicating if an order was extrapolated (True = extrapolated)
    """
    dnum = settings.get_dnum(det)
    ednum = 100000  # A large dummy number used for slit edge assignment. ednum should be larger than the number of edges detected

    msgs.info("Preparing trace frame for slit detection")
    # Generate a binned (or smoothed) version of the trace frame
    binarr = ndimage.uniform_filter(mstrace, size=(3, 1),mode='mirror')
    binbpx = slf._bpix[det-1].copy()
    plxbin = slf._pixlocn[det-1][:, :, 0].copy()
    plybin = slf._pixlocn[det-1][:, :, 1].copy()
    if msgs._debug["trace"]:
        # Use this for debugging
        binbpx = np.zeros(mstrace.shape, dtype=np.int)
        xs = np.arange(mstrace.shape[0] * 1.0) * settings.spect[dnum]['xgap']
        xt = 0.5 + np.arange(mstrace.shape[0] * 1.0) + xs
        ys = np.arange(mstrace.shape[1]) * settings.spect[dnum]['ygap'] * settings.spect[dnum]['ysize']
        yt = settings.spect[dnum]['ysize'] * (0.5 + np.arange(mstrace.shape[1] * 1.0)) + ys
        xloc, yloc = np.meshgrid(xt, yt)
        plxbin, plybin = xloc.T, yloc.T
    #    binby = 5
#    binarr = arcyutils.bin_x(mstrace, binby, 0)
#    binbpx = arcyutils.bin_x(slf._bpix[det-1], binby, 0)
#    plxbin = arcyutils.bin_x(slf._pixlocn[det-1][:,:,0], binby, 1)
#    plybin = arcyutils.bin_x(slf._pixlocn[det-1][:,:,1], binby, 1)

    # Specify how many times to repeat the median filter
    medrep = 3
    if len(settings.argflag['trace']['slits']['single']) > 0:
        edgearr = np.zeros(binarr.shape, dtype=np.int)
        # Add a user-defined slit?
        # Syntax is a list of values, 2 per detector that define the slit
        # according to column values.  The 2nd value (for the right edge)
        # must be >0 to be applied.  Example for LRISr [-1, -1, 7, 295]
        # which means the code skips user-definition for the first detector
        # but adds one for the 2nd.
        ledge, redge = (det-1)*2, (det-1)*2+1
        if settings.argflag['trace']['slits']['single'][redge] > 0:
            msgs.warn("Using input slit edges on detector {:d}: [{:g},{:g}]".format(
                    det,
                    settings.argflag['trace']['slits']['single'][ledge],
                    settings.argflag['trace']['slits']['single'][redge]))
            msgs.warn("Better know what you are doing!")
            edgearr[:, settings.argflag['trace']['slits']['single'][ledge]] = -1
            edgearr[:, settings.argflag['trace']['slits']['single'][redge]] = +1
    else:
        # Even better would be to fit the filt/sqrt(abs(binarr)) array with a Gaussian near the maximum in each column
        msgs.info("Detecting slit edges")
        # Generate sigma image
        sqmstrace = np.sqrt(np.abs(binarr))
        for ii in range(medrep):
            sqmstrace = ndimage.median_filter(sqmstrace, size=(3, 7))
        # Make sure there are no spuriously low pixels
        sqmstrace[(sqmstrace < 1.0) & (sqmstrace >= 0.0)] = 1.0
        sqmstrace[(sqmstrace > -1.0) & (sqmstrace <= 0.0)] = -1.0
        # Apply a Sobel filter
        filt = ndimage.sobel(sqmstrace, axis=1, mode=settings.argflag['trace']['slits']['sobel']['mode'])
        msgs.info("Applying bad pixel mask")
        filt *= (1.0 - binbpx)  # Apply to the bad pixel mask
        siglev = np.sign(filt)*(filt**2)/np.maximum(sqmstrace, min_sqm)
        tedges = np.zeros(binarr.shape, dtype=np.float)
        wl = np.where(siglev > +settings.argflag['trace']['slits']['sigdetect'])  # A positive gradient is a left edge
        wr = np.where(siglev < -settings.argflag['trace']['slits']['sigdetect'])  # A negative gradient is a right edge
        tedges[wl] = -1.0
        tedges[wr] = +1.0
        if False:
            import astropy.io.fits as pyfits
            hdu = pyfits.PrimaryHDU(filt)
            hdu.writeto("filt_{0:02d}.fits".format(det), overwrite=True)
            hdu = pyfits.PrimaryHDU(sqmstrace)
            hdu.writeto("sqmstrace_{0:02d}.fits".format(det), overwrite=True)
            hdu = pyfits.PrimaryHDU(binarr)
            hdu.writeto("binarr_{0:02d}.fits".format(det), overwrite=True)
            hdu = pyfits.PrimaryHDU(siglev)
            hdu.writeto("siglev_{0:02d}.fits".format(det), overwrite=True)
        # Clean the edges
        wcl = np.where((ndimage.maximum_filter1d(siglev, 10, axis=1) == siglev) & (tedges == -1))
        wcr = np.where((ndimage.minimum_filter1d(siglev, 10, axis=1) == siglev) & (tedges == +1))
        nedgear = np.zeros(siglev.shape, dtype=np.int)
        nedgear[wcl] = -1
        nedgear[wcr] = +1
        #nedgear = arcytrace.clean_edges(siglev, tedges)
        if maskBadRows:
            msgs.info("Searching for bad pixel rows")
            edgsum = np.sum(nedgear, axis=0)
            sigma = 1.4826*np.median(np.abs(edgsum-np.median(edgsum)))
            w = np.where(np.abs(edgsum) >= 1.5*sigma)[0]
            maskcols = np.unique(np.append(w, np.append(w+1, w-1)))
            msgs.info("Masking {0:d} bad pixel rows".format(maskcols.size))
            for i in range(maskcols.size):
                if maskcols[i] < 0 or maskcols[i] >= nedgear.shape[1]:
                    continue
                nedgear[:, maskcols[i]] = 0
        ######
        msgs.info("Applying bad pixel mask")
        nedgear *= (1-binbpx.astype(np.int))  # Apply to the bad pixel mask
        # eroll = np.roll(binbpx, 1, axis=1)
        # eroll[:,0] = eroll[:,1]
        # nedgear *= (1.0-eroll)  # Apply to the new detection algorithm (with shift)
        # Now roll back
        # nedgear = np.roll(nedgear, -1, axis=1)
        # edgearr = np.zeros_like(nedgear)
        # edgearr[np.where((nedgear == +1) | (tedgear == +1))] = +1
        # edgearr[np.where((nedgear == -1) | (tedgear == -1))] = -1
        sigedg = np.copy(siglev)
        sigedg[np.where(nedgear == 0)] = 0
        sigedg[np.where(np.isinf(sigedg) | np.isnan(sigedg))] = 0
        if settings.argflag['trace']['slits']['number'] > 0:
            # Now identify the number of most significantly detected peaks (specified by the user)
            amnmx = np.argsort(sigedg, axis=1)
            edgearr = np.zeros(binarr.shape, dtype=np.int)
            xsm = np.arange(binarr.shape[0])
            for ii in range(0, settings.argflag['trace']['slits']['number']):
                wset = np.where(sigedg[(xsm, amnmx[:, ii])] != 0)
                edgearr[(wset[0], amnmx[wset[0], ii])] = 1
                wset = np.where(sigedg[(xsm, amnmx[:, amnmx.shape[1] - 1 - ii])] != 0)
                edgearr[(wset[0], amnmx[wset[0], amnmx.shape[1] - 1 - ii])] = -1
        else:
            edgearr = np.copy(nedgear)

    # Assign a number to each of the edges
#    _edgearr = edgearr.copy()
#    t = time.clock()
#    _lcnt, _rcnt = arcytrace.match_edges(_edgearr, ednum)
#    print('Old match_edges: {0} seconds'.format(time.clock() - t))
    __edgearr = edgearr.copy()
#    t = time.clock()
    lcnt, rcnt = new_match_edges(__edgearr, ednum)
#    print('New match_edges: {0} seconds'.format(time.clock() - t))
#    print(lcnt, _lcnt, rcnt, _rcnt)
#    print(np.sum(__edgearr != _edgearr))
#    plt.imshow(_edgearr, origin='lower', interpolation='nearest', aspect='auto')
#    plt.show()
#    plt.imshow(__edgearr, origin='lower', interpolation='nearest', aspect='auto')
#    plt.show()
#    assert np.sum(_lcnt != lcnt) == 0, 'Difference between old and new match_edges, lcnt'
#    assert np.sum(_rcnt != rcnt) == 0, 'Difference between old and new match_edges, rcnt'
#    assert np.sum(__edgearr != _edgearr) == 0, 'Difference between old and new match_edges, edgearr'
    edgearr = __edgearr

    if lcnt >= ednum or rcnt >= ednum:
        msgs.error("Found more edges than allowed by ednum. Set ednum to a larger number.")
    if lcnt == 1:
        letxt = "edge"
    else:
        letxt = "edges"
    if rcnt == 1:
        retxt = "edge"
    else:
        retxt = "edges"
    msgs.info("{0:d} left {1:s} and {2:d} right {3:s} were found in the trace".format(lcnt, letxt, rcnt, retxt))
    if (lcnt == 0) and (rcnt == 0):
        if np.median(binarr) > 500:
            msgs.warn("Found flux but no edges.  Assuming they go to the edge of the detector.")
            edgearr[:, -1] = 2*ednum
            rcnt = 1
            edgearr[:, 0] = -2*ednum
            lcnt = 1
        else:
            msgs.error("Unable to trace any edges"+msgs.newline()+"try a different method to trace the order edges")
    elif rcnt == 0:
        msgs.warn("Unable to find a right edge. Adding one in.")
        # Respecting the BPM (using first column where there is no mask)
        sum_bpm = np.sum(binbpx, axis=0)
        gdi1 = np.max(np.where(sum_bpm == 0)[0])
        # Apply
        edgearr[:, gdi1] = 2*ednum
        rcnt = 1
    elif lcnt == 0:
        msgs.warn("Unable to find a left edge. Adding one in.")
        # Respecting the BPM (using first column where there is no mask)
        sum_bpm = np.sum(binbpx, axis=0)
        gdi0 = np.min(np.where(sum_bpm == 0)[0])
        # Apply
        edgearr[:, gdi0] = -2*ednum
        lcnt = 1
    msgs.info("Assigning slit edge traces")
    # Find the most common set of edges
    minvf, maxvf = plxbin[0, 0], plxbin[-1, 0]
    edgearrcp = edgearr.copy()
    # If slits are set as "close" by the user, take the absolute value
    # of the detections and ignore the left/right edge detections
    if settings.argflag['trace']['slits']['maxgap'] is not None:
        edgearrcp[np.where(edgearrcp < 0)] += 1 + np.max(edgearrcp) - np.min(edgearrcp)
    # Assign left edges
    msgs.info("Assigning left slit edges")
    if lcnt == 1:
        edgearrcp[np.where(edgearrcp <= -2*ednum)] = -ednum
    else:
        assign_slits(binarr, edgearrcp, lor=-1)
    # Assign right edges
    msgs.info("Assigning right slit edges")
    if rcnt == 1:
        edgearrcp[np.where(edgearrcp >= 2*ednum)] = ednum
    else:
        assign_slits(binarr, edgearrcp, lor=+1)
    if settings.argflag['trace']['slits']['maxgap'] is not None:
        vals = np.sort(np.unique(edgearrcp[np.where(edgearrcp != 0)]))
#        print('calling close_edges')
#        exit()
        hasedge = arcytrace.close_edges(edgearrcp, vals, int(settings.argflag['trace']['slits']['maxgap']))
        # Find all duplicate edges
        edgedup = vals[np.where(hasedge == 1)]
        if edgedup.size > 0:
            for jj in range(edgedup.size):
                # Raise all remaining edges by one
                if jj != edgedup.size-1:
                    edgedup[jj+1:] += 1
                edgearrcp[np.where(edgearrcp > edgedup[jj])] += 1
                # Now investigate the duplicate
                wdup = np.where(edgearrcp == edgedup[jj])
                alldup = edgearr[wdup]
                alldupu = np.unique(alldup)
                cntr = Counter(edg for edg in alldup)
                commn = cntr.most_common(alldupu.size)
                shftsml = np.zeros(len(commn))
                shftarr = np.zeros(wdup[0].size, dtype=np.int)
                wghtarr = np.zeros(len(commn))
                duploc = [None for ii in range(len(commn))]
                for ii in range(len(commn)):
                    wghtarr[ii] = commn[ii][1]
                    duploc[ii] = np.where(edgearr[wdup] == commn[ii][0])
                changesmade = True
                while changesmade:
                    # Keep shifting pixels until the best match is found
                    changesmade = False
                    # First calculate the old model
                    cf = arutils.func_fit(wdup[0], wdup[1]+shftarr,
                                          settings.argflag['trace']['slits']['function'],
                                          settings.argflag['trace']['slits']['polyorder'],
                                          minv=0, maxv=binarr.shape[0]-1)
                    cenmodl = arutils.func_val(cf, np.arange(binarr.shape[0]),
                                               settings.argflag['trace']['slits']['function'],
                                               minv=0, maxv=binarr.shape[0]-1)
                    chisqold = np.abs(cenmodl[wdup[0]]-wdup[1]-shftarr).sum()
                    for ii in range(1, len(commn)):
                        # Shift by +1
                        adj = np.zeros(wdup[0].size)
                        adj[duploc[ii]] += 1
                        cf = arutils.func_fit(wdup[0], wdup[1]+shftarr+adj,
                                              settings.argflag['trace']['slits']['function'],
                                              settings.argflag['trace']['slits']['polyorder'],
                                              minv=0, maxv=binarr.shape[0]-1)
                        cenmodl = arutils.func_val(cf, np.arange(binarr.shape[0]),
                                                   settings.argflag['trace']['slits']['function'],
                                                   minv=0, maxv=binarr.shape[0]-1)
                        chisqp = np.abs(cenmodl[wdup[0]]-wdup[1]-shftarr-adj).sum()
                        # Shift by -1
                        adj = np.zeros(wdup[0].size)
                        adj[duploc[ii]] -= 1
                        cf = arutils.func_fit(wdup[0], wdup[1]+shftarr+adj,
                                              settings.argflag['trace']['slits']['function'],
                                              settings.argflag['trace']['slits']['polyorder'],
                                              minv=0, maxv=binarr.shape[0]-1)
                        cenmodl = arutils.func_val(cf, np.arange(binarr.shape[0]),
                                                   settings.argflag['trace']['slits']['function'],
                                                   minv=0, maxv=binarr.shape[0]-1)
                        chisqm = np.abs(cenmodl[wdup[0]]-wdup[1]-shftarr-adj).sum()
                        # Test which solution is best:
                        if chisqold < chisqp and chisqold < chisqm:
                            # No changes are needed
                            continue
                        else:
                            changesmade = True
                            if chisqp < chisqm:
                                shftarr[duploc[ii]] += 1
                                shftsml[ii] += 1
                            else:
                                shftarr[duploc[ii]] -= 1
                                shftsml[ii] -= 1
                # Find the two most common edges
                cntr = Counter(sarr for sarr in shftarr)
                commn = cntr.most_common(2)
                if commn[0][0] > commn[1][0]:  # Make sure that suffix 'a' is assigned the leftmost edge
                    wdda = np.where(shftarr == commn[0][0])
                    wddb = np.where(shftarr == commn[1][0])
                    shadj = commn[0][0] - commn[1][0]
                else:
                    wdda = np.where(shftarr == commn[1][0])
                    wddb = np.where(shftarr == commn[0][0])
                    shadj = commn[1][0] - commn[0][0]
                wvla = np.unique(edgearr[wdup][wdda])
                wvlb = np.unique(edgearr[wdup][wddb])
                # Now generate the dual edge
#                print('calling dual_edge')
#                exit()
                arcytrace.dual_edge(edgearr, edgearrcp, wdup[0], wdup[1], wvla, wvlb, shadj,
                                    int(settings.argflag['trace']['slits']['maxgap']), edgedup[jj])
        # Now introduce new edge locations
        vals = np.sort(np.unique(edgearrcp[np.where(edgearrcp != 0)]))
#        print('calling close_slits')
#        exit()
        edgearrcp = arcytrace.close_slits(binarr, edgearrcp, vals, int(settings.argflag['trace']['slits']['maxgap']))
    # Update edgearr
    edgearr = edgearrcp.copy()
    iterate = True
    # Get the final estimates of left and right edges
    eaunq = np.unique(edgearr)
    lcnt = np.where(eaunq < 0)[0].size
    rcnt = np.where(eaunq > 0)[0].size
    if lcnt == 0:
        msgs.warn("Unable to find a left edge. Adding one in.")
        edgearr[:, 0] = -2 * ednum
        lcnt = 1
    if rcnt == 0:
        msgs.warn("Unable to find a right edge. Adding one in.")
        edgearr[:, -1] = 2 * ednum
        rcnt = 1
    if (lcnt == 1) & (rcnt > 1):
        msgs.warn("Only one left edge, and multiple right edges.")
        msgs.info("Restricting right edge detection to the most significantly detected edge.")
        wtst = np.where(eaunq > 0)[0]
        bval, bidx = -np.median(siglev[np.where(edgearr == eaunq[wtst[0]])]), 0
        for r in range(1, rcnt):
            wed = np.where(edgearr == eaunq[wtst[r]])
            tstv = -np.median(siglev[wed])
            if tstv > bval:
                bval = tstv
                bidx = r
        edgearr[np.where((edgearr > 0) & (edgearr != eaunq[wtst[bidx]]))] = 0
        edgearr[np.where(edgearr > 0)] = +ednum  # Reset the edge value
        rcnt = 1
    if (lcnt > 1) & (rcnt == 1):
        msgs.warn("Only one right edge, and multiple left edges.")
        msgs.info("Restricting left edge detection to the most significantly detected edge.")
        wtst = np.where(eaunq < 0)[0]
        bval, bidx = np.median(siglev[np.where(edgearr == eaunq[wtst[0]])]), 0
        for r in range(1, lcnt):
            wed = np.where(edgearr == eaunq[wtst[r]])
            tstv = np.median(siglev[wed])
            if tstv > bval:
                bval = tstv
                bidx = r
        edgearr[np.where((edgearr < 0) & (edgearr != eaunq[wtst[bidx]]))] = 0
        edgearr[np.where(edgearr < 0)] = -ednum  # Reset the edge value
        lcnt = 1
    if lcnt == 1:
        letxt = "edge"
    else:
        letxt = "edges"
    if rcnt == 1:
        retxt = "edge"
    else:
        retxt = "edges"
    msgs.info("{0:d} left {1:s} and {2:d} right {3:s} were found in the trace".format(lcnt, letxt, rcnt, retxt))
    while iterate:
        # Calculate the minimum and maximum left/right edges
        ww = np.where(edgearr < 0)
        lmin, lmax = -np.max(edgearr[ww]), -np.min(edgearr[ww])  # min/max are switched because of the negative signs
        ww = np.where(edgearr > 0)
        rmin, rmax = np.min(edgearr[ww]), np.max(edgearr[ww])  # min/max are switched because of the negative signs
        #msgs.info("Ignoring any slit that spans < {0:3.2f}x{1:d} pixels on the detector".format(settings.argflag['trace']['slits']['fracignore'], int(edgearr.shape[0]*binby)))
        msgs.info("Ignoring any slit that spans < {0:3.2f}x{1:d} pixels on the detector".format(settings.argflag['trace']['slits']['fracignore'], int(edgearr.shape[0])))
        fracpix = int(settings.argflag['trace']['slits']['fracignore']*edgearr.shape[0])
#        print('calling ignore_orders')
#        t = time.clock()
#        _edgearr = edgearr.copy()
#        _lnc, _lxc, _rnc, _rxc, _ldarr, _rdarr = arcytrace.ignore_orders(_edgearr, fracpix, lmin, lmax, rmin, rmax)
#        print('Old ignore_orders: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        __edgearr = edgearr.copy()
        lnc, lxc, rnc, rxc, ldarr, rdarr = new_ignore_orders(__edgearr, fracpix, lmin, lmax, rmin,
                                                             rmax)
#        print('New ignore_orders: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_lnc != lnc) == 0, 'Difference between old and new ignore_orders, lnc'
#        assert np.sum(_lxc != lxc) == 0, 'Difference between old and new ignore_orders, lxc'
#        assert np.sum(_rnc != rnc) == 0, 'Difference between old and new ignore_orders, rnc'
#        assert np.sum(_rxc != rxc) == 0, 'Difference between old and new ignore_orders, rxc'
#        assert np.sum(_ldarr != ldarr) == 0, 'Difference between old and new ignore_orders, ldarr'
#        assert np.sum(_rdarr != rdarr) == 0, 'Difference between old and new ignore_orders, rdarr'
#        assert np.sum(__edgearr != _edgearr) == 0, 'Difference between old and new ignore_orders, edgearr'

        edgearr = __edgearr

        lmin += lnc
        rmin += rnc
        lmax -= lxc
        rmax -= rxc
        iterate = False
        if settings.argflag['trace']['slits']['number'] == 1:  # Another check on slits for singleSlit
            if lmax < lmin:
                msgs.warn("Unable to find a left edge2. Adding one in.")
                iterate = True
                edgearr[:,0] = -2*ednum
                lcnt = 1
            if rmax < rmin:
                msgs.warn("Unable to find a right edge2. Adding one in.")
                iterate = True
                edgearr[:,-1] = 2*ednum
                rcnt = 1
    # Trace left slit edges
    # First, determine the model for the most common left slit edge
    wcm = np.where(edgearr < 0)
    cntr = Counter(edg for edg in edgearr[wcm])
    commn = cntr.most_common(1)
    wedx, wedy = np.where(edgearr == commn[0][0])
    msk, cf = arutils.robust_polyfit(wedx, wedy,
                                     settings.argflag['trace']['slits']['polyorder'],
                                     function=settings.argflag['trace']['slits']['function'],
                                     minv=0, maxv=binarr.shape[0]-1)
    cenmodl = arutils.func_val(cf, np.arange(binarr.shape[0]),
                               settings.argflag['trace']['slits']['function'],
                               minv=0, maxv=binarr.shape[0]-1)

    msgs.info("Fitting left slit traces")
    lcoeff = np.zeros((1+settings.argflag['trace']['slits']['polyorder'], lmax-lmin+1))
    ldiffarr = np.zeros(lmax-lmin+1)
    lwghtarr = np.zeros(lmax-lmin+1)
    lnmbrarr = np.zeros(lmax-lmin+1)
    offs = cenmodl[int(binarr.shape[0]/2)]
#    lfail = np.array([])
#    minvf, maxvf = slf._pixlocn[det-1][0, 0, 0], slf._pixlocn[det-1][-1, 0, 0]
    for i in range(lmin, lmax+1):
        w = np.where(edgearr == -i)
        if np.size(w[0]) <= settings.argflag['trace']['slits']['polyorder']+2:
            # lfail = np.append(lfail,i-lmin)
            continue
        tlfitx = plxbin[w]
        tlfity = plybin[w]
        ldiffarr[i-lmin] = np.mean(w[1]-cenmodl[w[0]]) + offs
        lwghtarr[i-lmin] = np.size(w[0])/float(binarr.shape[0])
        lnmbrarr[i-lmin] = -i
        #lcoeff[:, i-lmin] = arutils.func_fit(tlfitx, tlfity, settings.argflag['trace']['slits']['function'],
        #                                     settings.argflag['trace']['slits']['polyorder'], minv=minvf, maxv=maxvf)
        msk, lcoeff[:, i-lmin] = arutils.robust_polyfit(tlfitx, tlfity,
                                                        settings.argflag['trace']['slits']['polyorder'],
                                                        function=settings.argflag['trace']['slits']['function'],
                                                        minv=minvf, maxv=maxvf)
#		xv=np.linspace(0,edgearr.shape[0])
#		yv=np.polyval(coeffl[i-lmin,:],xv)
#		plt.plot(w[0],w[1],'ro')
#		plt.plot(xv,yv,'r-')
#		plt.show()
#		plt.clf()
    # Trace right slit edges
    # First, determine the model for the most common right slit edge
    wcm = np.where(edgearr > 0)
    cntr = Counter(edg for edg in edgearr[wcm])
    commn = cntr.most_common(1)
    wedx, wedy = np.where(edgearr == commn[0][0])
    msk, cf = arutils.robust_polyfit(wedx, wedy,
                                     settings.argflag['trace']['slits']['polyorder'],
                                     function=settings.argflag['trace']['slits']['function'],
                                     minv=0, maxv=binarr.shape[0]-1)
    cenmodl = arutils.func_val(cf, np.arange(binarr.shape[0]),
                               settings.argflag['trace']['slits']['function'],
                               minv=0, maxv=binarr.shape[0]-1)

    msgs.info("Fitting right slit traces")
    rcoeff = np.zeros((1+settings.argflag['trace']['slits']['polyorder'], rmax-rmin+1))
    rdiffarr = np.zeros(rmax-rmin+1)
    rwghtarr = np.zeros(rmax-rmin+1)
    rnmbrarr = np.zeros(rmax-rmin+1)
    offs = cenmodl[int(binarr.shape[0]/2)]
#	rfail = np.array([])
    for i in range(rmin, rmax+1):
        w = np.where(edgearr == i)
        if np.size(w[0]) <= settings.argflag['trace']['slits']['polyorder']+2:
#			rfail = np.append(rfail, i-rmin)
            continue
        tlfitx = plxbin[w]
        tlfity = plybin[w]
        #rcoeff[:, i-rmin] = arutils.func_fit(tlfitx, tlfity, settings.argflag['trace']['slits']['function'],
        #                                     settings.argflag['trace']['slits']['polyorder'], minv=minvf, maxv=maxvf)
        rdiffarr[i-rmin] = np.mean(w[1]-cenmodl[w[0]]) + offs
        rwghtarr[i-rmin] = np.size(w[0])/float(binarr.shape[0])
        rnmbrarr[i-rmin] = i
        msk, rcoeff[:, i-rmin] = arutils.robust_polyfit(tlfitx, tlfity,
                                                       settings.argflag['trace']['slits']['polyorder'],
                                                       function=settings.argflag['trace']['slits']['function'],
                                                       minv=minvf, maxv=maxvf)
    # Check if no further work is needed (i.e. there only exists one order)
    if (lmax+1-lmin == 1) and (rmax+1-rmin == 1):
        # Just a single order has been identified (i.e. probably longslit)
        msgs.info("Only one slit was identified. Should be a longslit.")
        xint = slf._pixlocn[det-1][:, 0, 0]
        lcenint = np.zeros((mstrace.shape[0], 1))
        rcenint = np.zeros((mstrace.shape[0], 1))
        lcenint[:, 0] = arutils.func_val(lcoeff[:, 0], xint, settings.argflag['trace']['slits']['function'],
                                         minv=minvf, maxv=maxvf)
        rcenint[:, 0] = arutils.func_val(rcoeff[:, 0], xint, settings.argflag['trace']['slits']['function'],
                                         minv=minvf, maxv=maxvf)
        return lcenint, rcenint, np.zeros(1, dtype=np.bool)
    msgs.info("Synchronizing left and right slit traces")
    # Define the array of pixel values along the dispersion direction
    xv = plxbin[:, 0]
    num = (lmax-lmin)//2
    lval = lmin + num  # Pick an order, somewhere in between lmin and lmax
    lv = (arutils.func_val(lcoeff[:, lval-lmin], xv, settings.argflag['trace']['slits']['function'], minv=minvf, maxv=maxvf)+0.5).astype(np.int)
    if np.any(lv < 0) or np.any(lv+1 >= binarr.shape[1]):
        msgs.warn("At least one slit is poorly traced")
        msgs.info("Refer to the manual, and adjust the input trace parameters")
        msgs.error("Cannot continue without a successful trace")
    mnvalp = np.median(binarr[:, lv+1])  # Go one row above and one row below an order edge,
    mnvalm = np.median(binarr[:, lv-1])  # then see which mean value is greater.

    """
    lvp = (arutils.func_val(lcoeff[:,lval+1-lmin],xv,settings.argflag['trace']['slits']['function'],min=minvf,max=maxvf)+0.5).astype(np.int)
    edgbtwn = arcytrace.find_between(edgearr,lv,lvp,1)
    print lval, edgbtwn
    # edgbtwn is a 3 element array that determines what is between two adjacent left edges
    # edgbtwn[0] is the next right order along, from left order lval
    # edgbtwn[1] is only !=-1 when there's an order overlap.
    # edgebtwn[2] is only used when a left order is found before a right order
    if edgbtwn[0] == -1 and edgbtwn[1] == -1:
        rsub = edgbtwn[2]-(lval) # There's an order overlap
    elif edgbtwn[1] == -1: # No overlap
        rsub = edgbtwn[0]-(lval)
    else: # There's an order overlap
        rsub = edgbtwn[1]-(lval)
    """
    if msgs._debug['trace']:
        debugger.set_trace()
    if mnvalp > mnvalm:
        lvp = (arutils.func_val(lcoeff[:, lval+1-lmin], xv, settings.argflag['trace']['slits']['function'],
                                minv=minvf, maxv=maxvf)+0.5).astype(np.int)
#        t = time.clock()
#        _edgbtwn = arcytrace.find_between(edgearr, lv, lvp, 1)
#        print('Old find_between: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        edgbtwn = new_find_between(edgearr, lv, lvp, 1)
#        print('New find_between: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_edgbtwn != edgbtwn) == 0, 'Difference between old and new find_between'

        # edgbtwn is a 3 element array that determines what is between two adjacent left edges
        # edgbtwn[0] is the next right order along, from left order lval
        # edgbtwn[1] is only !=-1 when there's an order overlap.
        # edgebtwn[2] is only used when a left order is found before a right order
        if edgbtwn[0] == -1 and edgbtwn[1] == -1:
            rsub = edgbtwn[2]-lval  # There's an order overlap
        elif edgbtwn[1] == -1:  # No overlap
            rsub = edgbtwn[0]-lval
        else:  # There's an order overlap
            rsub = edgbtwn[1]-lval
    else:
        lvp = (arutils.func_val(lcoeff[:, lval-1-lmin], xv, settings.argflag['trace']['slits']['function'],
                                minv=minvf, maxv=maxvf)+0.5).astype(np.int)
#        t = time.clock()
#        _edgbtwn = arcytrace.find_between(edgearr, lvp, lv, -1)
#        print('Old find_between: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        edgbtwn = new_find_between(edgearr, lvp, lv, -1)
#        assert np.sum(_edgbtwn != edgbtwn) == 0, 'Difference between old and new find_between'
#        print('New find_between: {0} seconds'.format(time.clock() - t))

        if edgbtwn[0] == -1 and edgbtwn[1] == -1:
            rsub = edgbtwn[2]-(lval-1)  # There's an order overlap
        elif edgbtwn[1] == -1:  # No overlap
            rsub = edgbtwn[0]-(lval-1)
        else:  # There's an order overlap
            rsub = edgbtwn[1]-(lval-1)

#	rva = arutils.func_val(rcoeff[:,0],midval,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#	for i in range(1,rmax-rmin+1):
#		rvb = arutils.func_val(rcoeff[:,i],midval,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#		if rvb > lv and rva < lv:
#			lvp = arutils.func_val(lcoeff[:,500-lmin],midval*2.0,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#			lvm = arutils.func_val(lcoeff[:,500-lmin],0.0,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#			rvap = arutils.func_val(rcoeff[:,i-1],midval*2.0,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#			rvam = arutils.func_val(rcoeff[:,i-1],0.0,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#			rvbp = arutils.func_val(rcoeff[:,i],midval*2.0,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#			rvbm = arutils.func_val(rcoeff[:,i],0.0,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)
#			mina = np.min([lv-rva,lvp-rvap,lvm-rvam])
#			minb = np.min([rvb-lv,rvbp-lvp,rvbm-lvm])
#			if mina <= 1.0 or minb <= 0.0:
#				msgs.error("Orders are too close or the fitting quality is too poor")
#			yva = np.arange(0.0,mina)
#			yvb = np.arange(0.0,minb)
#			xmga, ymga = np.meshgrid(xv,yva)
#			xmgb, ymgb = np.meshgrid(xv,yvb)
#			ymga += np.array([arutils.func_val(rcoeff[:,i-1],xv,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)])
#			ymgb += np.array([arutils.func_val(rcoeff[:,i],xv,settings.argflag['trace']['slits']['function'],min=xvmn,max=xvmx)])
#			xmga = xmga.flatten().astype(np.int)
#			ymga = ymga.flatten().astype(np.int)
#			xmgb = xmgb.flatten().astype(np.int)
#			ymgb = ymgb.flatten().astype(np.int)
#			meda = np.median(binarr[xmga,ymga])
#			medb = np.median(binarr[xmgb,ymgb])
#			if meda > medb:
#				rsub = rmin+i-1-500
#			else:
#				rsub = rmin+i-500
#			break
#		else:
#			rva = rvb
    msgs.info("Relabelling slit edges")
    rsub = int(round(rsub))
    if lmin < rmin-rsub:
        esub = lmin - (settings.argflag['trace']['slits']['pca']['extrapolate']['neg']+1)
    else:
        esub = (rmin-rsub) - (settings.argflag['trace']['slits']['pca']['extrapolate']['neg']+1)

    wl = np.where(edgearr < 0)
    wr = np.where(edgearr > 0)
    edgearr[wl] += esub
    edgearr[wr] -= (esub+rsub)
    lnmbrarr += esub
    rnmbrarr -= (esub+rsub)

    # Insert new rows into coefficients arrays if rsub != 0 (if orders were not labelled correctly, there will be a mismatch for the lcoeff and rcoeff)
    almin, almax = -np.max(edgearr[wl]), -np.min(edgearr[wl]) # min and max switched because left edges have negative values
    armin, armax = np.min(edgearr[wr]), np.max(edgearr[wr])
    nmord = settings.argflag['trace']['slits']['polyorder']+1
    if armin != almin:
        if armin < almin:
            lcoeff = np.append(np.zeros((nmord, almin-armin)), lcoeff, axis=1)
            ldiffarr = np.append(np.zeros(almin-armin), ldiffarr)
            lnmbrarr = np.append(np.zeros(almin-armin), lnmbrarr)
            lwghtarr = np.append(np.zeros(almin-armin), lwghtarr)
        else:
            rcoeff = np.append(np.zeros((nmord, armin-almin)), rcoeff, axis=1)
            rdiffarr = np.append(np.zeros(armin-almin), rdiffarr)
            rnmbrarr = np.append(np.zeros(armin-almin), rnmbrarr)
            rwghtarr = np.append(np.zeros(armin-almin), rwghtarr)
    if armax != almax:
        if armax < almax:
            rcoeff = np.append(rcoeff, np.zeros((nmord, almax-armax)), axis=1)
            rdiffarr = np.append(rdiffarr, np.zeros(almax-armax))
            rnmbrarr = np.append(rnmbrarr, np.zeros(almax-armax))
            rwghtarr = np.append(rwghtarr, np.zeros(almax-armax))
        else:
            lcoeff = np.append(lcoeff, np.zeros((nmord, armax-almax)), axis=1)
            ldiffarr = np.append(ldiffarr, np.zeros(armax-almax))
            lnmbrarr = np.append(lnmbrarr, np.zeros(armax-almax))
            lwghtarr = np.append(lwghtarr, np.zeros(armax-almax))

    # import astropy.io.fits as pyfits
    # hdu = pyfits.PrimaryHDU(edgearr)
    # hdu.writeto("edgearr_{0:02d}.fits".format(det))

    # Now consider traces where both the left and right edges are detected
    ordunq = np.unique(edgearr)
    lunqt = ordunq[np.where(ordunq < 0)[0]]
    runqt = ordunq[np.where(ordunq > 0)[0]]
    lunq = np.arange(lunqt.min(), lunqt.max()+1)
    runq = np.arange(runqt.min(), runqt.max()+1)
    # Determine which orders are detected on both the left and right edge
    gord = np.intersect1d(-lunq, runq, assume_unique=True)
    # We need to ignore the orders labelled rfail and lfail.
    lg = np.where(np.in1d(-lunq, gord))[0]
    rg = np.where(np.in1d(runq, gord))[0]
    lgm = np.where(np.in1d(-lunq, gord, invert=True))[0]
    rgm = np.where(np.in1d(runq, gord, invert=True))[0]
    maxord = np.max(np.append(gord, np.append(-lunq[lgm], runq[rgm])))
    lcent = arutils.func_val(lcoeff[:,-lunq[lg][::-1]-1-settings.argflag['trace']['slits']['pca']['extrapolate']['neg']], xv,
                             settings.argflag['trace']['slits']['function'], minv=minvf, maxv=maxvf)
    rcent = arutils.func_val(rcoeff[:,runq[rg]-1-settings.argflag['trace']['slits']['pca']['extrapolate']['neg']], xv,
                             settings.argflag['trace']['slits']['function'], minv=minvf, maxv=maxvf)
    slitcen = 0.5*(lcent+rcent).T
    ##############
    if settings.argflag['trace']['slits']['pca']['type'] == 'order':
        #maskord = np.where((np.all(lcoeff[:,lg],axis=0)==False)|(np.all(rcoeff[:,rg],axis=0)==False))[0]
        maskord = np.where((np.all(lcoeff, axis=0) == False) | (np.all(rcoeff, axis=0) == False))[0]
        ordsnd = np.arange(min(almin, armin), max(almax, armax)+1)
        totord = ordsnd[-1]+settings.argflag['trace']['slits']['pca']['extrapolate']['pos']
        # Identify the orders to be extrapolated during reconstruction
        extrapord = (1.0-np.in1d(np.linspace(1.0, totord, totord), gord).astype(np.int)).astype(np.bool)
        msgs.info("Performing a PCA on the order edges")
        ofit = settings.argflag['trace']['slits']['pca']['params']
        lnpc = len(ofit)-1
        msgs.work("May need to do a check here to make sure ofit is reasonable")
        coeffs = arutils.func_fit(xv, slitcen, settings.argflag['trace']['slits']['function'],
                                  settings.argflag['trace']['slits']['polyorder'], minv=minvf, maxv=maxvf)
        for i in range(ordsnd.size):
            if i in maskord:
                coeffs = np.insert(coeffs, i, 0.0, axis=1)
                slitcen = np.insert(slitcen, i, 0.0, axis=1)
                lcent = np.insert(lcent, i, 0.0, axis=0)
                rcent = np.insert(rcent, i, 0.0, axis=0)
        xcen = xv[:, np.newaxis].repeat(ordsnd.size, axis=1)
        fitted, outpar = arpca.basis(xcen, slitcen, coeffs, lnpc, ofit, x0in=ordsnd, mask=maskord,
                                     skipx0=False, function=settings.argflag['trace']['slits']['function'])
        if not msgs._debug['no_qa']:
#            arqa.pca_plot(slf, outpar, ofit, "Slit_Trace", pcadesc=pcadesc)
            arpca.pca_plot(slf, outpar, ofit, "Slit_Trace", pcadesc=pcadesc)
        # Extrapolate the remaining orders requested
        orders = 1+np.arange(totord)
        extrap_cent, outpar = arpca.extrapolate(outpar, orders, function=settings.argflag['trace']['slits']['function'])
        # Fit a function for the difference between left and right edges.
        diff_coeff, diff_fit = arutils.polyfitter2d(rcent-lcent, mask=maskord,
                                                    order=settings.argflag['trace']['slits']['diffpolyorder'])
        # Now extrapolate the order difference
        ydet = np.linspace(0.0, 1.0, lcent.shape[0])
        ydetd = ydet[1]-ydet[0]
        lnum = ordsnd[0]-1.0
        ydet = np.append(-ydetd*np.arange(1.0, 1.0+lnum)[::-1], ydet)
        ydet = np.append(ydet, 1.0+ydetd*np.arange(1.0, 1.0+settings.argflag['trace']['slits']['pca']['extrapolate']['pos']))
        xde, yde = np.meshgrid(np.linspace(0.0, 1.0, lcent.shape[1]), ydet)
        extrap_diff = arutils.polyval2d(xde, yde, diff_coeff).T
        msgs.info("Refining the trace for reconstructed and predicted orders")
        # NOTE::  MIGHT NEED TO APPLY THE BAD PIXEL MASK HERE TO BINARR
        msgs.work("Should the bad pixel mask be applied to the frame here?")
        refine_cent, outpar = refine_traces(binarr, outpar, extrap_cent, extrap_diff,
                                            [gord[0]-orders[0], orders[-1]-gord[-1]], orders,
                                            ofit[0], slf._pixlocn[det-1],
                                            function=settings.argflag['trace']['slits']['function'])
        # Generate the left and right edges
        lcen = refine_cent - 0.5*extrap_diff
        rcen = refine_cent + 0.5*extrap_diff
        # lcen = extrap_cent - 0.5*extrap_diff
        # rcen = extrap_cent + 0.5*extrap_diff
    elif settings.argflag['trace']['slits']['pca']['type'] == 'pixel':
        maskord = np.where((np.all(lcoeff, axis=0) == False) | (np.all(rcoeff, axis=0) == False))[0]
        allord = np.arange(ldiffarr.shape[0])
        ww = np.where(np.in1d(allord, maskord) == False)[0]
        # Unmask where an order edge is located
        maskrows = np.ones(binarr.shape[1], dtype=np.int)
        ldiffarr = np.round(ldiffarr[ww]).astype(np.int)
        rdiffarr = np.round(rdiffarr[ww]).astype(np.int)
        maskrows[ldiffarr] = 0
        maskrows[rdiffarr] = 0
        # Extract the slit edge ID numbers associated with the acceptable traces
        lnmbrarr = lnmbrarr[ww]
        rnmbrarr = rnmbrarr[ww]
        # Fill in left/right coefficients
        tcoeff = np.ones((settings.argflag['trace']['slits']['polyorder']+1, binarr.shape[1]))
        tcoeff[:, ldiffarr] = lcoeff[:, ww]
        tcoeff[:, rdiffarr] = rcoeff[:, ww]
        # Weight the PCA fit by the number of detections in each slit edge
        pxwght = np.zeros(binarr.shape[1])
        pxwght[ldiffarr] = lwghtarr[ww]
        pxwght[rdiffarr] = rwghtarr[ww]
        maskrw = np.where(maskrows == 1)[0]
        maskrw.sort()
        extrap_row = maskrows.copy()
        xv = np.arange(binarr.shape[0])
        # trace values
        trcval = arutils.func_val(tcoeff, xv, settings.argflag['trace']['slits']['function'],
                                  minv=minvf, maxv=maxvf).T
        msgs.work("May need to do a check here to make sure ofit is reasonable")
        ofit = settings.argflag['trace']['slits']['pca']['params']
        lnpc = len(ofit)-1
        # Only do a PCA if there are enough good slits
        if np.sum(1.0-extrap_row) > ofit[0]+1:
            # Perform a PCA on the locations of the slits
            msgs.info("Performing a PCA on the slit traces")
            ordsnd = np.arange(binarr.shape[1])
            xcen = xv[:, np.newaxis].repeat(binarr.shape[1], axis=1)
            fitted, outpar = arpca.basis(xcen, trcval, tcoeff, lnpc, ofit, weights=pxwght,
                                         x0in=ordsnd, mask=maskrw, skipx0=False,
                                         function=settings.argflag['trace']['slits']['function'])
            if not msgs._debug['no_qa']:
#                arqa.pca_plot(slf, outpar, ofit, "Slit_Trace", pcadesc=pcadesc, addOne=False)
                arpca.pca_plot(slf, outpar, ofit, "Slit_Trace", pcadesc=pcadesc, addOne=False)
            # Now extrapolate to the whole detector
            pixpos = np.arange(binarr.shape[1])
            extrap_trc, outpar = arpca.extrapolate(outpar, pixpos,
                                                   function=settings.argflag['trace']['slits']['function'])
            # Extract the resulting edge traces
            lcen = extrap_trc[:, ldiffarr]
            rcen = extrap_trc[:, rdiffarr]
            # Perform a final shift fit to ensure the traces closely follow the edge detections
            for ii in range(lnmbrarr.size):
                wedx, wedy = np.where(edgearr == lnmbrarr[ii])
                shft = np.mean(lcen[wedx, ii]-wedy)
                lcen[:, ii] -= shft
            for ii in range(rnmbrarr.size):
                wedx, wedy = np.where(edgearr == rnmbrarr[ii])
                shft = np.mean(rcen[wedx, ii]-wedy)
                rcen[:, ii] -= shft
        else:
            allord = np.arange(lcent.shape[0])
            maskord = np.where((np.all(lcent, axis=1) == False) | (np.all(rcent, axis=1) == False))[0]
            ww = np.where(np.in1d(allord, maskord) == False)[0]
            lcen = lcent[ww, :].T.copy()
            rcen = rcent[ww, :].T.copy()
        extrapord = np.zeros(lcen.shape[1], dtype=np.bool)
    else:
        allord = np.arange(lcent.shape[0])
        maskord = np.where((np.all(lcent, axis=1) == False) | (np.all(rcent, axis=1) == False))[0]
        ww = np.where(np.in1d(allord, maskord) == False)[0]
        lcen = lcent[ww, :].T.copy()
        rcen = rcent[ww, :].T.copy()
        extrapord = np.zeros(lcen.shape[1], dtype=np.bool)

    # Remove any slits that are completely off the detector
    nslit = lcen.shape[1]
    mask = np.zeros(nslit)
    for o in range(nslit):
        if np.min(lcen[:, o]) > mstrace.shape[1]:
            mask[o] = 1
            msgs.info("Slit {0:d} is off the detector - ignoring this slit".format(o+1))
        elif np.max(rcen[:, o]) < 0:
            mask[o] = 1
            msgs.info("Slit {0:d} is off the detector - ignoring this slit".format(o + 1))
    wok = np.where(mask == 0)[0]
    lcen = lcen[:, wok]
    rcen = rcen[:, wok]

    # Interpolate the best solutions for all orders with a cubic spline
    #msgs.info("Interpolating the best solutions for all orders with a cubic spline")
    # zmin, zmax = arplot.zscale(binarr)
    # plt.clf()
    # extnt = (slf._pixlocn[0,0,1], slf._pixlocn[0,-1,1], slf._pixlocn[0,0,0], slf._pixlocn[-1,0,0])
    # implot = plt.imshow(binarr, extent=extnt, origin='lower', interpolation='none', aspect='auto')
    # implot.set_cmap("gray")
    # plt.colorbar()
    # implot.set_clim(zmin, zmax)
    #xint = slf._pixlocn[det-1][:,0,0]
    #lcenint = np.zeros((mstrace.shape[0], lcen.shape[1]))
    #rcenint = np.zeros((mstrace.shape[0], rcen.shape[1]))
    lcenint = lcen.copy()
    rcenint = rcen.copy()
    #for i in range(lcen.shape[1]):
        # if i+1 not in gord: pclr = '--'
        # else: pclr = '-'
        # plt.plot(xv_corr, lcen_corr[:,i], 'g'+pclr, linewidth=0.1)
        # plt.plot(xv_corr, rcen_corr[:,i], 'b'+pclr, linewidth=0.1)
        #lcenint[:,i] = arutils.spline_interp(xint, xv, lcen[:,i])
        #rcenint[:,i] = arutils.spline_interp(xint, xv, rcen[:,i])
#       plt.plot(xint,lcenint[:,i],'k.')
#       plt.plot(xint,rcenint[:,i],'k.')
#       plt.show()
    # if tracedesc != "":
    #     plt.title(tracedesc)
    # slf._qa.savefig(dpi=1200, orientation='portrait', bbox_inches='tight')
    # plt.close()

    # Illustrate where the orders fall on the detector (physical units)
    if msgs._debug['trace']:
        viewer, ch = ginga.show_image(mstrace)
        ginga.show_slits(viewer, ch, lcenint, rcenint)
        debugger.set_trace()
    if settings.argflag['run']['qa']:
        msgs.work("Not yet setup with ginga")
    return lcenint, rcenint, extrapord


def refine_traces(binarr, outpar, extrap_cent, extrap_diff, extord, orders,
                  fitord, locations, function='polynomial'):
    """
    Parameters
    ----------
    binarr
    outpar
    extrap_cent
    extrap_diff
    extord
    orders
    fitord
    locations
    function

    Returns
    -------

    """
    # Refine the orders in the positive direction
    i = extord[1]
    hiord = phys_to_pix(extrap_cent[:, -i-2], locations, 1)
    nxord = phys_to_pix(extrap_cent[:, -i-1], locations, 1)
    mask = np.ones(orders.size)
    mask[0:extord[0]] = 0.0
    mask[-extord[1]:] = 0.0
    extfit = extrap_cent.copy()
    outparcopy = copy.deepcopy(outpar)
    while i > 0:
        loord = hiord
        hiord = nxord
        nxord = phys_to_pix(extrap_cent[:,-i], locations, 1)

        # Minimum counts between loord and hiord
#        print('calling minbetween')
#        t = time.clock()
#        _minarrL = arcytrace.minbetween(binarr, loord, hiord)
#        _minarrR = arcytrace.minbetween(binarr, hiord, nxord)
#        print('Old minbetween: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        minarrL = new_minbetween(binarr, loord, hiord)
        minarrR = new_minbetween(binarr, hiord, nxord)
#        print('New minbetween: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_minarrL != minarrL) == 0, \
#                'Difference between old and new minbetween, minarrL'
#        assert np.sum(_minarrR != minarrR) == 0, \
#                'Difference between old and new minbetween, minarrR'

        minarr = 0.5*(minarrL+minarrR)
        srchz = np.abs(extfit[:,-i]-extfit[:,-i-1])/3.0
        lopos = phys_to_pix(extfit[:,-i]-srchz, locations, 1)  # The pixel indices for the bottom of the search window
        numsrch = np.int(np.max(np.round(2.0*srchz-extrap_diff[:,-i])))
        diffarr = np.round(extrap_diff[:,-i]).astype(np.int)

#        print('calling find_shift')
#        t = time.clock()
#        _shift = arcytrace.find_shift(binarr, minarr, lopos, diffarr, numsrch)
#        print('Old find_shift: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        shift = new_find_shift(binarr, minarr, lopos, diffarr, numsrch)
#        print('New find_shift: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_shift != shift) == 0, 'Difference between old and new find_shift, shift'

        relshift = np.mean(shift+extrap_diff[:,-i]/2-srchz)
        if shift == -1:
            msgs.info("  Refining order {0:d}: NO relative shift applied".format(int(orders[-i])))
            relshift = 0.0
        else:
            msgs.info("  Refining order {0:d}: relative shift = {1:+f}".format(int(orders[-i]), relshift))
        # Renew guess for the next order
        mask[-i] = 1.0
        extfit, outpar, fail = arpca.refine_iter(outpar, orders, mask, -i, relshift, fitord, function=function)
        if fail:
            msgs.warn("Order refinement has large residuals -- check order traces")
            return extrap_cent, outparcopy
        i -= 1
    # Refine the orders in the negative direction
    i = extord[0]
    loord = phys_to_pix(extrap_cent[:,i+1], locations, 1)
    extrap_cent = extfit.copy()
    outparcopy = copy.deepcopy(outpar)
    while i > 0:
        hiord = loord
        loord = phys_to_pix(extfit[:,i], locations, 1)

#        print('calling minbetween')
#        t = time.clock()
#        _minarr = arcytrace.minbetween(binarr,loord, hiord)
#        print('Old minbetween: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        minarr = new_minbetween(binarr,loord, hiord)
#        print('New minbetween: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_minarr != minarr) == 0, 'Difference between old and new minbetween, minarr'

        srchz = np.abs(extfit[:,i]-extfit[:,i-1])/3.0
        lopos = phys_to_pix(extfit[:,i-1]-srchz, locations, 1)
        numsrch = np.int(np.max(np.round(2.0*srchz-extrap_diff[:,i-1])))
        diffarr = np.round(extrap_diff[:,i-1]).astype(np.int)

#        print('calling find_shift')
#        t = time.clock()
#        _shift = arcytrace.find_shift(binarr, minarr, lopos, diffarr, numsrch)
#        print('Old find_shift: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        shift = new_find_shift(binarr, minarr, lopos, diffarr, numsrch)
#        print('New find_shift: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_shift != shift) == 0, 'Difference between old and new find_shift, shift'

        relshift = np.mean(shift+extrap_diff[:,i-1]/2-srchz)
        if shift == -1:
            msgs.info("  Refining order {0:d}: NO relative shift applied".format(int(orders[i-1])))
            relshift = 0.0
        else:
            msgs.info("  Refining order {0:d}: relative shift = {1:+f}".format(int(orders[i-1]),relshift))
        # Renew guess for the next order
        mask[i-1] = 1.0
        extfit, outpar, fail = arpca.refine_iter(outpar, orders, mask, i-1, relshift, fitord, function=function)
        if fail:
            msgs.warn("Order refinement has large residuals -- check order traces")
            return extrap_cent, outparcopy
        i -= 1
    return extfit, outpar


def trace_tilt(slf, det, msarc, slitnum, censpec=None, maskval=-999999.9,
               trthrsh=1000.0, nsmth=0, method = "fweight"):
    """
    This function performs a PCA analysis on the arc tilts for a single spectrum (or order)
               trthrsh=1000.0, nsmth=0):

    Parameters
    ----------
    slf
    det
    msarc
    slitnum
    censpec
    maskval
    trthrsh
    nsmth
    method : str (fweight or cc)

    Returns
    -------
    trcdict : dict

    """
    def pad_dict(indict):
        """ If an arc line is considered bad, fill the
        dictionary arrays with null values
        """
        indict["xtfit"].append(None)
        indict["ytfit"].append(None)
        indict["wmask"].append(None)
        return indict

    # from pypit import arcyutils
    dnum = settings.get_dnum(det)

    msgs.work("Detecting lines for slit {0:d}".format(slitnum+1))
    ordcen = slf._pixcen[det-1].copy()
    tampl, tcent, twid, w, satsnd, _ = ararc.detect_lines(slf, det, msarc, censpec=censpec)
    satval = settings.spect[dnum]['saturation']*settings.spect[dnum]['nonlinear']
    # Order of the polynomials to be used when fitting the tilts.
    arcdet = (tcent[w]+0.5).astype(np.int)
    ampl = tampl[w]

    # Determine the best lines to use to trace the tilts
    ncont = 15
    aduse = np.zeros(arcdet.size, dtype=np.bool)  # Which lines should be used to trace the tilts
    w = np.where(ampl >= trthrsh)
    aduse[w] = 1
    # Remove lines that are within ncont pixels
    nuse = np.sum(aduse)
    detuse = arcdet[aduse]
    idxuse = np.arange(arcdet.size)[aduse]
    olduse = aduse.copy()
    for s in range(nuse):
        w = np.where((np.abs(arcdet-detuse[s]) <= ncont) & (np.abs(arcdet-detuse[s]) >= 1.0))[0]
        for u in range(w.size):
            if ampl[w[u]] > ampl[olduse][s]:
                aduse[idxuse[s]] = False
                break
    # Restricted to ID lines? [introduced to avoid LRIS ghosts]
    if settings.argflag['trace']['slits']['tilts']['idsonly']:
        ids_pix = np.round(np.array(slf._wvcalib[det-1]['xfit'])*(msarc.shape[0]-1))
        idxuse = np.arange(arcdet.size)[aduse]
        for s in idxuse:
            if np.min(np.abs(arcdet[s]-ids_pix)) > 2:
                msgs.info("Ignoring line at row={:d}".format(arcdet[s]))
                aduse[s] = False

    # Divide the detector into Nseg segments,
    # and find the brightest lines in each segment.
    # The total number of lines used to trace the tilts will be  = Nseg*Nuse + Nadd
    # Nseg = 4
    # Nuse = 2
    # Nadd = 8
    # segsz = msarc.shape[0]/float(Nseg)
    # aduse = np.zeros(arcdet.size, dtype=np.bool)  # Which lines should be used to trace the tilts
    # for s in range(Nseg):
    #     w = np.where((arcdet > s*segsz) & (arcdet <= (s+1)*segsz))[0]
    #     segampl = tampl[w]
    #     asrt = np.argsort(segampl)[::-1]
    #     for u in range(Nuse):
    #         aduse[w[asrt[u]]] = True
    # # Now include some additional bright lines
    # asrt = np.argsort(tampl)[::-1]
    # s, u = 0, 0
    # while u < Nadd:
    #     if not aduse[asrt[s]]:
    #         aduse[asrt[s]] = True
    #         u += 1
    #     s += 1

    # Setup the trace dictionary
    trcdict = {"xtfit":[], "ytfit":[], "wmask":[], "arcdet":arcdet, "aduse":aduse, "badlines":0}

    msgs.info("Modelling arc line tilts with {0:d} arc lines".format(np.sum(aduse)))
    if np.sum(aduse) == 0:
        msgs.warn("No arc lines were deemed usable in slit {0:d} for spectral tilt".format(slitnum))
        return None
    # Go along each order and trace the tilts
    # Start by masking every row, then later unmask the rows with usable arc lines
    msgs.work("This next step could be multiprocessed to speed up the reduction")
    nspecfit = 5
    badlines = 0
    for j in range(arcdet.size):
        # For each detection in this order
        #msgs.info("Tracing tilt of arc line {0:d}/{1:d}".format(j+1, arcdet.size))
        # Check if this is a saturated line
        ysat = msarc[arcdet[j]-nspecfit:arcdet[j]+nspecfit+1, ordcen[arcdet[j], slitnum]-nsmth:ordcen[arcdet[j], slitnum]+nsmth+1]
        if np.where(ysat > satval)[0].size != 0:
            aduse[j] = False
            badlines += 1
            trcdict = pad_dict(trcdict)
            continue
        # Get the size of the slit
        sz = int(np.floor(np.abs(slf._rordloc[det-1][arcdet[j], slitnum]-slf._lordloc[det-1][arcdet[j], slitnum])/2.0)) - 2
        xtfit = np.zeros(2*sz+1)
        ytfit = np.ones(2*sz+1)*maskval  # Fitted centroid
        etfit = np.zeros(2*sz+1)  # Fitted centroid error
        mtfit = np.ones(2*sz+1, dtype=np.int)   # Mask of bad fits
        #apfit = np.zeros(2*sz+1)  # Fitted Amplitude
        xfit = np.arange(-nspecfit, nspecfit+1, 1.0)
        wfit = np.ones(xfit.size, dtype=np.float)
        tstcc = True  # A boolean to tell the loop once a good set of pixels has been found to cross-correlate with
        # Fit up
        pcen = arcdet[j]
        if (pcen < nspecfit) or (pcen > msarc.shape[0]-(nspecfit+1)):
            # Too close to the end of the spectrum
            aduse[j] = False
            badlines += 1
            trcdict = pad_dict(trcdict)
            continue
        offchip = False
        centv = None
        for k in range(0, sz+1-nsmth):
            if (pcen < nspecfit) or (pcen > msarc.shape[0]-(nspecfit+1)):
                offchip = True
                break
            if ordcen[pcen, slitnum]+k >= msarc.shape[1]:
                offchip = True
                break
            # yfit = msarc[pcen-nspecfit:pcen+nspecfit+1,ordcen[arcdet[j],0]+k]
            yfit = msarc[pcen-nspecfit:pcen+nspecfit+1, ordcen[arcdet[j], slitnum]+k-nsmth:ordcen[arcdet[j], slitnum]+k+nsmth+1]
            if np.size(yfit) == 0:
                offchip = True
                break
            if len(yfit.shape) == 2:
                yfit = np.median(yfit, axis=1)
            # wgd = np.where((yfit<satval)&(yfit!=maskval))
            wgd = np.where(yfit == maskval)
            if wgd[0].size != 0:
                continue
            if method == "fweight":
                if centv is None:
                    centv = np.sum(yfit * (pcen+xfit))/np.sum(yfit)
                wfit[0] = 0.5 + (pcen-centv)
                wfit[-1] = 0.5 - (pcen-centv)
                sumxw = yfit * (pcen+xfit) * wfit
                sumw = yfit * wfit
                centv = np.sum(sumxw)/np.sum(sumw)
                fail = False
            elif method == "cc":
                # Get a copy of the array that will be used to cross-correlate
                if tstcc:
                    # ccyfit = msarc[arcdet[j]-nspecfit:arcdet[j]+nspecfit+1,ordcen[arcdet[j],0]+k]
                    ccyfit = msarc[arcdet[j]-nspecfit:arcdet[j]+nspecfit+1,ordcen[arcdet[j],slitnum]+k-nsmth:ordcen[arcdet[j],slitnum]+k+nsmth+1]
                    if len(ccyfit.shape) == 2:
                        ccyfit = np.median(ccyfit, axis=1)
                    wgd = np.where(ccyfit == maskval)
                    if wgd[0].size != 0:
                        continue
                    ccval = arcdet[j] + np.sum(xfit*ccyfit)/np.sum(ccyfit)
                    tstcc = False  # Once we have an array, there's no need to keep looking
                cc = np.correlate(ccyfit, yfit, mode='same')
                params, fail = arutils.gauss_lsqfit(xfit, cc, 0.0)
                centv = ccval + pcen - arcdet[j] - params[1]
            xtfit[k+sz] = ordcen[arcdet[j], slitnum] + k
            ytfit[k+sz] = centv
            etfit[k+sz] = 0.02
            #apfit[k+sz] = params[0]
            if fail:
                mtfit[k+sz] = 1
            else:
                pcen = int(0.5 + centv)
                mtfit[k+sz] = 0
        if offchip:
            # Don't use lines that go off the chip (could lead to a bad trace)
            aduse[j] = False
            badlines += 1
            trcdict = pad_dict(trcdict)
            continue
        for k in range(sz+1-nsmth, sz+1):
            xtfit[k+sz] = ordcen[arcdet[j], slitnum]+k
        # Fit down
        pcen = arcdet[j]
        centv = None
        for k in range(1,sz+1-nsmth):
            if (pcen < nspecfit) or (pcen > msarc.shape[0]-(nspecfit+1)):
                offchip = True
                break
            if ordcen[pcen, slitnum]-k < 0:
                offchip = True
                break
            # yfit = msarc[pcen-nspecfit:pcen+nspecfit+1,ordcen[arcdet[j],0]-k]
            yfit = msarc[pcen - nspecfit:pcen + nspecfit + 1,
                   ordcen[arcdet[j], slitnum] - k - nsmth:ordcen[arcdet[j], slitnum] - k + nsmth + 1]
            if len(yfit.shape) == 2:
                yfit = np.median(yfit, axis=1)
            if np.size(yfit) == 0:
                offchip = True
                break
            wgd = np.where(yfit == maskval)
            if wgd[0].size != 0:
                continue
            if method == "fweight":
                if centv is None:
                    centv = np.sum(yfit * (pcen+xfit))/np.sum(yfit)
                wfit[0] = 0.5 + (pcen-centv)
                wfit[-1] = 0.5 - (pcen-centv)
                sumxw = yfit * (pcen+xfit) * wfit
                sumw = yfit * wfit
                centv = np.sum(sumxw)/np.sum(sumw)
                fail = False
            elif method == "cc":
                # Get a copy of the array that will be used to cross-correlate
                # (testcc is probably already False from the Fit Up part of the loop, but it's best to be sure)
                if tstcc:
                    # ccyfit = msarc[arcdet[j]-nspecfit:arcdet[j]+nspecfit+1,ordcen[arcdet[j],0]-k]
                    ccyfit = msarc[arcdet[j]-nspecfit:arcdet[j]+nspecfit+1,ordcen[arcdet[j],slitnum]-k-nsmth:ordcen[arcdet[j],slitnum]-k+nsmth+1]
                    if len(ccyfit.shape) == 2:
                        ccyfit = np.median(ccyfit, axis=1)
                    wgd = np.where(ccyfit == maskval)
                    if wgd[0].size != 0:
                        continue
                    ccval = arcdet[j] + np.sum(xfit*ccyfit)/np.sum(ccyfit)
                    tstcc = False  # Once we have an array, there's no need to keep looking
                cc = np.correlate(ccyfit, yfit, mode='same')
                params, fail = arutils.gauss_lsqfit(xfit, cc, 0.0)
                centv = ccval + pcen - arcdet[j] - params[1]
            xtfit[sz-k] = ordcen[arcdet[j], slitnum] - k
            ytfit[sz-k] = centv
            etfit[sz-k] = 0.02
            #apfit[sz-k] = params[0]
            if fail:
                mtfit[sz-k] = 1
            else:
                pcen = int(0.5 + centv)
                mtfit[sz-k] = 0
        '''
        jxp_fix = False
        if jxp_fix:
            from desispec.bootcalib import trace_crude_init
            from desispec.bootcalib import trace_fweight as dbtf
            from pypit import ginga
            pcen = arcdet[j]
            img = msarc[pcen-nspecfit:pcen+nspecfit+1, ordcen[arcdet[j], slitnum]-sz:ordcen[arcdet[j], slitnum]+sz+1]
            rot_img = np.rot90(img,3) - 980.  # Bias!
            #
            xcen = np.array([6.6]) # 8.5, 173
            ypass = 173
            # Tune-up first
            for ii in range(4):
                xcen, xsig = dbtf(rot_img, xcen, ycen=np.array([ypass]).astype(int), invvar=None, radius=2.)
            xset, xerr = trace_crude_init(rot_img, xcen, ypass, invvar=None, radius=3., maxshift0=0.5, maxshift=0.15, maxerr=0.2)
            #xcen, xsig = dbtf(rot_img, np.array([6.5,6.5]), ycen=np.array([173,173]).astype(int), invvar=None, radius=2.)
            # Convert back
            y0 = pcen+nspecfit
            ycrude = y0 - xset[:,0]
            #debugger.set_trace()
            #cytfit = ytfit.copy()
            #cytfit[np.where(ytfit < 0)] = np.median(cytfit)
            #debugger.xplot(np.arange(img.shape[1]), ycrude, cytfit)
            #
            #trcdict['save_yt'] = ytfit.copy()
            assert ycrude.size == ytfit.size
            ytfit = ycrude
            #debugger.set_trace()
        '''

        if offchip:
            # Don't use lines that go off the chip (could lead to a bad trace)
            aduse[j] = False
            badlines += 1
            trcdict = pad_dict(trcdict)
            continue
        for k in range(sz+1-nsmth, sz+1):
            xtfit[sz-k] = ordcen[arcdet[j], slitnum]-k

        wmask = np.where(mtfit == 0)
        ytfit[np.where(mtfit == 1)] = maskval

        # Append the trace information into the dictionary
        trcdict["xtfit"].append(xtfit.copy())
        trcdict["ytfit"].append(ytfit.copy())
        trcdict["wmask"].append(wmask[0].copy())
    trcdict["aduse"] = aduse
    trcdict["badlines"] = badlines
    msgs.info("Completed spectral tilt tracing".format(np.sum(aduse)))
    return trcdict


def trace_weighted(frame, ltrace, rtrace, mask=None, wght="flux"):
    """ Estimate the trace of an object in a single slit,
    weighted by the specified method.

    Parameters:
    -----------
    frame : 2D ndarray
      Image for tracing
    ltrace : ndarray
      Left slit edge trace
    rtrace : ndarray
      Right slit edge trace
    mask : ndarray, optional
      Mask of pixels to ignore while tracing
    wght : str
      Method that should be used to weight the tracing:
        wght="flux"  will weight the trace by the flux of each pixel
        wght="uniform"  will use uniform weights

    Returns:
    --------
    trace : ndarray
      array containing the trace of the object along the slit
    error : ndarray
      the associated error of the object trace
    """

    nspec, nspat = frame.shape
    lidx = int(np.ceil(np.min(ltrace)))-1
    ridx = int(np.floor(np.max(rtrace)))+1
    if lidx < 0:
        lidx = 0
    elif lidx >= nspat:
        msgs.info("Slit is off the detector - not tracing")
        return None, None
    if ridx >= nspat:
        ridx = nspat-1
    elif ridx <= 0:
        msgs.info("Slit is off the detector - not tracing")
        return None, None
    extfram = frame[:, lidx:ridx+1]
    if isinstance(wght, (str, ustr)):
        if wght == "flux":
            extwght = extfram.copy()
        elif wght == "uniform":
            extwght = np.ones(extfram.shape, dtype=np.float)
    else:
        # Assume extwght is a numpy array
        extwght = wght
    # Calculate the weights
    idxarr = np.outer(np.ones(nspec), np.arange(lidx, ridx+1))
    ldiff = idxarr - ltrace.reshape(nspec, 1)
    rdiff = idxarr - rtrace.reshape(nspec, 1) - 1.0
    msgs.work("Not sure why -1 is needed here, might be a bug about how ltrace and rtrace are defined")
    lwhr = np.where(ldiff <= -1.0)
    rwhr = np.where(rdiff >= +0.0)
    extwght[lwhr] = 0.0
    extwght[rwhr] = 0.0
    lwhr = np.where((-1.0 < ldiff) & (ldiff <= 0.0))
    rwhr = np.where((-1.0 <= rdiff) & (rdiff < 0.0))
    extwght[lwhr] *= 1.0+ldiff[lwhr]
    extwght[rwhr] *= -rdiff[rwhr]
    # Apply mask
    if mask is not None:
        extwght[np.where(mask[:, lidx:ridx+1] != 0)] *= 0.0
    # Determine the weighted trace
    wghtsum = np.sum(extwght, axis=1)
    wfact = 1.0/(wghtsum + (wghtsum == 0.0))
    trace = np.sum(idxarr*extwght, axis=1) * wfact * (wghtsum != 0)
    error = np.sqrt(np.sum((idxarr-trace.reshape(nspec, 1))**2 * extwght, axis=1) * wfact)
    return trace, error


def trace_fweight(fimage, xinit, ltrace=None, rtraceinvvar=None, radius=3.):
    """ Python port of trace_fweight.pro from IDLUTILS

    Parameters:
    -----------
    fimage: 2D ndarray
      Image for tracing
    xinit: ndarray
      Initial guesses for x-trace
    invvar: ndarray, optional
      Inverse variance array for the image
    radius: float, optional
      Radius for centroiding; default to 3.0
    """

    # Init
    nx = fimage.shape[1]
    ny = fimage.shape[0]
    ncen = len(xinit)
    xnew = copy.deepcopy(xinit)
    xerr = np.zeros(ncen) + 999.

    ycen = np.arange(ny, dtype=int)
    invvar = 0. * fimage + 1.
    x1 = xinit - radius + 0.5
    x2 = xinit + radius + 0.5
    ix1 = np.floor(x1).astype(int)
    ix2 = np.floor(x2).astype(int)

    fullpix = int(np.maximum(np.min(ix2-ix1)-1, 0))
    sumw = np.zeros(ny)
    sumxw = np.zeros(ny)
    sumwt = np.zeros(ny)
    sumsx1 = np.zeros(ny)
    sumsx2 = np.zeros(ny)
    qbad = np.array([False]*ny) 

    if invvar is None: 
        invvar = np.zeros_like(fimage) + 1. 

    # Compute
    for ii in range(0,fullpix+3):
        spot = ix1 - 1 + ii
        ih = np.clip(spot,0,nx-1)
        xdiff = spot - xinit
        #
        wt = np.clip(radius - np.abs(xdiff) + 0.5,0,1) * ((spot >= 0) & (spot < nx))
        sumw = sumw + fimage[ycen,ih] * wt
        sumwt = sumwt + wt
        sumxw = sumxw + fimage[ycen,ih] * xdiff * wt
        var_term = wt**2 / (invvar[ycen,ih] + (invvar[ycen,ih] == 0))
        sumsx2 = sumsx2 + var_term
        sumsx1 = sumsx1 + xdiff**2 * var_term
        #qbad = qbad or (invvar[ycen,ih] <= 0)
        qbad = np.any([qbad, invvar[ycen,ih] <= 0], axis=0)

    # Fill up
    good = (sumw > 0) & (~qbad)
    if np.sum(good) > 0:
        delta_x = sumxw[good]/sumw[good]
        xnew[good] = delta_x + xinit[good]
        xerr[good] = np.sqrt(sumsx1[good] + sumsx2[good]*delta_x**2)/sumw[good]

    bad = np.any([np.abs(xnew-xinit) > radius + 0.5, xinit < radius - 0.5, xinit > nx - 0.5 - radius], axis=0)
    if np.sum(bad) > 0:
        xnew[bad] = xinit[bad]
        xerr[bad] = 999.0

    # Return
    return xnew, xerr


def echelle_tilt(slf, msarc, det, pcadesc="PCA trace of the spectral tilts", maskval=-999999.9):
    """ Determine the spectral tilts in each echelle order

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    msarc : numpy ndarray
      Wavelength calibration frame that will be used to trace constant wavelength
    det : int
      Index of the detector
    pcadesc : str (optional)
      A description of the tilts
    maskval : float (optional)
      Mask value used in numpy arrays

    Returns
    -------
    tilts : ndarray
      Locations of the left slit edges (in physical pixel coordinates)
    satmask : ndarray
      Locations of the right slit edges (in physical pixel coordinates)
    extrapord : ndarray
      A boolean mask indicating if an order was extrapolated (True = extrapolated)
    """
    arccen, maskslit, satmask = get_censpec(slf, msarc, det, gen_satmask=True)
    # If the user sets no tilts, return here
    if settings.argflag['trace']['slits']['tilts']['method'].lower() == "zero":
        # Assuming there is no spectral tilt
        tilts = np.outer(np.linspace(0.0, 1.0, msarc.shape[0]), np.ones(msarc.shape[1]))
        return tilts, satmask, None
    norders = maskslit.size

    # Now model the tilt for each slit
    tiltang, centval = None, None
    slitcnt = 0
    for o in range(norders):
        if maskslit[o] == 1:
            continue
        # Determine the tilts for this slit
        trcdict = trace_tilt(slf, det, msarc, o, censpec=arccen[:, slitcnt])
        slitcnt += 1
        if trcdict is None:
            # No arc lines were available to determine the spectral tilt
            continue
        # ------------------------------------------------------------------------------
        if msgs._debug['tilts']:
            msgs.info("BMJ testing: debugging tilts")
            if o == 0:
                debugger.chk_arc_tilts(msarc, trcdict, sedges=(slf._lordloc[det-1][:,o], slf._rordloc[det-1][:,o]), clearcanvas=True)
            else:
                debugger.chk_arc_tilts(msarc, trcdict, sedges=(slf._lordloc[det-1][:,o], slf._rordloc[det-1][:,o]), clearcanvas=False)
            #debugger.set_trace()
            if o == norders-1:
                debugger.set_trace()
        # ------------------------------------------------------------------------------
        aduse = trcdict["aduse"]
        arcdet = trcdict["arcdet"]
        if tiltang is None:
            tiltang = maskval * np.ones((aduse.size, norders))
            centval = maskval * np.ones((aduse.size, norders))
            totnum = aduse.size
        else:
            if aduse.size > totnum:
                tiltang = np.append(tiltang, maskval * np.ones((aduse.size-totnum, norders)), axis=0)
                centval = np.append(centval, maskval * np.ones((aduse.size-totnum, norders)), axis=0)
                totnum = aduse.size
        for j in range(aduse.size):
            if not aduse[j]:
                continue
            lrdiff = slf._rordloc[det-1][arcdet[j], o] - slf._lordloc[det-1][arcdet[j], o]
            xtfit = 2.0 * (trcdict["xtfit"][j]-slf._lordloc[det-1][arcdet[j], o])/lrdiff - 1.0
            ytfit = trcdict["ytfit"][j]
            wmask = trcdict["wmask"][j]
            if wmask.size < settings.argflag['trace']['slits']['tilts']['order']+2:
                maskslit[o] = 1
                continue
            null, tcoeff = arutils.robust_polyfit(xtfit[wmask], ytfit[wmask],
                                                  settings.argflag['trace']['slits']['tilts']['order'], sigma=2.0)
            # Save the tilt angle
            tiltang[j, o] = -tcoeff[1]  # tan(tilt angle) --- BMJ: added the minus as suggested by rcooke-ast
            centval[j, o] =  tcoeff[0]  # centroid of arc line

    msgs.info("Fitting tilt angles")
    tcoeff = np.ones((settings.argflag['trace']['slits']['tilts']['disporder'] + 1, tiltang.shape[1]))
    maskord = np.where(maskslit == 1)[0]
    extrap_ord = np.zeros(norders)
    for o in range(norders):
        if o in maskord:
            extrap_ord[o] = 1
            continue
        w = np.where(tiltang[:, o] != maskval)
        if np.size(w[0]) <= settings.argflag['trace']['slits']['tilts']['disporder'] + 2:
            extrap_ord[o] = 1.0
            maskord = np.append(maskord, o)
        else:
            null, tempc = arutils.robust_polyfit(centval[:, o][w], tiltang[:, o][w],
                                                 settings.argflag['trace']['slits']['tilts']['disporder'],
                                                 function=settings.argflag['trace']['slits']['function'], sigma=2.0,
                                                 minv=0.0, maxv=msarc.shape[0] - 1)
            tcoeff[:, o] = tempc
    # Sort which orders are masked
    maskord.sort()
    xv = np.arange(msarc.shape[0])
    tiltval = arutils.func_val(tcoeff, xv, settings.argflag['trace']['slits']['function'],
                               minv=0.0, maxv=msarc.shape[0] - 1).T
    ofit = settings.argflag['trace']['slits']['tilts']['params']
    lnpc = len(ofit) - 1
    if np.sum(1.0 - extrap_ord) > ofit[0] + 1:  # Only do a PCA if there are enough good orders
        # Perform a PCA on the tilts
        msgs.info("Performing a PCA on the spectral tilts")
        ordsnd = np.arange(norders) + 1.0
        xcen = xv[:, np.newaxis].repeat(norders, axis=1)
        fitted, outpar = arpca.basis(xcen, tiltval, tcoeff, lnpc, ofit, x0in=ordsnd, mask=maskord, skipx0=False,
                                     function=settings.argflag['trace']['slits']['function'])
        if not msgs._debug['no_qa']:
            #pcadesc = "Spectral Tilt PCA"
#            arqa.pca_plot(slf, outpar, ofit, 'Arc', pcadesc=pcadesc, addOne=False)
            arpca.pca_plot(slf, outpar, ofit, 'Arc', pcadesc=pcadesc, addOne=False)
        # Extrapolate the remaining orders requested
        orders = 1.0 + np.arange(norders)
        extrap_tilt, outpar = arpca.extrapolate(outpar, orders, function=settings.argflag['trace']['slits']['function'])
        tilts = extrap_tilt
#        arqa.pca_arctilt(slf, tiltang, centval, tilts)
        arpca.pca_arctilt(slf, tiltang, centval, tilts)
    else:
        outpar = None
        msgs.warn("Could not perform a PCA when tracing the order tilts" + msgs.newline() +
                  "Not enough well-traced orders")
        msgs.info("Attempting to fit tilts by assuming the tilt is order-independent")
        xtiltfit = np.array([])
        ytiltfit = np.array([])
        for o in range(tiltang.shape[1]):
            w = np.where(tiltang[:, o] != -999999.9)
            if np.size(w[0]) != 0:
                xtiltfit = np.append(xtiltfit, centval[:, o][w])
                ytiltfit = np.append(ytiltfit, tiltang[:, o][w])
        if np.size(xtiltfit) > settings.argflag['trace']['slits']['tilts']['disporder'] + 2:
            tcoeff = arutils.func_fit(xtiltfit, ytiltfit, settings.argflag['trace']['slits']['function'],
                                      settings.argflag['trace']['slits']['tilts']['disporder'],
                                      minv=0.0, maxv=msarc.shape[0] - 1)
            tiltval = arutils.func_val(tcoeff, xv, settings.argflag['trace']['slits']['function'], minv=0.0,
                                       maxv=msarc.shape[0] - 1)
            tilts = tiltval[:, np.newaxis].repeat(tiltang.shape[1], axis=1)
        else:
            msgs.warn("There are still not enough detections to obtain a reliable tilt trace")
            msgs.info("Assuming there is no tilt")
            tilts = np.zeros_like(slf._lordloc)

    # Generate tilts image
#    print('calling tilts_image')
#    t = time.clock()
#    _tiltsimg = arcytrace.tilts_image(tilts, slf._lordloc[det-1], slf._rordloc[det-1],
#                                     settings.argflag['trace']['slits']['pad'], msarc.shape[1])
#    print('Old tilts_image: {0} seconds'.format(time.clock() - t))
#    t = time.clock()
    tiltsimg = new_tilts_image(tilts, slf._lordloc[det-1], slf._rordloc[det-1],
                                settings.argflag['trace']['slits']['pad'], msarc.shape[1])
#    print('New tilts_image: {0} seconds'.format(time.clock() - t))
#    assert np.sum(_tiltsimg != tiltsimg) == 0, 'Difference between old and new tilts_image'

    return tiltsimg, satmask, outpar


def multislit_tilt(slf, msarc, det, maskval=-999999.9):
    """ Determine the spectral tilt of each slit in a multislit image

    Parameters
    ----------
    slf : Class instance
      An instance of the Science Exposure class
    msarc : numpy ndarray
      Wavelength calibration frame that will be used to trace constant wavelength
    det : int
      Index of the detector
    maskval : float (optional)
      Mask value used in numpy arrays

    Returns
    -------
    tilts : ndarray
      Locations of the left slit edges (in physical pixel coordinates)
    satmask : ndarray
      Locations of the right slit edges (in physical pixel coordinates)
    extrapord : ndarray
      A boolean mask indicating if an order was extrapolated (True = extrapolated)
    """
    arccen, maskslit = get_censpec(slf, msarc, det, gen_satmask=False)
    satmask = np.zeros_like(slf._pixcen)
    # If the user sets no tilts, return here
    if settings.argflag['trace']['slits']['tilts']['method'].lower() == "zero":
        # Assuming there is no spectral tilt
        tilts = np.outer(np.linspace(0.0, 1.0, msarc.shape[0]), np.ones(msarc.shape[1]))
        return tilts, satmask, None

    ordcen = slf._pixcen[det - 1].copy()
    fitxy = [settings.argflag['trace']['slits']['tilts']['order'], 1]

    # Now trace the tilt for each slit
    for o in range(arccen.shape[1]):
        # Determine the tilts for this slit
        trcdict = trace_tilt(slf, det, msarc, o, censpec=arccen[:, o], nsmth=3)
        if trcdict is None:
            # No arc lines were available to determine the spectral tilt
            continue
        if msgs._debug['tilts']:
            debugger.chk_arc_tilts(msarc, trcdict, sedges=(slf._lordloc[det-1][:,o], slf._rordloc[det-1][:,o]))
            debugger.set_trace()
        # Extract information from the trace dictionary
        aduse = trcdict["aduse"]
        arcdet = trcdict["arcdet"]
        xtfits = trcdict["xtfit"]
        ytfits = trcdict["ytfit"]
        wmasks = trcdict["wmask"]
        badlines = trcdict["badlines"]
        # Initialize some arrays
        maskrows = np.ones(msarc.shape[0], dtype=np.int)
        tcoeff = np.ones((settings.argflag['trace']['slits']['tilts']['order'] + 1, msarc.shape[0]))
        xtilt = np.ones((msarc.shape[1], arcdet.size)) * maskval
        ytilt = np.ones((msarc.shape[1], arcdet.size)) * maskval
        ztilt = np.ones((msarc.shape[1], arcdet.size)) * maskval
        mtilt = np.ones((msarc.shape[1], arcdet.size)) * maskval
        wtilt = np.ones((msarc.shape[1], arcdet.size)) * maskval
        # Analyze each spectral line
        for j in range(arcdet.size):
            if not aduse[j]:
                continue
            xtfit = xtfits[j]
            ytfit = ytfits[j]
            wmask = wmasks[j]
            xint = int(xtfit[0])
            sz = (xtfit.size-1)//2

            # Perform a scanning polynomial fit to the tilts
            # model = arcyutils.polyfit_scan_intext(xtfit, ytfit, np.ones(ytfit.size, dtype=np.float), mtfit,
            #                                       2, sz/6, 3, maskval)
            wmfit = np.where(ytfit != maskval)
            if wmfit[0].size > settings.argflag['trace']['slits']['tilts']['order'] + 1:
                cmfit = arutils.func_fit(xtfit[wmfit], ytfit[wmfit], settings.argflag['trace']['slits']['function'],
                                         settings.argflag['trace']['slits']['tilts']['order'],
                                         minv=0.0, maxv=msarc.shape[1] - 1.0)
                model = arutils.func_val(cmfit, xtfit, settings.argflag['trace']['slits']['function'],
                                         minv=0.0, maxv=msarc.shape[1] - 1.0)
            else:
                aduse[j] = False
                badlines += 1
                continue

            if maskval in model:
                # Model contains masked values
                aduse[j] = False
                badlines += 1
                continue

            # Perform a robust polynomial fit to the traces
            if settings.argflag['trace']['slits']['tilts']['method'].lower() == "spca":
                yfit = ytfit[wmask] / (msarc.shape[0] - 1.0)
            else:
                yfit = (2.0 * model[sz] - ytfit[wmask]) / (msarc.shape[0] - 1.0)
            wmsk, mcoeff = arutils.robust_polyfit(xtfit[wmask], yfit,
                                                  settings.argflag['trace']['slits']['tilts']['order'],
                                                  function=settings.argflag['trace']['slits']['function'],
                                                  sigma=2.0, minv=0.0, maxv=msarc.shape[1] - 1.0)
            # Update the mask
            wmask = wmask[np.where(wmsk == 0)]

            # Save the tilt angle, and unmask the row
            factr = (msarc.shape[0] - 1.0) * arutils.func_val(mcoeff, ordcen[arcdet[j], 0],
                                                              settings.argflag['trace']['slits']['function'],
                                                              minv=0.0, maxv=msarc.shape[1] - 1.0)
            idx = int(factr + 0.5)
            if (idx > 0) and (idx < msarc.shape[0]):
                maskrows[idx] = 0
                tcoeff[:, idx] = mcoeff.copy()
            # Restrict to good IDs?
            if settings.argflag['trace']['slits']['tilts']['idsonly']:
                if not aduse[j]:
                    maskrows[idx] = 1

            xtilt[xint:xint + 2 * sz + 1, j] = xtfit / (msarc.shape[1] - 1.0)
            ytilt[xint:xint + 2 * sz + 1, j] = arcdet[j] / (msarc.shape[0] - 1.0)
            ztilt[xint:xint + 2 * sz + 1, j] = ytfit / (msarc.shape[0] - 1.0)
            if settings.argflag['trace']['slits']['tilts']['method'].lower() == "spline":
                mtilt[xint:xint + 2 * sz + 1, j] = model / (msarc.shape[0] - 1.0)
            elif settings.argflag['trace']['slits']['tilts']['method'].lower() == "interp":
                mtilt[xint:xint + 2 * sz + 1, j] = (2.0 * model[sz] - model) / (msarc.shape[0] - 1.0)
            else:
                mtilt[xint:xint + 2 * sz + 1, j] = (2.0 * model[sz] - model) / (msarc.shape[0] - 1.0)
            wbad = np.where(ytfit == maskval)[0]
            ztilt[xint + wbad, j] = maskval
            if wmask.size != 0:
                sigg = max(1.4826 * np.median(np.abs(ytfit - model)[wmask]) / np.sqrt(2.0), 1.0)
                wtilt[xint:xint + 2 * sz + 1, j] = 1.0 / sigg
            # Extrapolate off the slit to the edges of the chip
            nfit = 6  # Number of pixels to fit a linear function to at the end of each trace
            xlof, xhif = np.arange(xint, xint + nfit), np.arange(xint + 2 * sz + 1 - nfit, xint + 2 * sz + 1)
            xlo, xhi = np.arange(xint), np.arange(xint + 2 * sz + 1, msarc.shape[1])
            glon = np.mean(xlof * mtilt[xint:xint + nfit, j]) - np.mean(xlof) * np.mean(mtilt[xint:xint + nfit, j])
            glod = np.mean(xlof ** 2) - np.mean(xlof) ** 2
            clo = np.mean(mtilt[xint:xint + nfit, j]) - (glon / glod) * np.mean(xlof)
            yhi = mtilt[xint + 2 * sz + 1 - nfit:xint + 2 * sz + 1, j]
            ghin = np.mean(xhif * yhi) - np.mean(xhif) * np.mean(yhi)
            ghid = np.mean(xhif ** 2) - np.mean(xhif) ** 2
            chi = np.mean(yhi) - (ghin / ghid) * np.mean(xhif)
            mtilt[0:xint, j] = (glon / glod) * xlo + clo
            mtilt[xint + 2 * sz + 1:, j] = (ghin / ghid) * xhi + chi
        if badlines != 0:
            msgs.warn("There were {0:d} additional arc lines that should have been traced".format(badlines) +
                      msgs.newline() + "(perhaps lines were saturated?). Check the spectral tilt solution")

        # Masking
        maskrw = np.where(maskrows == 1)[0]
        maskrw.sort()
        extrap_row = maskrows.copy()
        xv = np.arange(msarc.shape[1])
        # Tilt values
        tiltval = arutils.func_val(tcoeff, xv, settings.argflag['trace']['slits']['function'],
                                   minv=0.0, maxv=msarc.shape[1] - 1.0).T
        msgs.work("May need to do a check here to make sure ofit is reasonable")
        ofit = settings.argflag['trace']['slits']['tilts']['params']
        lnpc = len(ofit) - 1
        # Only do a PCA if there are enough good orders
        if np.sum(1.0 - extrap_row) > ofit[0] + 1:
            # Perform a PCA on the tilts
            msgs.info("Performing a PCA on the tilts")
            ordsnd = np.linspace(0.0, 1.0, msarc.shape[0])
            xcen = xv[:, np.newaxis].repeat(msarc.shape[0], axis=1)
            fitted, outpar = arpca.basis(xcen, tiltval, tcoeff, lnpc, ofit, weights=None,
                                         x0in=ordsnd, mask=maskrw, skipx0=False,
                                         function=settings.argflag['trace']['slits']['function'])
#            arqa.pca_plot(slf, outpar, ofit, 'Arc', pcadesc="Spectral Tilt PCA", addOne=False)
            arpca.pca_plot(slf, outpar, ofit, 'Arc', pcadesc="Spectral Tilt PCA", addOne=False)
            # Extrapolate the remaining orders requested
            orders = np.linspace(0.0, 1.0, msarc.shape[0])
            extrap_tilt, outpar = arpca.extrapolate(outpar, orders, function=settings.argflag['trace']['slits']['function'])
            polytilts = extrap_tilt.T
        else:
            # Fit the model with a 2D polynomial
            msgs.warn("Could not perform a PCA when tracing the spectral tilt" + msgs.newline() +
                      "Not enough well-traced arc lines")
            msgs.info("Fitting tilts with a low order, 2D polynomial")
            wgd = np.where(xtilt != maskval)
            coeff = arutils.polyfit2d_general(xtilt[wgd], ytilt[wgd], mtilt[wgd], fitxy)
            polytilts = arutils.polyval2d_general(coeff, np.linspace(0.0, 1.0, msarc.shape[1]),
                                                  np.linspace(0.0, 1.0, msarc.shape[0]))

        if settings.argflag['trace']['slits']['tilts']['method'].lower() == "interp":
            msgs.info("Interpolating and Extrapolating the tilts")
            xspl = np.linspace(0.0, 1.0, msarc.shape[1])
            # yspl = np.append(0.0, np.append(arcdet[np.where(aduse)]/(msarc.shape[0]-1.0), 1.0))
            # yspl = np.append(0.0, np.append(polytilts[arcdet[np.where(aduse)], msarc.shape[1]/2], 1.0))
            ycen = np.diag(polytilts[arcdet[np.where(aduse)], ordcen[arcdet[np.where(aduse)]]])
            yspl = np.append(0.0, np.append(ycen, 1.0))
            zspl = np.zeros((msarc.shape[1], np.sum(aduse) + 2))
            zspl[:, 1:-1] = mtilt[:, np.where(aduse)[0]]
            # zspl[:, 1:-1] = polytilts[arcdet[np.where(aduse)[0]], :].T
            zspl[:, 0] = zspl[:, 1] + polytilts[0, :] - polytilts[arcdet[np.where(aduse)[0][0]], :]
            zspl[:, -1] = zspl[:, -2] + polytilts[-1, :] - polytilts[arcdet[np.where(aduse)[0][-1]], :]
            # Make sure the endpoints are set to 0.0 and 1.0
            zspl[:, 0] -= zspl[ordcen[0, 0], 0]
            zspl[:, -1] = zspl[:, -1] - zspl[ordcen[-1, 0], -1] + 1.0
            # Prepare the spline variables
            # if False:
            #     pmin = 0
            #     pmax = -1
            # else:
            #     pmin = int(max(0, np.min(slf._lordloc[det-1])))
            #     pmax = int(min(msarc.shape[1], np.max(slf._rordloc[det-1])))
            # xsbs = np.outer(xspl, np.ones(yspl.size))
            # ysbs = np.outer(np.ones(xspl.size), yspl)
            # zsbs = zspl[wgd]
            # Restrict to good portion of the image
            tiltspl = interp.RectBivariateSpline(xspl, yspl, zspl, kx=3, ky=3)
            yval = np.linspace(0.0, 1.0, msarc.shape[0])
            tilts = tiltspl(xspl, yval, grid=True).T
        elif settings.argflag['trace']['slits']['tilts']['method'].lower() == "spline":
            msgs.info("Performing a spline fit to the tilts")
            wgd = np.where((ytilt != maskval) & (ztilt != maskval))
            txsbs = xtilt[wgd]
            tysbs = ytilt[wgd]
            tzsbs = ztilt[wgd]
            twsbs = wtilt[wgd]
            # Append the end points
            wlo = np.where((ytilt == np.min(tysbs)) & (ytilt != maskval) & (ztilt != maskval))
            whi = np.where((ytilt == np.max(tysbs)) & (ytilt != maskval) & (ztilt != maskval))
            xlo = (xtilt[wlo] * (msarc.shape[1] - 1.0)).astype(np.int)
            xhi = (xtilt[whi] * (msarc.shape[1] - 1.0)).astype(np.int)
            xsbs = np.append(xtilt[wlo], np.append(txsbs, xtilt[whi]))
            ysbs = np.append(np.zeros(wlo[0].size), np.append(tysbs, np.ones(whi[0].size)))
            zlo = ztilt[wlo] + polytilts[0, xlo] - polytilts[arcdet[np.where(aduse)[0][0]], xlo]
            zhi = ztilt[whi] + polytilts[-1, xhi] - polytilts[arcdet[np.where(aduse)[0][-1]], xhi]
            zsbs = np.append(zlo, np.append(tzsbs, zhi))
            wsbs = np.append(wtilt[wlo], np.append(twsbs, wtilt[whi]))
            # Generate the spline curve
            tiltspl = interp.SmoothBivariateSpline(xsbs, zsbs, ysbs, w=wsbs, kx=3, ky=3, s=xsbs.size,
                                                   bbox=[0.0, 1.0, min(zsbs.min(), 0.0), max(zsbs.max(), 1.0)])
            xspl = np.linspace(0.0, 1.0, msarc.shape[1])
            yspl = np.linspace(0.0, 1.0, msarc.shape[0])
            tilts = tiltspl(xspl, yspl, grid=True).T
            # QA
            if msgs._debug['tilts']:
                tiltqa = tiltspl(xsbs, zsbs, grid=False)
                plt.clf()
                # plt.imshow((zsbs-tiltqa)/zsbs, origin='lower')
                # plt.imshow((ysbs-tiltqa)/ysbs, origin='lower')
                plt.plot(xsbs, (ysbs - tiltqa) / ysbs, 'bx')
                plt.plot(xsbs, 1.0 / (wsbs * ysbs), 'r-')
                plt.ylim(-5e-2, 5e-2)
                # plt.colorbar()
                plt.show()
                debugger.set_trace()
        elif settings.argflag['trace']['slits']['tilts']['method'].lower() == "spca":
            # Slit position
            xspl = np.linspace(0.0, 1.0, msarc.shape[1])
            # Trace positions down center of the order
            ycen = np.diag(polytilts[arcdet[np.where(aduse)], ordcen[arcdet[np.where(aduse)]]])
            yspl = np.append(0.0, np.append(ycen, 1.0))
            # Trace positions as measured+modeled
            zspl = np.zeros((msarc.shape[1], np.sum(aduse) + 2))
            zspl[:, 1:-1] = polytilts[arcdet[np.where(aduse)[0]], :].T
            zspl[:, 0] = zspl[:, 1] + polytilts[0, :] - polytilts[arcdet[np.where(aduse)[0][0]], :]
            zspl[:, -1] = zspl[:, -2] + polytilts[-1, :] - polytilts[arcdet[np.where(aduse)[0][-1]], :]
            # Make sure the endpoints are set to 0.0 and 1.0
            zspl[:, 0] -= zspl[ordcen[0, 0], 0]
            zspl[:, -1] = zspl[:, -1] - zspl[ordcen[-1, 0], -1] + 1.0
            # Prepare the spline variables
            if False:
                pmin = 0
                pmax = -1
            else:
                pmin = int(max(0, np.min(slf._lordloc[det - 1])))
                pmax = int(min(msarc.shape[1], np.max(slf._rordloc[det - 1])))
            xsbs = np.outer(xspl, np.ones(yspl.size))[pmin:pmax, :]
            ysbs = np.outer(np.ones(xspl.size), yspl)[pmin:pmax, :]
            zsbs = zspl[pmin:pmax, :]
            # Spline
            msgs.work('Consider adding weights to SmoothBivariate in spca')
            tiltspl = interp.SmoothBivariateSpline(xsbs.flatten(),
                                                   zsbs.flatten(),
                                                   ysbs.flatten(), kx=3, ky=3, s=xsbs.size)
            # Finish
            yval = np.linspace(0.0, 1.0, msarc.shape[0])
            tilts = tiltspl(xspl, yval, grid=True).T
            if False:
                tiltqa = tiltspl(xsbs.flatten(), zsbs.flatten(), grid=False).reshape(xsbs.shape)
                plt.clf()
                # plt.imshow((zsbs-tiltqa)/zsbs, origin='lower')
                plt.imshow((ysbs - tiltqa) / ysbs, origin='lower')
                plt.colorbar()
                plt.show()
                debugger.set_trace()
        elif settings.argflag['trace']['slits']['tilts']['method'].lower() == "pca":
            tilts = polytilts.copy()

    # Now do the QA
    msgs.info("Preparing arc tilt QA data")
    tiltsplot = tilts[arcdet, :].T
    tiltsplot *= (msarc.shape[0] - 1.0)
    # Shift the plotted tilts about the centre of the slit
    ztilto = ztilt.copy()
    adj = np.diag(tilts[arcdet, ordcen[arcdet]])
    zmsk = np.where(ztilto == maskval)
    ztilto = 2.0 * np.outer(np.ones(ztilto.shape[0]), adj) - ztilto
    ztilto[zmsk] = maskval
    ztilto[np.where(ztilto != maskval)] *= (msarc.shape[0] - 1.0)
    for i in range(arcdet.size):
        w = np.where(ztilto[:, i] != maskval)
        if w[0].size != 0:
            twa = (xtilt[w[0], i] * (msarc.shape[1] - 1.0) + 0.5).astype(np.int)
            # fitcns = np.polyfit(w[0], ztilt[w[0], i] - tiltsplot[twa, i], 0)[0]
            fitcns = np.median(ztilto[w[0], i] - tiltsplot[twa, i])
            # if abs(fitcns) > 1.0:
            #     msgs.warn("The tilt of Arc Line {0:d} might be poorly traced".format(i+1))
            tiltsplot[:, i] += fitcns

    xdat = xtilt.copy()
    xdat[np.where(xdat != maskval)] *= (msarc.shape[1] - 1.0)

    msgs.info("Plotting arc tilt QA")
#    arqa.plot_orderfits(slf, tiltsplot, ztilto, xdata=xdat, xmodl=np.arange(msarc.shape[1]),
#                        textplt="Arc line", maxp=9, desc="Arc line spectral tilts", maskval=maskval)
    plot_orderfits(slf, tiltsplot, ztilto, xdata=xdat, xmodl=np.arange(msarc.shape[1]),
                   textplt="Arc line", maxp=9, desc="Arc line spectral tilts", maskval=maskval)
    return tilts, satmask, outpar


def get_censpec(slf, frame, det, gen_satmask=False):
    """
    The value of "tilts" returned by this function is of the form:
    tilts = tan(tilt angle), where "tilt angle" is the angle between
    (1) the line representing constant wavelength and
    (2) the column of pixels that is most closely parallel with the spatial direction of the slit.

    The angle is determined relative to the axis defined by ...

    In other words, tilts = y/x according to the docs/get_locations_orderlength.JPG file.

    """
    dnum = settings.get_dnum(det)

    ordcen = 0.5*(slf._lordloc[det-1]+slf._rordloc[det-1])
    ordwid = 0.5*np.abs(slf._lordloc[det-1]-slf._rordloc[det-1])
    if gen_satmask:
        msgs.info("Generating a mask of arc line saturation streaks")
#        t = time.clock()
#        _satmask = arcyarc.saturation_mask(frame,
#                            settings.spect[dnum]['saturation']*settings.spect[dnum]['nonlinear'])
#        print('Old saturation_mask: {0} seconds'.format(time.clock() - t))
        satmask = ararc.new_saturation_mask(frame,
                            settings.spect[dnum]['saturation']*settings.spect[dnum]['nonlinear'])
#        print('New saturation_mask: {0} seconds'.format(time.clock() - t))
#        # Allow for minor differences
#        if np.sum(_satmask != satmask) > 0.1*np.prod(satmask.shape):
#            plt.imshow(_satmask, origin='lower', interpolation='nearest', aspect='auto')
#            plt.colorbar()
#            plt.show()
#            plt.imshow(satmask, origin='lower', interpolation='nearest', aspect='auto')
#            plt.colorbar()
#            plt.show()
#            plt.imshow(_satmask-satmask, origin='lower', interpolation='nearest', aspect='auto')
#            plt.colorbar()
#            plt.show()
#
#        assert np.sum(_satmask != satmask) < 0.1*np.prod(satmask.shape), \
#                    'Old and new saturation_mask are too different'

#        print('calling order saturation')
#        t = time.clock()
#        _satsnd = arcyarc.order_saturation(satmask, (ordcen+0.5).astype(int),
#                                          (ordwid+0.5).astype(int))
#        print('Old order_saturation: {0} seconds'.format(time.clock() - t))
#        t = time.clock()
        satsnd = ararc.new_order_saturation(satmask, (ordcen+0.5).astype(int),
                                             (ordwid+0.5).astype(int))
#        print('New order_saturation: {0} seconds'.format(time.clock() - t))
#        assert np.sum(_satsnd != satsnd) == 0, \
#                    'Difference between old and new order_saturation, satsnd'

    # Extract a rough spectrum of the arc in each slit
    msgs.info("Extracting an approximate arc spectrum at the centre of each slit")
    tordcen = None
    maskslit = np.zeros(ordcen.shape[1], dtype=np.int)
    for i in range(ordcen.shape[1]):
        wl = np.size(np.where(ordcen[:, i] <= slf._pixlocn[det-1][0,0,1])[0])
        wh = np.size(np.where(ordcen[:, i] >= slf._pixlocn[det-1][0,-1,1])[0])
        if wl == 0 and wh == 0:
            # The center of the slit is always on the chip
            if tordcen is None:
                tordcen = np.zeros((ordcen.shape[0], 1), dtype=np.float)
                tordcen[:, 0] = ordcen[:, i]
            else:
                tordcen = np.append(tordcen, ordcen[:, i].reshape((ordcen.shape[0], 1)), axis=1)
        else:
            # A slit isn't always on the chip
            if tordcen is None:
                tordcen = np.zeros((ordcen.shape[0], 1), dtype=np.float)
            else:
                tordcen = np.append(tordcen, ordcen[:, i].reshape((ordcen.shape[0], 1)), axis=1)
            maskslit[i] = 1
    w = np.where(maskslit == 0)[0]
    if tordcen is None:
        msgs.warn("Could not determine which slits are fully on the detector")
        msgs.info("Assuming all slits are fully on the detector")
        ordcen = phys_to_pix(ordcen, slf._pixlocn[det-1], 1)
    else:
        ordcen = phys_to_pix(tordcen[:,w], slf._pixlocn[det-1], 1)

    pixcen = np.arange(frame.shape[0])
    temparr = pixcen.reshape(frame.shape[0], 1).repeat(ordcen.shape[1], axis=1)
    # Average over three pixels to remove some random fluctuations, and increase S/N
    op1 = ordcen+1
    op2 = ordcen+2
    om1 = ordcen-1
    om2 = ordcen-2
    w = np.where(om1 < 0)
    om1[w] += 1
    w = np.where(om2 == -1)
    om2[w] += 1
    w = np.where(om2 == -2)
    om2[w] += 2
    w = np.where(op1 >= frame.shape[1])
    op1[w] -= 1
    w = np.where(op2 == frame.shape[1])
    op2[w] -= 1
    w = np.where(op2 == frame.shape[1]+1)
    op2[w] -= 2
    arccen = (1.0/5.0) * (frame[temparr, ordcen] +
              frame[temparr, op1] + frame[temparr, op2] +
              frame[temparr, om1] + frame[temparr, om2])
    del temparr

    if gen_satmask:
        return arccen, maskslit, satsnd
    else:
        return arccen, maskslit


def gen_pixloc(frame, det, gen=True):
    """
    Generate an array of physical pixel coordinates

    Parameters
    ----------
    frame : ndarray
      uniformly illuminated and normalized flat field frame
    det : int
      Index of the detector

    Returns
    -------
    locations : ndarray
      A 3D array containing the x center, y center, x width and y width of each pixel.
      The returned array has a shape:   frame.shape + (4,)
    """
    dnum = settings.get_dnum(det)
    msgs.info("Deriving physical pixel locations on the detector")
    locations = np.zeros((frame.shape[0],frame.shape[1],4))
    if gen:
        msgs.info("Pixel gap in the dispersion direction = {0:4.3f}".format(settings.spect[dnum]['xgap']))
        msgs.info("Pixel size in the dispersion direction = {0:4.3f}".format(1.0))
        xs = np.arange(frame.shape[0]*1.0)*settings.spect[dnum]['xgap']
        xt = 0.5 + np.arange(frame.shape[0]*1.0) + xs
        msgs.info("Pixel gap in the spatial direction = {0:4.3f}".format(settings.spect[dnum]['ygap']))
        msgs.info("Pixel size in the spatial direction = {0:4.3f}".format(settings.spect[dnum]['ysize']))
        ys = np.arange(frame.shape[1])*settings.spect[dnum]['ygap']*settings.spect[dnum]['ysize']
        yt = settings.spect[dnum]['ysize']*(0.5 + np.arange(frame.shape[1]*1.0)) + ys
        xloc, yloc = np.meshgrid(xt, yt)
#		xwid, ywid = np.meshgrid(xs,ys)
        msgs.info("Saving pixel locations")
        locations[:,:,0] = xloc.T
        locations[:,:,1] = yloc.T
        locations[:,:,2] = 1.0
        locations[:,:,3] = settings.spect[dnum]['ysize']
    else:
        msgs.error("Have not yet included an algorithm to automatically generate pixel locations")
    return locations


def phys_to_pix(array, pixlocn, axis):
    """
    Generate an array of physical pixel coordinates

    Parameters
    ----------
    array : ndarray
      An array of physical pixel locations
    pixlocn : ndarray
      A 3D array containing the x center, y center, x width and y width of each pixel.
    axis : int
      The axis that the input array probes

    Returns
    -------
    pixarr : ndarray
      The pixel locations of the input array (as seen on a computer screen)
    """
    diff = pixlocn[:,0,0] if axis == 0 else pixlocn[0,:,1]

#    print('calling phys_to_pix')
#    t = time.clock()
#    _pixarr = arcytrace.phys_to_pix(np.array([array]).T, diff).flatten() \
#                if len(np.shape(array)) == 1 else arcytrace.phys_to_pix(array, diff)
#    print('Old phys_to_pix: {0} seconds'.format(time.clock() - t))
#    t = time.clock()
    pixarr = new_phys_to_pix(array, diff)
#    print('New phys_to_pix: {0} seconds'.format(time.clock() - t))
#    assert np.sum(_pixarr != pixarr) == 0, 'Difference between old and new phys_to_pix, pixarr'

    return pixarr


def prune_peaks(hist, pks, pkidx):
    """
    Identify the most well defined peaks

    Parameters
    ----------
    hist : ndarray
      Histogram of detections
    pks : ndarray
      Indices of candidate peak locations
    pkidx : int
      Index of highest peak

    Returns
    -------
    msk : ndarray
      An mask of good peaks (1) and bad peaks (0)
    """

    sz_i = pks.shape[0]

    msk = np.zeros(sz_i, dtype=np.int)

    lgd = 1  # Was the previously inspected peak a good one?
    for ii in range(0, sz_i-1):
        cnt = 0
        for jj in range(pks[ii], pks[ii+1]):
            if hist[jj] == 0:
                cnt += 1
        if cnt < 2:
            # If the difference is unacceptable, both peaks are bad
            msk[ii] = 0
            msk[ii+1] = 0
            lgd = 0
        else:
            # If the difference is acceptable, the right peak is acceptable,
            # the left peak is acceptable if it was not previously labelled as unacceptable
            if lgd == 1:
                msk[ii] = 1
            msk[ii+1] = 1
            lgd = 1
    # Now only consider the peaks closest to the highest peak
    lgd = 1
    for ii in range(pkidx, sz_i):
        if msk[ii] == 0:
            lgd = 0
        elif lgd == 0:
            msk[ii] = 0
    lgd = 1
    for ii in range(0, pkidx):
        if msk[pkidx-ii] == 0:
            lgd = 0
        elif lgd == 0:
            msk[pkidx-ii] = 0
    return msk


def slit_image(slf, det, scitrace, obj, tilts=None):
    """ Generate slit image for a given object
    Ignores changing plate scale (for now)
    The slit is approximated as a straight line in this calculation
    which should be reasonably accurate.  Better, the object profile
    generated from this approximation is applied in the same fashion
    so that the 'error' is compensated for.
    Parameters
    ----------
    slf
    det
    scitrace
    obj
    Returns
    -------
    slit_img : ndarray
    """
    # Setup
    if tilts is None:
        tilts = slf._tilts[det-1]
    ximg = np.outer(np.ones(tilts.shape[0]), np.arange(tilts.shape[1]))
    dypix = 1./tilts.shape[0]
    #  Trace
    xtrc = np.round(scitrace['traces'][:, obj]).astype(int)
    wch = np.where((xtrc >= 0) & (xtrc <= tilts.shape[1]-1))
    msgs.work("Use 2D spline to evaluate tilts")
    trc_tilt = np.zeros(tilts.shape[0], dtype=np.float)
    trc_tilt[wch] = tilts[np.arange(tilts.shape[0])[wch], xtrc[wch]]
    trc_tilt_img = np.outer(trc_tilt, np.ones(tilts.shape[1]))
    # Slit image
    msgs.work("Should worry about changing plate scale")
    dy = (tilts - trc_tilt_img)/dypix  # Pixels
    dx = ximg - np.outer(scitrace['traces'][:, obj], np.ones(tilts.shape[1]))
    slit_img = np.sqrt(dx**2 - dy**2)
    neg = dx < 0.
    slit_img[neg] *= -1
    # Return
    return slit_img


def find_obj_minima(trcprof, fwhm=3., nsmooth=3, nfind=8, xedge=0.03,
        sig_thresh=5., peakthresh=None, triml=2, trimr=2, debug=False):
    ''' Find objects using a ported version of nminima from IDL (idlutils)
    
    Parameters
    ----------
    trcprof : ndarray
      1D array of the slit profile including objects
    fwhm : float, optional
      FWHM estimate of the seeing
    nsmooth : int, optional
      Kernel in pixels for Gaussian smoothing of trcprof
    nfind : int, optional
      Number of sources to identify
    xedge : float, optional
      Fraction of the slit on each edge to ignore peaks within
    sig_thresh : float, optional
      Number of sigma to reject when fitting
    peakthresh : float, optional
      Include objects whose peak exceeds this fraction of the highest peak
    triml : int, optional
    trimr : int, optional
    debug : bool, optional

    Returns
    -------
    objl : ndarray  (npeak)
      Left edges for each peak found
    objr : ndarray  (npeak)
      Right edges for each peak found
    bckl : ndarray (npeak, npix)
      Background regions in the slit, left of the peak
    bckr : ndarray (npeak, npix)
      Background regions in the slit, right of the peak
    '''
    npix = trcprof.size
    # Smooth
    if nsmooth > 0:
        yflux = convolve(-1*trcprof, Gaussian1DKernel(nsmooth))
    else:
        yflux = -1*trcprof
    #
    xvec = np.arange(len(yflux))
    # Find peaks
    peaks, sigmas, ledges, redges = arutils.find_nminima(yflux, xvec, minsep=fwhm, nfind=nfind, width=int(fwhm))
    fint = interp.interp1d(xvec, yflux, bounds_error=False, fill_value=0.)
    ypeaks = -1.*fint(peaks)
    # Sky background (for significance)
    imask = xvec == xvec
    for ipeak in peaks:  # Mask out for sky determination
        ibad = np.abs(xvec-ipeak) < fwhm
        imask[ibad] = False
    clip_yflux = sigma_clip(yflux[imask], axis=0, sigma=2.5)
    sky_std = np.std(clip_yflux)
    # Toss out peaks near edges
    xp = peaks/float(npix)/2.
    gdobj = (xp < (1-xedge)) * (xp > xedge)
    # Keep those above the threshold
    if np.sum(gdobj) > 1:
        threshold = sig_thresh*sky_std
        if peakthresh is not None:
            threshold = max(threshold, peakthresh*max(ypeaks[gdobj]))
        # Parse again
        gdthresh = (ypeaks > threshold)
        if np.any(gdthresh):  # Parse if at least one is bright enough
            gdobj = gdobj & gdthresh
    # Setup for finish
    nobj = np.sum(gdobj)
    objl = ledges[gdobj]  # Might need to set these by threshold
    objr = redges[gdobj]
    bck_mask = np.ones_like(trcprof).astype(int)
    # Mask out objects
    for ledge, redge in zip(objl,objr):
        bck_mask[ledge:redge+1] = 0
    # Mask out edges
    bck_mask[0:triml] = 0
    bck_mask[-trimr:] = 0
    bckl = np.outer(bck_mask, np.ones(nobj))
    bckr = bckl.copy()
    idx = np.arange(npix).astype(int)
    for kk,iobjr in enumerate(objr):
        pos = idx > iobjr
        bckl[pos,kk] = 0
        neg = idx < objl[kk]
        bckr[neg,kk] = 0
    # Return
    return objl, objr, bckl, bckr
