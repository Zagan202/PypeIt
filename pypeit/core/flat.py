""" Core module for methods related to flat fielding
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import inspect

import numpy as np
import os

from scipy import interpolate

from matplotlib import pyplot as plt


from pypeit import msgs

from pypeit import utils
#from pypeit import arparse as settings
from pypeit.core import parse
from pypeit.core import qa
from pypeit.core import pca
from pypeit.core import pixels
from pypeit.core import tracewave

from pypeit import debugger
from pypeit import utils
from pypeit.core import pydl
from matplotlib import pyplot as plt
import scipy

def tweak_slit_edges(slit_left_in, slit_righ_in, ximg_fit, normimg, tweak_slits_thresh, tweak_slits_maxfrac):

    slitwidth = np.median(slit_righ_in - slit_left_in) # How many pixels wide is the slit at each Y?
    # Determine the maximum at the left and right end of the slit
    ileft = (ximg_fit > 0.1) & (ximg_fit < 0.4)
    xleft = ximg_fit[ileft]
    norm_max_left = normimg[ileft].max()
    xmax_left = xleft[normimg[ileft].argmax()]
    irigh = (ximg_fit > 0.6) & (ximg_fit < 0.9)
    xrigh = ximg_fit[irigh]
    norm_max_righ = normimg[irigh].max()
    xmax_righ = xrigh[normimg[irigh].argmax()]

    tweak_left = False
    tweak_righ = False
    slit_left_out = np.copy(slit_left_in)
    slit_righ_out = np.copy(slit_righ_in)
    msgs.info('Tweaking slit boundaries using slit illumination function')
    step = 0.001
    # march out from middle to find left edge
    msgs.info('Left threshold = {:5.3f}'.format(tweak_slits_thresh * norm_max_left) +
              ' --  or {:5.3f}'.format(
                  100.0 * tweak_slits_thresh) + ' % of left side max of illumination function = {:5.3f}'.format(norm_max_left))
    for xleft in np.arange(0.5, ximg_fit.min(), -step):
        norm_now = np.interp(xleft, ximg_fit, normimg)
        if (norm_now < tweak_slits_thresh * norm_max_left) & (xleft < tweak_slits_maxfrac):
            slit_left_out += xleft * slitwidth
            tweak_left = True
            msgs.info('Tweaking left slit boundary by {:5.3f}'.format(100 * xleft) +
                      ' %, or {:7.3f}'.format(xleft * slitwidth) + ' pixels')
            if np.abs(xleft - tweak_slits_maxfrac) < 0.01:
                msgs.warn(
                    'Left slit boundary tweak limited by maximum changed allowed by tweak_slits_maxfracn={:5.3f}'.format(
                        100.0 * tweak_slits_maxfrac) + ' %')
            break
    msgs.info('Right threshold = {:5.3f}'.format(tweak_slits_thresh * norm_max_righ) +
              ' --  or {:5.3f}'.format(
                  100.0 * tweak_slits_thresh) + ' % of right side max of illumination function = {:5.3f}'.format(norm_max_righ))
    # march out from middle  to find right edge
    for xrigh in np.arange(0.5, ximg_fit.max(), step):
        norm_now = np.interp(xrigh, ximg_fit, normimg)
        if (norm_now < tweak_slits_thresh * norm_max_righ) & ((1.0 - xrigh) < tweak_slits_maxfrac):
            slit_righ_out -= (1.0 - xrigh) * slitwidth
            tweak_righ = True
            msgs.info('Tweaking right slit boundary by {:5.3f}'.format(100 * (1.0 - xrigh)) +
                      ' %, or {:7.3f}'.format((1.0 - xrigh) * slitwidth) + ' pixels')
            if np.abs((1.0 - xrigh) - tweak_slits_maxfrac) < 0.01:
                msgs.warn(
                    'Right slit boundary tweak limited by maximum changed allowed by tweak_slits_maxfracn={:5.3f}'.format(
                        100.0 * tweak_slits_maxfrac) + ' %')
            break

    tweak_dict = {'xleft': xleft, 'xrigh': xrigh,
                  'norm_max_left': norm_max_left, 'norm_max_righ': norm_max_righ,
                  'tweak_left': tweak_left, 'tweak_righ': tweak_righ}

    return slit_left_out, slit_righ_out, tweak_dict



def fit_flat(flat, tilts_dict, thismask_in, slit_left_in, slit_righ_in, inmask = None,spec_samp_fine = 1.2, spec_samp_coarse = 50.0,
             spat_samp = 5.0, spat_illum_thresh = 0.01, npoly = None, trim_edg = (3.0,3.0),
             tweak_slits = True, tweak_slits_thresh = 0.93, tweak_slits_maxfrac = 0.07, nonlinear_counts =1e10, debug = False):


    """ Compute pixelflat and illumination flat from a flat field image.

    Parameters
    ----------
    flat :  float ndarray, shape (nspec, nspat)
        Flat field image in units of electrons.


    tilts_dict: dict
          Dictionary containing wavelength tilts image and other information indicating how wavelengths move across the slit

    thismask_in:  boolean ndarray, shape (nspec, nspat)
        Boolean mask image specifying the pixels which lie on the slit/order according to the initial slit/order bounadries.
        The convention is: True = on the slit/order, False  = off the slit/order

    slit_left_in:  float ndarray, shape  (nspec, 1) or (nspec)
        Left boundary of slit/order to be extracted (given as floating pt pixels).

    slit_righ_in:  float ndarray, shape  (nspec, 1) or (nspec)
        Right boundary of slit/order to be extracted (given as floating pt pixels).


    Optional Parameters
    -------------------
    inmask: boolean ndarray, shape (nspec, nspat), default inmask = None
      Input mask for pixels not to be included in sky subtraction fits. True = Good (not masked), False = Bad (masked)

    spec_samp_fine: float, default = 1.2
      bspline break point spacing in units of pixels for spectral fit to flat field blaze function.

    spec_samp_coarse: float, default = 50.0
      bspline break point spacing in units of pixels for 2-d bspline-polynomial fit to flat field image residuals.
      This should be a large number unless you are trying to fit a sky flat with lots of features.

    spat_samp: float, default = 5.0
      Spatial sampling for spatial slit illumination function. This is the width of the median filter in pixels used to
      determine the slit illumination function, and thus sets the minimum scale on which the illumination function will
      have features.

    spat_illum_thresh: float, default = 0.01
      Spatial illumination function threshold. If the slits have

    trim_edg: tuple of floats  (left_edge, right_edge), default (3,3)
      indicates how many pixels to trim from left and right slit edges for creating the edgemask, which is used to mask
      the edges from the initial (fine) spectroscopic fit to the blaze function.

    npoly: int, default = None
      Order of polynomial for 2-d bspline-polynomial fit to flat field image residuals. The code determines the order of
      these polynomials to each slit automatically depending on the slit width, which is why the default is None.
      Do not attempt to set this paramter unless you know what you are doing.


    tweak_slits: bool, default = True
      Slit edges will be tweaked such the left and right bounadaries intersect the location where the illumination
      function falls below tweak_slits_thresh (see below) of its maximum value near the center (moving out from the center)

    tweak_slits_thresh: float, default = 0.85
      If tweak_slits is True, this sets the illumination function threshold used to tweak the slits

    tweak_slits_maxfrac: float, default = 0.05
      Maximum fractinoal amount (of slit width) allowed for each trimming the left and right slit boundaries, i.e. the
      default is 5% which means slits would shrink by at most 10%.

    debug: bool, default = False
      Show plots useful for debugging. This will block further execution of the code until the plot windows are closed.


   TODO update for new behavior
    Returns
    -------
    pixeflat:   ndarray with size = np.sum(thismask)
      Pixelflat gives pixel-to-pixel variations of detector response. Values are centered about unity and
      are returned at the locations where thismask == True

    illumflat:   ndarray with size = np.sum(thismask)
      Illumination flat gives variations of the slit illumination function across the spatial direction of the detect.
      Values are centered about unity and are returned at the locations where thismask == True. The slit illumination
      function is computed by dividing out the spectral response and collapsing out the spectral direction.

    flat_model:  ndarray with size = np.sum(thismask)
      Full 2-d model image of the input flat image in units of electrons.  The pixelflat is defined to be flat/flat_model.


    Revision History
    ----------------
    11-Mar-2005  First version written by Scott Burles.
    2005-2018    Improved by J. F. Hennawi and J. X. Prochaska
    3-Sep-2018 Ported to python by J. F. Hennawi and significantly improved
    """

    shape = flat.shape
    if shape != tilts_dict['tilts'].shape:
        msgs.error('Something is very wrong. Tilt image shape does not match flat field image shape')

    nspec = shape[0]
    nspat = shape[1]

    # Compute some things using the original slit boundaries and thismask_in

    # Approximate number of pixels sampling each spatial pixel for this (original) slit.
    npercol = np.fmax(np.floor(np.sum(thismask_in)/nspec),1.0)
    # Demand at least 10 pixels per row (on average) per degree of the polynomial
    if npoly is None:
        npoly_in = 7
        npoly = np.fmax(np.fmin(npoly_in, (np.ceil(npercol/10.)).astype(int)),1)

    # Generate the edgemask using the original slit boundaries and thismask_in
    ximg_in, edgmask_in = pixels.ximg_and_edgemask(slit_left_in, slit_righ_in, thismask_in, trim_edg=trim_edg)

    # Create a tilts image that encompasses the whole image, rather than just the thismask_in slit pixels
    tilts = tracewave.coeff2tilts(tilts_dict['coeffs'], shape, tilts_dict['func2D'], max_tilt=1.2, min_tilt=-0.2)
    piximg = tilts * (nspec-1)
    pixvec = np.arange(nspec)

    # Create a fractional position image ximg that encompasses the whole image, rather than just the thismask_in slit pixels
    spat_img = np.outer(np.ones(nspec), np.arange(nspat)) # spatial position everywhere along image
    slit_left_img = np.outer(slit_left_in, np.ones(nspat))   # left slit boundary replicated spatially
    slitwidth_img = np.outer(slit_righ_in - slit_left_in, np.ones(nspat)) # slit width replicated spatially
    ximg = (spat_img - slit_left_img)/slitwidth_img

    # Create a wider slitmask image with shift pixels padded on each side
    pad = 5.0
    slitmask_pad = pixels.slit_pixels(slit_left_in, slit_righ_in, shape, pad)
    thismask = (slitmask_pad > 0) # mask enclosing the wider slit bounadries

    if inmask is None:
        inmask = np.copy(thismask)

    # Fit the spectral direction of the blaze. We do this in the log
    log_flat = np.log(np.fmax(flat, 1.0))
    inmask_log = ((flat > 1.0) & inmask)
    log_ivar = inmask_log.astype(float)/0.5**2 # set errors to just be 0.5 in the log

    # Flat field pixels for fitting spectral direction. Restrict to original slit pixels
    fit_spec = thismask_in & inmask & np.invert(edgmask_in) & (flat < nonlinear_counts)
    isrt_spec = np.argsort(piximg[fit_spec])
    pix_fit = piximg[fit_spec][isrt_spec]
    log_flat_fit = log_flat[fit_spec][isrt_spec]
    log_ivar_fit = log_ivar[fit_spec][isrt_spec]
    inmask_log_fit = inmask_log[fit_spec][isrt_spec]
    nfit_spec = np.sum(fit_spec)
    logrej = 0.5 # rejectino threshold for spectral fit in log(image)
    msgs.info('Spectral fit of flatfield for {:}'.format(nfit_spec) + ' pixels')

    # ToDo Figure out how to deal with the fits going crazy at the edges of the chip in spec direction
    spec_set_fine, outmask_spec, specfit, _ = utils.bspline_profile(pix_fit, log_flat_fit, log_ivar_fit,
                                                                    np.ones_like(pix_fit), inmask = inmask_log_fit,
                                                                    nord = 4, upper=logrej, lower=logrej,
                                                                    kwargs_bspline = {'bkspace':spec_samp_fine},
                                                                    kwargs_reject={'groupbadpix':True, 'maxrej': 5})

    # Debugging/checking spectral fit
    if debug:
        goodbk = spec_set_fine.mask
        specfit_bkpt, _ = spec_set_fine.value(spec_set_fine.breakpoints[goodbk])
        was_fit_and_masked = (outmask_spec == False)
        plt.clf()
        ax = plt.gca()
        ax.plot(pix_fit,log_flat_fit, color='k', marker='o', markersize=0.4, mfc='k', fillstyle='full',
                linestyle='None', label = 'all pixels')
        ax.plot(pix_fit[was_fit_and_masked],log_flat_fit[was_fit_and_masked], color='red', marker='+',
                markersize=1.5, mfc='red', fillstyle='full', linestyle='None', label='masked')
        ax.plot(pix_fit, specfit, color='cornflowerblue', label = 'fit to blaze')
        ax.plot(spec_set_fine.breakpoints[goodbk], specfit_bkpt, color='lawngreen', marker='o', markersize=2.0,
                mfc='lawngreen', fillstyle='full', linestyle='None', label='bspline breakpoints')
        ax.set_ylim((0.99*specfit.min(),1.01*specfit.max()))
        plt.legend()
        plt.xlabel('Spectral Pixel')
        plt.ylabel('log(flat counts)')
        plt.show()

    # Evaluate and save
    spec_model = np.ones_like(flat)
    spec_model[thismask], _ = np.exp(spec_set_fine.value(piximg[thismask]))
    norm_spec = np.ones_like(flat)
    norm_spec[thismask] = flat[thismask]/np.fmax(spec_model[thismask], 1.0)

    # Flat field pixels for fitting spatial direction
    # Determine maximum counts in median filtered flat spectrum. Only fit pixels > 0.1 of this maximum
    specvec = np.exp(np.interp(pixvec, pix_fit, specfit))
    spec_sm = utils.fast_running_median(specvec, np.fmax(np.ceil(0.10*nspec).astype(int),10))
    spec_sm_max = np.fmin(spec_sm.max(),nonlinear_counts)
    fit_spat = thismask & inmask & (flat < nonlinear_counts) & (spec_model > 1.0) & (spec_model > 0.1*spec_sm_max) & \
               (norm_spec > 0.0) & (norm_spec < 1.7)
    isrt_spat = np.argsort(ximg[fit_spat])
    ximg_fit = ximg[fit_spat][isrt_spat]
    norm_spec_fit = norm_spec[fit_spat][isrt_spat]
    norm_spec_ivar = np.ones_like(norm_spec_fit)/(spat_illum_thresh**2)
    nfit_spat = np.sum(fit_spat)

    slitwidth = np.median(slit_righ_in - slit_left_in) # How many pixels wide is the slit at each Y?
    ximg_resln = spat_samp/slitwidth

    med_width = (np.ceil(nfit_spat*ximg_resln)).astype(int)
    normimg_raw = utils.fast_running_median(norm_spec_fit,med_width)
    #normimg_raw = scipy.ndimage.filters.median_filter(norm_spec_fit[imed], size=med_width, mode='reflect')
    sig_res = np.fmax(med_width/20.0,0.5)
    normimg = scipy.ndimage.filters.gaussian_filter1d(normimg_raw,sig_res, mode='nearest')

    # mask regions where illumination function takes on extreme values
    if np.any(~np.isfinite(normimg)):
        msgs.error('Inifinities in slit illumination function computation normimg')

    # Determine the breakpoint spacing from the sampling of the ximg
    ximg_samp = np.median(ximg_fit - np.roll(ximg_fit,1))
    ximg_1pix = 1.0/slitwidth
    # Use breakpoints at a spacing of a 1/10th of a pixel, but do not allow a bsp smaller than the typical sampling
    ximg_bsp  = np.fmax(ximg_1pix/10.0, ximg_samp*1.2)
    bsp_set = pydl.bspline(ximg_fit,nord=4, bkspace=ximg_bsp)
    fullbkpt = bsp_set.breakpoints
    spat_set, outmask_spat, spatfit, _ = utils.bspline_profile(ximg_fit, normimg, np.ones_like(normimg),np.ones_like(normimg),
                                                               nord=4,upper=5.0, lower=5.0,fullbkpt = fullbkpt)

    # Evaluate and save
    illumflat = np.ones_like(flat)
    illumflat[thismask], _ = spat_set.value(ximg[thismask])
    norm_spec_spat = np.ones_like(flat)
    norm_spec_spat[thismask] = flat[thismask]/np.fmax(spec_model[thismask], 1.0)/np.fmax(illumflat[thismask],0.01)

    if tweak_slits:
        slit_left_out, slit_righ_out, tweak_dict = tweak_slit_edges(slit_left_in, slit_righ_in, ximg_fit, normimg,
                                                                    tweak_slits_thresh, tweak_slits_maxfrac)
        # Recreate all the quantities we need based on the tweaked slits
        slitmask_out = pixels.slit_pixels(slit_left_out, slit_righ_out, shape, 0)
        thismask_out = (slitmask_out > 0)
        ximg_out, edgmask_out = pixels.ximg_and_edgemask(slit_left_out, slit_righ_out, thismask_out, trim_edg=trim_edg)
        # Note that nothing changes with the tilts, since these were already extrapolated across the whole image.
    else:
        slit_left_out = np.copy(slit_left)
        slit_righ_out = np.copy(slit_righ)
        thismask_out = thismask_in
        ximg_out = ximg_in

    # Add an approximate pixel axis at the top
    if debug:
        plt.clf()
        ax = plt.gca()
        ax.plot(ximg_fit, norm_spec_fit, color='k', marker='o', markersize=0.4, mfc='k', fillstyle='full',linestyle='None',
                label = 'all pixels')
        #ax.plot(ximg_fit[~imed], norm_spec_fit[~imed], color='darkred', marker='+',markersize=4.0, mfc='red',
        #        fillstyle='full', linestyle='None', label = 'masked')
        #ax.plot(ximg_fit[imed], normfit[imed], color='orange', label = 'median spatial profile')
        ax.plot(ximg_fit, spatfit, color='cornflowerblue', label = 'final slit illumination function')
        ymin = np.fmax(0.8 * spatfit.min(), 0.5)
        ymax = 1.2*spatfit.max()
        ax.set_ylim((np.fmax(0.8 * spatfit.min(), 0.5), 1.2 * spatfit.max()))
        ax.set_xlim(ximg_fit.min(), ximg_fit.max())
        plt.vlines(0.0, ymin, ymax, color='lightgreen', linestyle=':', linewidth=2.0, label='original left edge',zorder=8)
        plt.vlines(1.0,ymin,ymax, color='red',linestyle=':', linewidth = 2.0, label='original right edge',zorder=9)
        if tweak_slits:
            if tweak_dict['tweak_left']:
                label = 'threshold = {:5.2f}'.format(tweak_slits_thresh) + ' % of max of left illumprofile'

                plt.hlines(tweak_slits_thresh*tweak_dict['norm_max_left'], ximg_fit.min(), 0.5, color='lightgreen',
                           linewidth=3.0,label=label, zorder=10)
                plt.vlines(tweak_dict['xleft'],ymin,ymax, color='lightgreen',linestyle='--', linewidth = 3.0, label='tweaked left edge',zorder=11)
            if tweak_dict['tweak_righ']:
                label = 'threshold = {:5.2f}'.format(tweak_slits_thresh) + ' % of max of right illumprofile'

                plt.hlines(tweak_slits_thresh * tweak_dict['norm_max_righ'], 0.5, ximg_fit.max(), color='red', linewidth=3.0,
                           label=label, zorder=10)
                plt.vlines(tweak_dict['xrigh'],ymin,ymax, color='red',linestyle='--', linewidth = 3.0, label='tweaked right edge',zorder=20)
        plt.legend()
        plt.xlabel('Normalized Slit Position')
        plt.ylabel('Normflat Spatial Profile')
        plt.show()

    msgs.info('Performing illumination + scattered light flat field fit')

    # Flat field pixels for fitting spectral direction
    isrt_spec = np.argsort(piximg[thismask_out])
    pix_twod = piximg[thismask_out][isrt_spec]
    ximg_twod = ximg_out[thismask_out][isrt_spec]
    norm_twod = norm_spec_spat[thismask_out][isrt_spec]

    fitmask = inmask[thismask_out][isrt_spec] & (np.abs(norm_twod - 1.0) < 0.30)
    # Here we ignore the formal photon counting errors and simply assume that a typical error per pixel.
    # This guess is somewhat aribtrary. We then set the rejection threshold with sigrej_illum
    var_value = 0.01
    norm_twod_ivar = fitmask.astype(float)/(var_value**2)
    sigrej_illum = 4.0

    poly_basis = pydl.fpoly(2.0*ximg_twod - 1.0, npoly).T

    # Perform the full 2d fit now
    twod_set, outmask_twod, twodfit, _ = utils.bspline_profile(pix_twod, norm_twod, norm_twod_ivar,poly_basis,
                                                               inmask = fitmask, nord = 4,
                                                               upper=sigrej_illum, lower=sigrej_illum,
                                                               kwargs_bspline = {'bkspace':spec_samp_coarse},
                                                               kwargs_reject={'groupbadpix':True, 'maxrej': 10})

    if debug:
        resid = (norm_twod  - twodfit)
        badpix = np.invert(outmask_twod) & fitmask
        goodpix = outmask_twod & fitmask
        plt.clf()
        ax = plt.gca()
        ax.plot(pix_twod[goodpix], resid[goodpix], color='k', marker='o', markersize=0.2, mfc='k', fillstyle='full',linestyle='None',
                label = 'good points')
        ax.plot(pix_twod[badpix],resid[badpix], color='red', marker='+',markersize=0.5, mfc='red', fillstyle='full', linestyle='None', label='masked')
        plt.hlines(sigrej_illum*var_value,pix_twod.min(),pix_twod.max(), color='lawngreen',linestyle='--',
                   label='rejection thresholds',zorder=10,linewidth=2.0)
        plt.hlines(-sigrej_illum*var_value,pix_twod.min(),pix_twod.max(), color='lawngreen',linestyle='--',
                   zorder=10,linewidth=2.0)
        ax.set_ylim((-0.05,0.05))
        ax.set_xlim((pix_twod.min(), pix_twod.max()))
        plt.legend()
        plt.xlabel('Spectral Pixel')
        plt.ylabel('Residuals from pixelflat 2-d fit')
        plt.show()

        plt.clf()
        ax = plt.gca()
        ax.plot(ximg_twod[goodpix], resid[goodpix], color='k', marker='o', markersize=0.2, mfc='k', fillstyle='full',
                linestyle='None',
                label='good points')
        ax.plot(ximg_twod[badpix], resid[badpix], color='red', marker='+', markersize=0.5, mfc='red', fillstyle='full',
                linestyle='None', label='masked')
        plt.hlines(sigrej_illum * var_value, ximg_twod.min(), ximg_twod.max(), color='lawngreen', linestyle='--',
                   label='rejection thresholds', zorder=10,linewidth=2.0)
        plt.hlines(-sigrej_illum * var_value, ximg_twod.min(), ximg_twod.max(), color='lawngreen', linestyle='--',
                   zorder=10,linewidth=2.0)
        ax.set_ylim((-0.05, 0.05))
        ax.set_xlim(-0.02, 1.02)
        plt.legend()
        plt.xlabel('Normalized Slit Position')
        plt.ylabel('Residuals from pixelflat 2-d fit')
        plt.show()

    # Evaluate and save
    twod_model = np.ones_like(flat)
    twod_this = np.zeros_like(twodfit)
    twod_this[isrt_spec] = twodfit
    twod_model[thismask_out] = twod_this

    # Compute all the final output images output
    pixelflat = np.ones_like(flat)
    flat_model = np.ones_like(flat)
    flat_model[thismask_out] = twod_model[thismask_out]*np.fmax(illumflat[thismask_out],0.05)*np.fmax(spec_model[thismask_out],1.0)
    pixelflat[thismask_out] = flat[thismask_out]/flat_model[thismask_out]

    # ToDo Add some code here to treat the edges and places where fits go bad?
    # Set the pixelflat to 1.0 wherever the flat was nonlinear
    pixelflat[flat >= nonlinear_counts] = 1.0

    return pixelflat, illumflat, flat_model, thismask_out, slit_left_out, slit_righ_out




def flatfield(sciframe, flatframe, bpix, illum_flat=None, snframe=None, varframe=None):
    """ Flat field the input image

    .. todo::
        - Is bpix required?

    Parameters
    ----------
    sciframe : 2d image
    flatframe : 2d image
    illum_flat : 2d image, optional
      slit profile image
    snframe : 2d image, optional
    det : int
      Detector index
    varframe : ndarray
      variance image

    Returns
    -------
    flat-field image
    and updated sigma array if snframe is input
    or updated variance array if varframe is input

    """
    if (varframe is not None) & (snframe is not None):
        msgs.error("Cannot set both varframe and snframe")

    # Fold in the slit profile
    if illum_flat is not None:
        msgs.info('Dividing by illumination flat')
        flatframe *= illum_flat

    # New image
    retframe = np.zeros_like(sciframe)
    w = np.where(flatframe > 0.0)
    retframe[w] = sciframe[w]/flatframe[w]
    if w[0].size != flatframe.size:
        ww = np.where(flatframe <= 0.0)
        bpix[ww] = 1.0
    # Variance?
    if varframe is not None:
        # This is risky -- Be sure your flat is well behaved!!
        retvar = np.zeros_like(sciframe)
        retvar[w] = varframe[w]/flatframe[w]**2
        return retframe, retvar
    # Error image
    if snframe is None:
        return retframe
    else:
        errframe = np.zeros_like(sciframe)
        wnz = np.where(snframe>0.0)
        errframe[wnz] = retframe[wnz]/snframe[wnz]
        return retframe, errframe



# JFH These routines below are all deprecated
def get_ampscale(datasec_img, msflat, namp):
    """ Normalize the flat-field frame

    Parameters
    ----------
    datasec_img : ndarray
    msflat : ndarray
      Flat-field image
    namp : int

    Returns
    -------
    sclframe : ndarray
      A frame to scale all amplifiers to the same counts at the amplifier borders
    """
    sclframe = np.ones_like(msflat)
    ampdone = np.zeros(namp, dtype=int) # 1 = amplifiers have been assigned a scale
    ampdone[0]=1
    while np.sum(ampdone) != namp:
        abst, bbst, nbst, n0bst, n1bst = -1, -1, -1, -1, -1 # Reset the values for the most overlapping amplifier
        for a in range(0, namp): # amplifier 'a' is always the reference amplifier
            if ampdone[a] == 0: continue
            for b in range(0, namp):
                if ampdone[b] == 1 or a == b: continue
                tstframe = np.zeros_like(msflat)
                tstframe[np.where(datasec_img == a+1)] = 1
                tstframe[np.where(datasec_img == b+1)] = 2
                # Determine the total number of adjacent edges between amplifiers a and b
                n0 = np.sum(tstframe[1:,:]-tstframe[:-1,:])
                n1 = np.sum(tstframe[:,1:]-tstframe[:,:-1])
                if (abs(n0)+abs(n1)) > nbst:
                    n0bst = n0
                    n1bst = n1
                    nbst = abs(n0)+abs(n1)
                    abst = a
                    bbst = b
        # Determine the scaling factor for these two amplifiers
        tstframe = np.zeros_like(msflat)
        tstframe[np.where(datasec_img == abst+1)] = 1
        tstframe[np.where(datasec_img == bbst+1)] = 2
        if abs(n0bst) > abs(n1bst):
            # The amplifiers overlap on the zeroth index
            w = np.where(tstframe[1:,:]-tstframe[:-1,:] != 0)
            sclval = np.median(msflat[w[0][0]+1, w[1]])/np.median(msflat[w[0][0], w[1]])
            # msflat[w[0][0], w[1][0:50]] = 1.0E10
            # msflat[w[0][0]-1, w[1][0:50]] = -1.0E10
            # arutils.ds9plot(msflat)
            if n0bst > 0:
                # Then pixel w[0][0] falls on amplifier a
                sclval = sclframe[w[0][0], w[1]] * sclval
            else:
                # pixel w[0][0] falls on amplifier b
                sclval = sclframe[w[0][0]+1, w[1]] / sclval
        else:
            # The amplifiers overlap on the first index
            w = np.where(tstframe[:,1:]-tstframe[:,:-1] != 0)
            sclval = np.median(msflat[w[0], w[1][0]+1]/msflat[w[0], w[1][0]])
            if n1bst > 0:
                # Then pixel w[1][0] falls on amplifier a
                sclval = sclframe[w[0], w[1][0]] * sclval
            else:
                # pixel w[1][0] falls on amplifier b
                sclval = sclframe[w[0], w[1][0]+1] / sclval
        # Finally, apply the scale factor thwe amplifier b
        w = np.where(datasec_img == bbst+1)
        sclframe[w] = np.median(sclval)
        ampdone[bbst] = 1
    return sclframe


def slit_profile(slit, mstrace, tilts, slordloc, srordloc, slitpix, pixwid,
                 ntckx=3, ntcky=20):
    """

    Parameters
    ----------
    slit : int
      Slit (indexed from 0)
    mstrace : ndarray
      Flat field image
    tilts : ndarray
      Tilts image
    slordloc : ndarray (nwave)
      Left edge of the slit
    srordloc : ndarray (nwave)
      Right edge of the slit
    slitpix : ndarray
      Slit pixel image
    pixwid : ndarray
      Slit width array
    ntckx : int, optional
      Spacking of knots in spatial dimensions
    ntcky : int, optional
      Spacking of knots in spectral dimensions

    Returns
    -------
    modvals : ndarray
      Pixels in the slit
    nrmvals : ndarray
      Pixels in the slit
    msblaze_slit : ndarray (nwave)
    blazeext_slit : ndarray (nwave)
    iextrap_slit : float
      0 = Do not extrapolate
      1 = Do extrapolate

    """
    # TODO -- Refactor to use new bspline when it is ready
    iextrap_slit = 0.
    word = np.where(slitpix == slit+1)
    if word[0].size <= (ntcky+1)*(2*pixwid[slit]+1):
        msgs.warn("There are not enough pixels in slit {0:d}".format(slit+1))
        return None, None, None, None, 1.
    spatval = (word[1] - slordloc[word[0]])/(srordloc[word[0]] - slordloc[word[0]])
    specval = tilts[word]
    fluxval = mstrace[word]

    # Only use pixels where at least half the slit is on the chip
    cordloc = 0.5 * (slordloc[word[0]] + srordloc[word[0]])
    wcchip = ((cordloc > 0.0) & (cordloc < mstrace.shape[1]-1.0))

    # Derive the blaze function
    wsp = np.where((spatval > 0.25) & (spatval < 0.75) & wcchip)
    if wsp[0].size <= (ntcky+1)*(2*pixwid[slit]+1):
        msgs.warn("There are not enough pixels in slit {0:d}".format(slit+1))
        return None, None, None, None, 1.
    if (np.min(word[0]) > 0) or (np.max(word[0]) < mstrace.shape[0]-1):
        iextrap_slit = 1.0
    tcky = np.linspace(min(0.0, np.min(specval[wsp])), max(1.0, np.max(specval[wsp])), ntcky)
    tcky = tcky[np.where((tcky > np.min(specval[wsp])) & (tcky < np.max(specval[wsp])))]
    srt = np.argsort(specval[wsp])
    # Only perform a bspline if there are enough pixels for the specified knots
    if tcky.size >= 2:
        yb, ye = min(np.min(specval), tcky[0]), max(np.max(specval), tcky[-1])
        bspline_par = dict(xmin=yb, xmax=ye, everyn=specval[wsp].size//tcky.size)  # knots=tcky)
        mask, blzspl = utils.robust_polyfit(specval[wsp][srt], fluxval[wsp][srt], 3, function='bspline',
                                              sigma=5., maxone=False, bspline_par=bspline_par)
        #xmin=yb, xmax=ye, everyn=specval[wsp].size//tcky.size)  # knots=tcky)
        blz_flat = utils.func_val(blzspl, specval, 'bspline')
        msblaze_slit = utils.func_val(blzspl, np.linspace(0.0, 1.0, slordloc.shape[0]), 'bspline')
    else:
        mask, blzspl = utils.robust_polyfit(specval[wsp][srt], fluxval[wsp][srt], 2, function='polynomial',
                                              sigma=5., maxone=False)
        blz_flat = utils.func_val(blzspl, specval, 'polynomial')
        msblaze_slit = utils.func_val(blzspl, np.linspace(0.0, 1.0, slordloc.shape[0]), 'polynomial')
        iextrap_slit = 1.0

    # Extract a spectrum of the trace frame
    xext = np.arange(mstrace.shape[0])
    yext = np.round(0.5 * (slordloc + srordloc)).astype(np.int)
    wcc = np.where((yext > 0) & (yext < mstrace.shape[1] - 1.0))

    blazeext_slit = np.zeros(slordloc.shape[0])
    blazeext_slit[wcc[0]] = mstrace[(xext[wcc], yext[wcc],)]
    if wcc[0].size != mstrace.shape[0]:
        iextrap_slit = 1.0

    # Calculate the slit profile
    sprof_fit = fluxval / (blz_flat + (blz_flat == 0.0))
    wch = np.where(wcchip)
    tckx = np.linspace(min(0.0, np.min(spatval[wch])), max(1.0, np.max(spatval[wch])), ntckx)
    tckx = tckx[np.where((tckx > np.min(spatval[wch])) & (tckx < np.max(spatval[wch])))]
    srt = np.argsort(spatval[wch])
    # Only perform a bspline if there are enough pixels for the specified knots
    if tckx.size >= 1:
        xb, xe = min(np.min(spatval), tckx[0]), max(np.max(spatval), tckx[-1])
        bspline_par = dict(xmin=xb, xmax=xe, everyn=specval[wch].size//tckx.size)  # knots=tcky)
        mask, sltspl = utils.robust_polyfit(spatval[wch][srt], sprof_fit[wch][srt], 3, function='bspline',
                                              sigma=5., maxone=False, bspline_par=bspline_par)
        #xmin=xb, xmax=xe, everyn=spatval[wch].size//tckx.size)  #, knots=tckx)
        slt_flat = utils.func_val(sltspl, spatval, 'bspline')
        sltnrmval = utils.func_val(sltspl, 0.5, 'bspline')
    else:
        mask, sltspl = utils.robust_polyfit(spatval[srt], sprof_fit[srt], 2, function='polynomial',
                                              sigma=5., maxone=False)
        slt_flat = utils.func_val(sltspl, spatval, 'polynomial')
        sltnrmval = utils.func_val(sltspl, 0.5, 'polynomial')
        iextrap_slit = 1.0

    modvals = blz_flat * slt_flat
    # Normalize to the value at the centre of the slit
    nrmvals = blz_flat * sltnrmval

    return modvals, nrmvals, msblaze_slit, blazeext_slit, iextrap_slit


def prep_ntck(pixwid, method='bspline', params=[20], get_slitprofile=True, ntcky=None):
    """
    Prepare the number of knots for the bspline fitting

    Parameters
    ----------
    pixwid : int
      Width of slit in pixels
    settings : dict
      Could probably replace this with the few parameters needed
    ntcky : int, optional
      Number of knots in spectral

    Returns
    -------
    ntckx : int
    ntcky : int

    """
    # Set the number of knots in the spectral direction
    if ntcky is None:
        if method == 'bspline':
            ntcky = params[0]
            if params[0] < 1.0:
                ntcky = int(1.0/ntcky)+0.5
        else:
            ntcky = 20
    else:
        if ntcky < 1.0:
            ntcky = int(1.0 / ntcky) + 0.5
    ntcky = int(ntcky)
    # Set the number of knots in the spatial direction
    # TODO -- Should this be set per slit/order?
    ntckx = 2 * np.max(pixwid)
    if not get_slitprofile:
        # The slit profile is not needed, so just do the quickest possible fit
        ntckx = 3
    # Return
    return ntckx, ntcky


def norm_slits(mstrace, datasec_img, lordloc, rordloc, pixwid,
                 slitpix, det, tilts, settings_argflag, settings_spect, ntcky=None):
    """ Generate an image of the spatial slit profile.

    DEPRECATED?

    Parameters
    ----------
    mstrace : ndarray
      Master trace frame that is used to trace the slit edges.
    datasec_img : ndarra
      Image of amp positions
    lordloc : ndarray
    rordloc : ndarray
    pixwid : int
    slitpix : ndarray
      Image of slit positions
    det : int
    tilts : ndarray
    settings_argflag : dict
    settings_spect : dict
    ntcky : int, optional
      Number of bspline knots in the spectral direction.

    Returns
    -------
    slit_profile : ndarray
      An image containing the slit profile
    mstracenrm : ndarray
      The input trace frame, normalized by the blaze function (but still contains the slit profile)
    msblaze : ndarray
      A model of the blaze function of each slit
    blazeext : ndarray
      The blaze function extracted down the centre of the slit
    extrap_slit : ndarray
      Mask indicating if a slit is well-determined (0) or poor (1). If the latter, the slit profile
      and blaze function for those slits should be extrapolated or determined from another means
    """
    dnum = parse.get_dnum(det)
    nslits = lordloc.shape[1]

    # First, determine the relative scale of each amplifier (assume amplifier 1 has a scale of 1.0)
    if (settings_spect[dnum]['numamplifiers'] > 1) & (nslits > 1):
        sclframe = get_ampscale(datasec_img, mstrace, settings_spect[dnum]['numamplifiers'])
        # Divide the master flat by the relative scale frame
        mstrace /= sclframe

    mstracenrm = mstrace.copy()
    msblaze = np.ones_like(lordloc)
    blazeext = np.ones_like(lordloc)
    slit_profiles = np.ones_like(mstrace)
    extrap_slit = np.zeros(nslits, dtype=np.int)

    # Tck
    ntckx, ntcky = prep_ntck(pixwid, settings_argflag['reduce'], ntcky=ntcky)

    # Calculate the slit and blaze profiles
    msgs.work("Multiprocess this step")
    for slit in range(nslits):
        word = np.where(slitpix == slit+1)
        modvals, nrmvals, msblaze_slit, blazeext_slit, iextrap_slit = slit_profile(
            slit, mstrace, tilts, lordloc[:,slit], rordloc[:,slit], slitpix,
            pixwid, ntckx=ntckx, ntcky=ntcky)
        if modvals is None:
            extrap_slit[slit] = 1.0
            continue
        else:
            extrap_slit[slit] = iextrap_slit
        #
        if settings_argflag["reduce"]["slitprofile"]["perform"]:
            # Leave slit_profiles as ones if the slitprofile is not being determined, otherwise, set the model.
            slit_profiles[word] = modvals/nrmvals
        mstracenrm[word] /= nrmvals
        # Fill
        msblaze[:,slit] = msblaze_slit
        blazeext[:,slit] = blazeext_slit

    # Return
    return slit_profiles, mstracenrm, msblaze, blazeext, extrap_slit


def slit_profile_pca(mstrace, tilts, msblaze, extrap_slit, slit_profiles,
                     lordloc, rordloc, pixwid, slitpix, setup, debug=False):
    """ Perform a PCA analysis on the spatial slit profile and blaze function.

    Parameters
    ----------
    mstrace : ndarray
    tilts : ndarray
    msblaze : ndarray
      A model of the blaze function of each slit
    extrap_slit : ndarray
      Mask indicating if a slit is well-determined (0) or poor (1). If the latter, the slit profile
      and blaze function for those slits should be extrapolated or determined from another means
    slit_profiles : ndarray
      An image containing the slit profile
    lordloc : ndarray
    rordloc : ndarray
    pixwid : ndarray
    slitpix : ndarray
    setup : str

    Returns
    -------
    slit_profiles : ndarray
      An image containing the slit profile
    mstracenrm : ndarray
      The input trace frame, normalized by the blaze function (but still contains the slit profile)
    extrap_blz : ndarray
      A model of the blaze function of each slit
    """
    #################
    # Parameters to include in settings file
    fitfunc = "legendre"
    ordfit = 4
    ofit = [2, 3, 3, 2, 2]
    sordfit = 2
    sofit = [1, 3, 1]
    #################

    nslits = extrap_slit.size
    gds = np.where(extrap_slit == 0)
    maskord = np.where(extrap_slit == 1)[0]
    specfit = np.arange(mstrace.shape[0])
    nspec = np.max(pixwid)*10
    spatbins = np.linspace(-0.25, 1.25, nspec + 1)
    # Perform a PCA on the spectral (i.e. blaze) function
    blzmxval = np.ones((1, nslits))
    lorr = 0
    for o in range(0, nslits):
        # if extrap_slit[o] == 1:
        #     continue
        # Find which pixels are on the slit
        wch = np.where((lordloc[:, o] > 0.0) &
                       (rordloc[:, o] < mstrace.shape[1]-1.0))
        cordloc = np.round(0.5 * (lordloc[:, o] + rordloc[:, o])).astype(np.int)
        if wch[0].size < mstrace.shape[0]:
            # The entire order is not on the chip
            if cordloc[int(0.5*mstrace.shape[0])] < mstrace.shape[1]/2:
                lorr = -1  # Once a full order is found, go left
                continue
            else:
                lorr = +1  # Go right
        else:
            blzmxval[0, o] = np.median(mstrace[wch[0], cordloc[wch]])
        if lorr == -1:
            # A full order has been found, go back and fill in the gaps
            for i in range(1, o+1):
                wch = np.where((lordloc[:, o-i] > 0.0) &
                               (rordloc[:, o-i] < mstrace.shape[1] - 1.0))
                # Calculate the previous order flux
                cordloc = np.round(0.5 * (lordloc[:, o-i+1] + rordloc[:, o-i+1])).astype(np.int)
                prval = mstrace[wch[0], cordloc[wch]]
                # Calculate the current order flux
                cordloc = np.round(0.5 * (lordloc[:, o-i] + rordloc[:, o-i])).astype(np.int)
                mnval = mstrace[wch[0], cordloc[wch]]
                wnz = np.where(prval != 0.0)
                blzmxval[0, o-i] = blzmxval[0, o-i+1] * np.median(mnval[wnz] / prval[wnz])
            lorr = 0
        elif lorr == +1:
            # Calibrate the current order with the previous one
            mnval = mstrace[wch[0], cordloc[wch]]
            cordloc = np.round(0.5 * (lordloc[:, o-1] + rordloc[:, o-1])).astype(np.int)
            prval = mstrace[wch[0], cordloc[wch]]
            wnz = np.where(prval != 0.0)
            blzmxval[0, o] = blzmxval[0, o-1] * np.median(mnval[wnz] / prval[wnz])
            lorr = 0

    # Check for nan values (i.e. when median is given a zero element array)
    blznan = np.isnan(blzmxval[0, :])
    if np.any(blznan):
        # Find the acceptable values and linearly interpolate
        blzx = np.arange(nslits)
        wnnan = np.where(~blznan)
        fblz = interpolate.interp1d(blzx[wnnan], blzmxval[0, wnnan],
                                    kind="linear", bounds_error=False, fill_value="extrapolate")
        blzmxval = fblz(blzx).reshape(blzmxval.shape)
    elif np.all(blznan):
        msgs.bug("All of the blaze values are NaN... time to debug")
        debugger.set_trace()

    # Calculate the mean blaze function of all good orders
    blzmean = np.mean(msblaze[:, gds[0]], axis=1)
    blzmean /= np.max(blzmean)
    blzmean = blzmean.reshape((blzmean.size, 1))
    msblaze /= blzmean
    msblaze /= blzmxval
    # Fit the blaze functions
    fitcoeff = np.ones((ordfit+1, nslits))
    for o in range(nslits):
        if extrap_slit[o] == 1:
            continue
        wmask = np.where(msblaze[:, o] != 0.0)[0]
        null, bcoeff = utils.robust_polyfit(specfit[wmask], msblaze[wmask, o],
                                              ordfit, function=fitfunc, sigma=2.0,
                                              minv=0.0, maxv=mstrace.shape[0])
        fitcoeff[:, o] = bcoeff

    lnpc = len(ofit) - 1
    xv = np.arange(mstrace.shape[0])
    blzval = utils.func_val(fitcoeff, xv, fitfunc,
                              minv=0.0, maxv=mstrace.shape[0] - 1).T
    # Only do a PCA if there are enough good orders
    if np.sum(1.0 - extrap_slit) > ofit[0] + 1:
        # Perform a PCA on the tilts
        msgs.info("Performing a PCA on the spectral blaze function")
        ordsnd = np.arange(nslits) + 1.0
        xcen = xv[:, np.newaxis].repeat(nslits, axis=1)
        fitted, outpar = pca.basis(xcen, blzval, fitcoeff, lnpc, ofit, x0in=ordsnd, mask=maskord, skipx0=False,
                                     function=fitfunc)
        if not debug:
#            arqa.pca_plot(slf, outpar, ofit, "Blaze_Profile", pcadesc="PCA of blaze function fits")
            pca.pca_plot(slf.setup, outpar, ofit, "Blaze_Profile",
                           pcadesc="PCA of blaze function fits")
        # Extrapolate the remaining orders requested
        orders = 1.0 + np.arange(nslits)
        extrap_blz, outpar = pca.extrapolate(outpar, orders, function=fitfunc)
        extrap_blz *= blzmean
        extrap_blz *= blzmxval
    else:
        msgs.warn("Could not perform a PCA on the order blaze function" + msgs.newline() +
                  "Not enough well-traced orders")
        msgs.info("Using direct determination of the blaze function instead")
        extrap_blz = msblaze*blzmean

    # Normalize the trace frame, but don't remove the slit profile
    mstracenrm = mstrace.copy()
    for o in range(nslits):
        word = np.where(slitpix == o+1)
        specval = tilts[word]
        blzspl = interpolate.interp1d(np.linspace(0.0, 1.0, mstrace.shape[0]), extrap_blz[:, o],
                                      kind="linear", fill_value="extrapolate")
        mstracenrm[word] /= blzspl(specval)

    # Now perform a PCA on the spatial (i.e. slit) profile
    # First generate the original model of the spatial slit profiles
    msslits = np.zeros((nspec, nslits))
    mskslit = np.ones((nspec, nslits))
    for o in range(nslits):
        if extrap_slit[o] == 1:
            continue
        word = np.where(slitpix == o+1)
        spatval = (word[1] + 0.5 - lordloc[:, o][word[0]]) /\
                  (rordloc[:, o][word[0]] - lordloc[:, o][word[0]])
        groups = np.digitize(spatval, spatbins)
        modelw = slit_profiles[word]
        for mm in range(1, spatbins.size):
            tmp = modelw[groups == mm]
            if tmp.size != 0.0:
                msslits[mm - 1, o] = tmp.mean()
            else:
                mskslit[mm - 1, o] = 0.0

    # Calculate the spatial profile of all good orders
    sltmean = np.mean(msslits[:, gds[0]], axis=1)
    sltmean = sltmean.reshape((sltmean.size, 1))
    msslits /= (sltmean + (sltmean == 0))

    # Fit the spatial profiles
    spatfit = 0.5*(spatbins[1:]+spatbins[:-1])
    fitcoeff = np.ones((sordfit+1, nslits))
    for o in range(nslits):
        if extrap_slit[o] == 1:
            continue
        wmask = np.where(mskslit[:, o] == 1.0)[0]
        null, bcoeff = utils.robust_polyfit(spatfit[wmask], msslits[wmask, o],
                                              sordfit, function=fitfunc, sigma=2.0,
                                              minv=spatfit[0], maxv=spatfit[-1])
        fitcoeff[:, o] = bcoeff

    lnpc = len(sofit) - 1
    sltval = utils.func_val(fitcoeff, spatfit, fitfunc,
                              minv=spatfit[0], maxv=spatfit[-1]).T
    # Only do a PCA if there are enough good orders
    if np.sum(1.0 - extrap_slit) > sofit[0] + 1:
        # Perform a PCA on the tilts
        msgs.info("Performing a PCA on the spatial slit profiles")
        ordsnd = np.arange(nslits) + 1.0
        xcen = spatfit[:, np.newaxis].repeat(nslits, axis=1)
        fitted, outpar = pca.basis(xcen, sltval, fitcoeff, lnpc, sofit, x0in=ordsnd, mask=maskord, skipx0=False,
                                     function=fitfunc)
        if not debug:
#            arqa.pca_plot(slf, outpar, sofit, "Slit_Profile", pcadesc="PCA of slit profile fits")
            pca.pca_plot(setup, outpar, sofit, "Slit_Profile", pcadesc="PCA of slit profile fits")
        # Extrapolate the remaining orders requested
        orders = 1.0 + np.arange(nslits)
        extrap_slt, outpar = pca.extrapolate(outpar, orders, function=fitfunc)
        extrap_slt *= sltmean
        extrap_slt *= mskslit
    else:
        msgs.warn("Could not perform a PCA on the spatial slit profiles" + msgs.newline() +
                  "Not enough well-traced orders")
        msgs.info("Using direct determination of the slit profiles instead")
        extrap_slt = (msslits*mskslit)*sltmean

    # Normalize the trace frame, but don't remove the slit profile
    slit_profiles = np.ones_like(mstrace)
    for o in range(nslits):
        tlordloc = lordloc[:, o]
        trordloc = rordloc[:, o]
        word = np.where(slitpix == o+1)
        spatval = (word[1] - tlordloc[word[0]])/(trordloc[word[0]] - tlordloc[word[0]])

        sltspl = interpolate.interp1d(spatfit, extrap_slt[:, o],
                                      kind="linear", fill_value="extrapolate")
        slit_profiles[word] = sltspl(spatval)

    return slit_profiles, mstracenrm, extrap_blz



def slit_profile_qa(mstrace, model, lordloc, rordloc, msordloc, textplt="Slit", maxp=16, desc="",
                    setup=None, outroot=None):
    """ Generate a QA plot for the slit profile of each slit

    Parameters
    ----------
    mstrace : ndarray
      trace frame
    model : ndarray
      model of slit profiles, same shape as frame.
    lordloc : ndarray
      left edge locations of all slits
    rordloc : ndarray
      right edge locations of all slits
    msordloc : ndarray
      An array the same size as frame that determines which pixels contain a given order.
    textplt : str, optional
      A string printed above each panel
    maxp : int, (optional)
      Maximum number of panels per page
    desc : str, (optional)
      A description added to the top of each page
    setup : str, optional
    outroot : str, optional
    """

    plt.rcdefaults()
    plt.rcParams['font.family']= 'times new roman'

    # Outfile
    method = inspect.stack()[0][3]
    if outroot is None:
        outroot = qa.set_qa_filename(setup, method)

    npix, nord = lordloc.shape
    nbins = 40
    bins = np.linspace(-0.25, 1.25, nbins+1)
    pages, npp = qa.get_dimen(nord, maxp=maxp)
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
            # Get data to be plotted
            word = np.where(msordloc == ndone+j+1)
            if word[0].size == 0:
                msgs.warn("There are no pixels in slit {0:d}".format(ndone + j + 1))
                # Delete the axis
                if pages[i][1] == 1: ind = (ipx,)
                elif pages[i][0] == 1: ind = (ipy,)
                else: ind = (ipy, ipx)
                f.delaxes(axes[ind])
                ipx += 1
                if ipx == pages[i][0]:
                    ipx = 0
                    ipy += 1
                continue
            spatval = (word[1] + 0.5 - lordloc[:, ndone+j][word[0]]) / (rordloc[:, ndone+j][word[0]] - lordloc[:, ndone+j][word[0]])
            fluxval = mstrace[word]
            mxval = 1.25
            modvals = np.zeros(nbins)
            if axesIdx:
                cnts, xedges, yedges, null = axes[ind].hist2d(spatval, fluxval, bins=bins, cmap=plt.cm.gist_heat_r)
                groups = np.digitize(spatval, xedges)
                modelw = model[word]
                for mm in range(1, xedges.size):
                    modvals[mm-1] = modelw[groups == mm].mean()
                axes[ind].plot(0.5*(xedges[1:]+xedges[:-1]), modvals, 'b-', linewidth=2.0)
                axes[ind].plot([0.0, 0.0], [0.0, mxval], 'r-')
                axes[ind].plot([1.0, 1.0], [0.0, mxval], 'r-')
            else:
                cnts, xedges, yedges, null = axes.hist2d(spatval, fluxval, bins=bins, cmap=plt.cm.gist_heat_r)
                groups = np.digitize(spatval, xedges)
                modelw = model[word]
                for mm in range(1, xedges.size):
                    modvals[mm-1] = modelw[groups == mm].mean()
                axes.plot(0.5*(xedges[1:]+xedges[:-1]), modvals, 'b-', linewidth=2.0)
                axes.plot([0.0, 0.0], [0.0, mxval], 'r-')
                axes.plot([1.0, 1.0], [0.0, mxval], 'r-')
            if axesIdx:
                axes[ind].axis([xedges[0], xedges[-1], 0.0, 1.1*mxval])
                axes[ind].set_title("{0:s} {1:d}".format(textplt, ndone+j+1))
                axes[ind].tick_params(labelsize=10)
            else:
                axes.axis([xedges[0], xedges[-1], 0.0, 1.1*mxval])
                axes.set_title("{0:s} {1:d}".format(textplt, ndone+j+1))
                axes.tick_params(labelsize=10)
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
                ipx += 1
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


'''
def sn_frame(slf, sciframe, idx):
    # Dark Current noise
    dnoise = settings.spect['det']['darkcurr'] * float(slf._fitsdict["exptime"][idx])/3600.0
    # The effective read noise
    rnoise = np.sqrt(settings.spect['det']['ronoise']**2 + (0.5*settings.spect['det']['gain'])**2)
    errframe = np.abs(sciframe) + rnoise + dnoise
    # If there are negative pixels, mask them as bad pixels
    w = np.where(errframe <= 0.0)
    if w[0].size != 0:
        msgs.warn("The error frame is negative for {0:d} pixels".format(w[0].size)+msgs.newline()+"Are you sure the bias frame is correct?")
        msgs.info("Masking these {0:d} pixels".format(w[0].size))
        errframe[w]  = 0.0
        slf._bpix[w] = 1.0
    w = np.where(errframe > 0.0)
    snframe = np.zeros_like(sciframe)
    snframe[w] = sciframe[w]/np.sqrt(errframe[w])
    return snframe
'''


def flatfield(sciframe, flatframe, bpix, illum_flat=None, snframe=None, varframe=None):
    """ Flat field the input image

    .. todo::
        - Is bpix required?

    Parameters
    ----------
    sciframe : 2d image
    flatframe : 2d image
    illum_flat : 2d image, optional
      slit profile image
    snframe : 2d image, optional
    det : int
      Detector index
    varframe : ndarray
      variance image

    Returns
    -------
    flat-field image
    and updated sigma array if snframe is input
    or updated variance array if varframe is input

    """
    if (varframe is not None) & (snframe is not None):
        msgs.error("Cannot set both varframe and snframe")

    # Fold in the slit profile
    if illum_flat is not None:
        msgs.info('Dividing by illumination flat')
        flatframe *= illum_flat

    # New image
    retframe = np.zeros_like(sciframe)
    w = np.where(flatframe > 0.0)
    retframe[w] = sciframe[w]/flatframe[w]
    if w[0].size != flatframe.size:
        ww = np.where(flatframe <= 0.0)
        bpix[ww] = 1.0
    # Variance?
    if varframe is not None:
        # This is risky -- Be sure your flat is well behaved!!
        retvar = np.zeros_like(sciframe)
        retvar[w] = varframe[w]/flatframe[w]**2
        return retframe, retvar
    # Error image
    if snframe is None:
        return retframe
    else:
        errframe = np.zeros_like(sciframe)
        wnz = np.where(snframe>0.0)
        errframe[wnz] = retframe[wnz]/snframe[wnz]
        return retframe, errframe


'''
def flatnorm(slf, det, msflat, bpix, maskval=-999999.9, overpix=6, plotdesc=""):
    """ Normalize the flat-field frame

    *** CAUTION ***  This function might be deprecated.

    Parameters
    ----------
    slf : class
      An instance of the Science Exposure class
    det : int
      Detector number
    msflat : ndarray
      Flat-field image
    maskval : float
      Global floating point mask value used throughout the code
    overpix : int
      overpix/2 = the number of pixels to extend beyond each side of the order trace
    plotdesc : str
      A title for the plotted QA

    Returns
    -------
    msnormflat : ndarray
      The normalized flat-field frame
    msblaze : ndarray
      A 2d array containing the blaze function for each slit
    """
    dnum = settings.get_dnum(det)

    msgs.info("Normalizing the master flat field frame")
    norders = slf._lordloc[det-1].shape[1]
    # First, determine the relative scale of each amplifier (assume amplifier 1 has a scale of 1.0)
    if (settings.spect[dnum]['numamplifiers'] > 1) & (norders > 1):
        sclframe = get_ampscale(slf, det, msflat)
        # Divide the master flat by the relative scale frame
        msflat /= sclframe
    else:
        sclframe = np.ones(msflat, dtype=np.float)
    # Determine the blaze
    polyord_blz = 2  # This probably doesn't need to be a parameter that can be set by the user
    # Look at the end corners of the detector to get detector size in the dispersion direction
    #xstr = slf._pixlocn[det-1][0,0,0]-slf._pixlocn[det-1][0,0,2]/2.0
    #xfin = slf._pixlocn[det-1][-1,-1,0]+slf._pixlocn[det-1][-1,-1,2]/2.0
    #xint = slf._pixlocn[det-1][:,0,0]
    # Find which pixels are within the order edges
    msgs.info("Identifying pixels within each order")

    ordpix = order_pixels(slf._pixlocn[det-1], slf._lordloc[det-1], slf._rordloc[det-1])

    msgs.info("Applying bad pixel mask")
    ordpix *= (1-bpix.astype(np.int))
    mskord = np.zeros(msflat.shape)
    msgs.info("Rectifying the orders to estimate the background locations")
    #badorders = np.zeros(norders)
    msnormflat = maskval*np.ones_like(msflat)
    msblaze = maskval*np.ones((msflat.shape[0],norders))
    msgs.work("Must consider different amplifiers when normalizing and determining the blaze function")
    msgs.work("Multiprocess this step to make it faster")
    flat_ext1d = maskval*np.ones((msflat.shape[0],norders))
    for o in range(norders):
        if settings.argflag["reduce"]["flatfield"]["method"].lower() == "bspline":
            msgs.info("Deriving blaze function of slit {0:d} with a bspline".format(o+1))
            tilts = slf._tilts[det - 1].copy()
            gdp = (msflat != maskval) & (ordpix == o + 1)
            srt = np.argsort(tilts[gdp])
            everyn = settings.argflag['reduce']['flatfield']['params'][0]
            if everyn > 0.0 and everyn < 1.0:
                everyn *= msflat.shape[0]
                everyn = int(everyn + 0.5)
            everyn *= slf._pixwid[det - 1][o]
            if np.where(gdp)[0].size < 2*everyn:
                msgs.warn("Not enough pixels in slit {0:d} to fit a bspline")
                continue
            bspl = utils.func_fit(tilts[gdp][srt], msflat[gdp][srt], 'bspline', 3, everyn=everyn)
            model_flat = utils.func_val(bspl, tilts.flatten(), 'bspline')
            model = model_flat.reshape(tilts.shape)
            word = np.where(ordpix == o + 1)
            msnormflat[word] = msflat[word] / model[word]
            msblaze[:, o] = utils.func_val(bspl, np.linspace(0.0, 1.0, msflat.shape[0]), 'bspline')
            mskord[word] = 1.0
            flat_ext1d[:, o] = np.sum(msflat * mskord, axis=1) / np.sum(mskord, axis=1)
            mskord *= 0.0
        else:
            msgs.error("Flatfield method {0:s} is not supported".format(settings.argflag["reduce"]["flatfield"]["method"]))
    # Send the blaze away to be plotted and saved
    if "2dpca" in settings.argflag["reduce"]["flatfield"].keys():
        if settings.argflag["reduce"]["flatfield"]["2dpca"] >= 1:
            msgs.info("Performing a 2D PCA on the blaze fits")
            msblaze = arpca.pca2d(msblaze, settings.argflag["reduce"]["flatfield"]["2dpca"])
    # Plot the blaze model
    if not msgs._debug['no_qa']:
        msgs.info("Saving blaze fits to QA")
#        arqa.plot_orderfits(slf, msblaze, flat_ext1d, desc=plotdesc, textplt="Order")
        artrace.plot_orderfits(slf, msblaze, flat_ext1d, desc=plotdesc, textplt="Order")
    # If there is more than 1 amplifier, apply the scale between amplifiers to the normalized flat
    if (settings.spect[dnum]['numamplifiers'] > 1) & (norders > 1):
        msnormflat *= sclframe
    return msnormflat, msblaze
'''
