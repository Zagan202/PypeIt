""" Primary module for guiding the reduction of long slit data
"""
from __future__ import (print_function, absolute_import, division, unicode_literals)

import numpy as np
import yaml
from pypit import arparse as settings
from pypit import arflux
from pypit import arload
from pypit import armasters
from pypit import armbase
#from pypit import armsgs
from pypit import msgs
from pypit import arproc
from pypit import arsave
from pypit import arsetup
from pypit import artrace
from astropy import units
#from pypit import arqa

from linetools import utils as ltu

from pypit import ardebug as debugger

# Logging
#msgs = armsgs.get_logger()


def ARMLSD(fitsdict, reuseMaster=False, reloadMaster=True):
    """
    Automatic Reduction and Modeling of Long Slit Data

    Parameters
    ----------
    fitsdict : dict
      Contains relevant information from fits header files
    reuseMaster : bool
      If True, a master frame that will be used for another science frame
      will not be regenerated after it is first made.
      This setting comes with a price, and if a large number of science frames are
      being generated, it may be more efficient to simply regenerate the master
      calibrations on the fly.

    Returns
    -------
    status : int
      Status of the reduction procedure
      0 = Successful full execution
      1 = Successful processing of setup or calcheck
    """
    status = 0

    # Create a list of science exposure classes
    sciexp, setup_dict = armbase.setup_science(fitsdict)
    if sciexp == 'setup':
        status = 1
        return status
    elif sciexp == 'calcheck':
        status = 2
        return status
    else:
        numsci = len(sciexp)

    # Create a list of master calibration frames
    #masters = armasters.MasterFrames(settings.spect['mosaic']['ndet'])

    # Masters
    #settings.argflag['reduce']['masters']['file'] = setup_file

    # Start reducing the data
    for sc in range(numsci):

        slf = sciexp[sc]
        scidx = slf._idx_sci[0]
        msgs.info("Reducing file {0:s}, target {1:s}".format(fitsdict['filename'][scidx], slf._target_name))
        msgs.sciexp = slf  # For QA writing on exit, if nothing else.  Could write Masters too
        if reloadMaster and (sc > 0):
            settings.argflag['reduce']['masters']['reuse'] = True
        # Loop on Detectors
        for kk in range(settings.spect['mosaic']['ndet']):
            det = kk + 1  # Detectors indexed from 1
            if settings.argflag['reduce']['detnum'] is not None:
                if det not in map(int,settings.argflag['reduce']['detnum']):
                    msgs.warn("Skipping detector {:d}".format(det))
                    continue
                else:
                    msgs.warn("Restricting the reduction to detector {:d}".format(det))
            slf.det = det
            ###############
            # Get data sections
            arproc.get_datasec_trimmed(slf, fitsdict, det, scidx)
            # Setup
            setup = arsetup.instr_setup(slf, det, fitsdict, setup_dict, must_exist=True)
            settings.argflag['reduce']['masters']['setup'] = setup
            slf.setup = setup
            ###############
            # Generate master bias frame
            update = slf.MasterBias(fitsdict, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="bias")
            ###############
            # Generate a bad pixel mask (should not repeat)
            update = slf.BadPixelMask(fitsdict, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="arc")
            ###############
            # Generate a master arc frame
            update = slf.MasterArc(fitsdict, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="arc")
            ###############
            # Set the number of spectral and spatial pixels, and the bad pixel mask is it does not exist
            slf._nspec[det-1], slf._nspat[det-1] = slf._msarc[det-1].shape
            if slf._bpix[det-1] is None:
                slf.SetFrame(slf._bpix, np.zeros((slf._nspec[det-1], slf._nspat[det-1])), det)
            '''
            ###############
            # Estimate gain and readout noise for the amplifiers
            msgs.work("Estimate Gain and Readout noise from the raw frames...")
            update = slf.MasterRN(fitsdict, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="readnoise")
            '''
            ###############
            # Generate a master trace frame
            update = slf.MasterTrace(fitsdict, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="flat", chktype="trace")
            ###############
            # Generate an array that provides the physical pixel locations on the detector
            slf.GetPixelLocations(det)
            # Determine the edges of the spectrum (spatial)
            if ('trace'+settings.argflag['reduce']['masters']['setup'] not in settings.argflag['reduce']['masters']['loaded']):
                ###############
                # Determine the edges of the spectrum (spatial)
                lordloc, rordloc, extord = artrace.trace_slits(slf, slf._mstrace[det-1], det, pcadesc="PCA trace of the slit edges")
                slf.SetFrame(slf._lordloc, lordloc, det)
                slf.SetFrame(slf._rordloc, rordloc, det)

                # Convert physical trace into a pixel trace
                msgs.info("Converting physical trace locations to nearest pixel")
                pixcen = artrace.phys_to_pix(0.5*(slf._lordloc[det-1]+slf._rordloc[det-1]), slf._pixlocn[det-1], 1)
                pixwid = (slf._rordloc[det-1]-slf._lordloc[det-1]).mean(0).astype(np.int)
                lordpix = artrace.phys_to_pix(slf._lordloc[det-1], slf._pixlocn[det-1], 1)
                rordpix = artrace.phys_to_pix(slf._rordloc[det-1], slf._pixlocn[det-1], 1)
                slf.SetFrame(slf._pixcen, pixcen, det)
                slf.SetFrame(slf._pixwid, pixwid, det)
                slf.SetFrame(slf._lordpix, lordpix, det)
                slf.SetFrame(slf._rordpix, rordpix, det)
                msgs.info("Identifying the pixels belonging to each slit")
                slitpix = arproc.slit_pixels(slf, slf._mstrace[det-1].shape, det)
                slf.SetFrame(slf._slitpix, slitpix, det)
                # Save to disk
                armasters.save_masters(slf, det, mftype='trace')
                # Save QA for slit traces
#                arqa.slit_trace_qa(slf, slf._mstrace[det-1], slf._lordpix[det-1],
#                                       slf._rordpix[det-1], extord,
#                                       desc="Trace of the slit edges D{:02d}".format(det), use_slitid=det)
                artrace.slit_trace_qa(slf, slf._mstrace[det-1], slf._lordpix[det-1],
                                      slf._rordpix[det-1], extord,
                                      desc="Trace of the slit edges D{:02d}".format(det),
                                      use_slitid=det)
                armbase.UpdateMasters(sciexp, sc, det, ftype="flat", chktype="trace")

            ###############
            # Generate the 1D wavelength solution
            update = slf.MasterWaveCalib(fitsdict, sc, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="arc", chktype="trace")

            ###############
            # Derive the spectral tilt
            if slf._tilts[det-1] is None:
                try:
                    tilts = armasters.get_master_frame(slf, "tilts")
                except IOError:
                    # First time tilts are derived for this arc frame --> derive the order tilts
                    tilts, satmask, outpar = artrace.multislit_tilt(slf, slf._msarc[det-1], det)
                    slf.SetFrame(slf._tilts, tilts, det)
                    slf.SetFrame(slf._satmask, satmask, det)
                    slf.SetFrame(slf._tiltpar, outpar, det)
                    armasters.save_masters(slf, det, mftype='tilts')
                else:
                    slf.SetFrame(slf._tilts, tilts, det)

            ###############
            # Prepare the pixel flat field frame
            update = slf.MasterFlatField(fitsdict, det)
            if update and reuseMaster: armbase.UpdateMasters(sciexp, sc, det, ftype="flat", chktype="pixelflat")

            ###############
            # Generate/load a master wave frame
            update = slf.MasterWave(fitsdict, sc, det)
            if update and reuseMaster:
                armbase.UpdateMasters(sciexp, sc, det, ftype="arc", chktype="wave")

            ###############
            # Check if the user only wants to prepare the calibrations only
            msgs.info("All calibration frames have been prepared")
            if settings.argflag['run']['preponly']:
                msgs.info("If you would like to continue with the reduction, disable the command:" + msgs.newline() +
                          "run preponly False")
                continue

            ###############
            # Write setup
            #setup = arsort.calib_setup(sc, det, fitsdict, setup_dict, write=True)
            # Write MasterFrames (currently per detector)
            #armasters.save_masters(slf, det, setup)

            ###############
            # Load the science frame and from this generate a Poisson error frame
            msgs.info("Loading science frame")
            sciframe = arload.load_frames(fitsdict, [scidx], det,
                                          frametype='science',
                                          msbias=slf._msbias[det-1])
            sciframe = sciframe[:, :, 0]
            # Extract
            msgs.info("Processing science frame")
            arproc.reduce_multislit(slf, sciframe, scidx, fitsdict, det)

            ###############
            # Using model sky, calculate a flexure correction

        ###############
        # Flux
        ###############
        if(settings.argflag['reduce']['calibrate']['flux']==True):
            # Standard star (is this a calibration, e.g. goes above?)
            msgs.info("Processing standard star")
            msgs.info("Assuming one star per detector mosaic")
            msgs.info("Waited until last detector to process")

            if(settings.argflag['reduce']['calibrate']['sensfunc']['archival']=='None'):
                update = slf.MasterStandard(fitsdict)
                if update and reuseMaster:
                    armbase.UpdateMasters(sciexp, sc, 0, ftype="standard")
            else:
                sensfunc = yaml.load(open(settings.argflag['reduce']['calibrate']['sensfunc']['archival']))
                # Yaml does not do quantities, so make the sensfunc min/max wave quantities
                sensfunc['wave_max']*=units.angstrom
                sensfunc['wave_min']*=units.angstrom
                slf.SetMasterFrame(sensfunc, "sensfunc", None, mkcopy=False)
                msgs.info("Using archival sensfunc {:s}".format(settings.argflag['reduce']['calibrate']['sensfunc']['archival']))

            msgs.info("Fluxing with {:s}".format(slf._sensfunc['std']['name']))
            for kk in range(settings.spect['mosaic']['ndet']):
                det = kk + 1  # Detectors indexed from 1
                if slf._specobjs[det-1] is not None:
                    arflux.apply_sensfunc(slf, det, scidx, fitsdict)
                else:
                    msgs.info("There are no objects on detector {0:d} to apply a flux calibration".format(det))

        # Write 1D spectra
        save_format = 'fits'
        if save_format == 'fits':
            arsave.save_1d_spectra_fits(slf, fitsdict)
        elif save_format == 'hdf5':
            arsave.save_1d_spectra_hdf5(slf)
        else:
            msgs.error(save_format + ' is not a recognized output format!')
        arsave.save_obj_info(slf, fitsdict)
        # Write 2D images for the Science Frame
        arsave.save_2d_images(slf, fitsdict)
        # Free up some memory by replacing the reduced ScienceExposure class
        sciexp[sc] = None
    return status
