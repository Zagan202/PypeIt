

import numpy as np
from astropy.io import fits
from astropy.table import Table
from pydl.pydlutils.trace import TraceSet
from pypit import ginga
from matplotlib import pyplot as plt

path = '/Users/joe/REDUX/lris_redux/May_2008/0938+5317/Blue1200/'

# slitfile
slitfile = path + 'slits-lblue0059.fits.gz'
# science image
scifile = path + 'Science/sci-lblue0084.fits.gz'

# Open up science output
hdu_sci = fits.open(scifile)
sciimg = hdu_sci[0].data
sciivar = hdu_sci[1].data
sky_model = hdu_sci[2].data
objstruct = Table.read(scifile,hdu=5)
# Grab the object trace
nobj = len(objstruct)
trace_in = objstruct['XPOS']
slit_str = np.array(objstruct['SLITID'],dtype=str)
obj_str = np.array(objstruct['OBJID'],dtype=str)
id_str = [x + '-' + y for x,y in zip(slit_str,obj_str)]
title_string = np.core.defchararray.add(np.array(objstruct['SLITID'],dtype=str),np.array(objstruct['OBJID'],dtype=str))

# Open up slitfile and read in tset_slits
tset_fits = Table.read(slitfile)
left_dum = Table(tset_fits[0])
righ_dum = Table(tset_fits[1])
left_dum.write('left_dum.fits',overwrite=True)
righ_dum.write('righ_dum.fits',overwrite=True)


hdu_left = fits.open('left_dum.fits')
fits_rec_left = hdu_left[1].data
hdu_righ = fits.open('righ_dum.fits')
fits_rec_righ = hdu_righ[1].data
#tset_fits = hdu[1].data
#tset_left = Table(ttset_fits[0],'dum_file.fits')
# Convert to python objects
tset_left = TraceSet(fits_rec_left)
tset_righ = TraceSet(fits_rec_righ)
# Do the initialization here
#
rows,left_edge = tset_left.xy()
rows,righ_edge = tset_righ.xy()
nslits = tset_left.nTrace

# Let's take a look at the image and the slits
viewer, ch = ginga.show_image(sciimg-sky_model)
ginga.show_slits(viewer, ch, left_edge.T, righ_edge.T,np.arange(nslits+1))
for iobj in range(nobj):
    ginga.show_trace(viewer, ch, trace_in[iobj,:],id_str[iobj], color='green')

# parameters for extract_asymbox2
box_rad = 7
left = trace_in[5:,:] - box_rad
right = trace_in[5:,:] + box_rad
image = sciimg - sky_model
ivar = sciivar
ycen = None
model = None
weight_image = None


dim = left.shape
ndim = len(dim)
npix = dim[1]
if (ndim == 1):
    nTrace = 1
else:
    nTrace = dim[0]

if ycen == None:
    if ndim == 1:
        ycen = np.arange(npix,dtype='int')
    elif ndim == 2:
        ycen = np.outer(np.ones(nTrace,dtype='int'),np.arange(npix,dtype='int'),)
    else:
        raise ValueError('left is not 1 or 2 dimensional')

if np.size(left) != np.size(ycen):
    raise ValueError('Number of elements in LEFT and YCEN must be equal')

idims = image.shape
nx = idims[1]
ny = idims[0]
ncen = np.size(left)

maxwindow = np.max(right - left)
tempx = np.int(maxwindow + 3.0)

bigleft = np.outer(left[:],np.ones(tempx))
bigright = np.outer(right[:],np.ones(tempx))
spot = np.outer(np.ones(npix*nTrace),np.arange(tempx)) + bigleft - 1
bigy = np.outer(ycen[:],np.ones(tempx,dtype='int'))

#bigleft = np.outer(np.ones(tempx),left[:])
#bigright = np.outer(np.ones(tempx),right[:])
#spot = np.outer(np.arange(tempx),np.ones(npix*nTrace)) + bigleft - 1
#bigy = np.outer(np.ones(tempx,dtype='int'),ycen[:])

fullspot = np.array(np.fmin(np.fmax(np.round(spot + 1) - 1,0),nx-1),int)
fracleft = np.fmax(np.fmin(fullspot - bigleft,0.5),-0.5)
fracright = np.fmax(np.fmin(bigright - fullspot,0.5),-0.5)
bigleft = 0.0
bigright = 0.0
bool_mask1 = (spot >= -0.5) & (spot < (nx-0.5))
bool_mask2 = (bigy >= 0) & (bigy <= (ny-1))
weight =  (np.fmin(np.fmax(fracleft + fracright,0),1))*bool_mask1*bool_mask2
spot = 0.0
fracleft = 0.0
fracright = 0.0
bigy = np.fmin(np.fmax(bigy,0),ny-1)

if weight_image != None:
    temp = np.array([weight_image[x1,y1]*image[x1,y1] for (x1, y1) in zip(bigy.flatten(), fullspot.flatten())])
    temp2 = np.reshape(weight.flatten()*temp,(nTrace,npix,tempx))
    fextract = np.sum(temp2,axis=2)
    temp_wi = np.array([weight_image[x1,y1] for (x1, y1) in zip(bigy.flatten(), fullspot.flatten())])
    temp2_wi = np.reshape(weight.flatten()*temp_wi,(nTrace,npix,tempx))
    f_ivar = np.sum(temp2_wi,axis=2)
    fextract = fextract/(f_ivar + (f_ivar == 0))*(f_ivar > 0)
else:
    # Might be more pythonic way to code this. I needed to switch the flattening order in order to get
    # this to work
    temp = np.array([image[x1,y1] for (x1, y1) in zip(bigy.flatten(), fullspot.flatten())])
    temp2 = np.reshape(weight.flatten()*temp,(nTrace,npix,tempx))
    fextract = np.sum(temp2,axis=2)

    # IDL version model functionality not implemented yet



# Omit the model functionality right now
#if model != None:
#    model = image*0
#    modelwt = image*0


# This stuff down here is for eventually debugging long_gprofile
#SN_GAUSS = 3.0
#thisfwhm = 4.0
#nccd = 1
#hwidth = 3.*np.max(thisfwhm)+1
#MAX_TRACE_CORR = 2.0
#wvmnx = [2900., 30000]
#thisfwhm = np.fmax(thisfwhm,1.0)
# require the FWHM to be greater than 1 pixel

#xnew = trace_in
#dims = image.shape
#ncol = dims[0]
#nrow = dims[1]

#ncol =



