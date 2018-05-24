

import numpy as np
from astropy.io import fits
from astropy.table import Table
from pydl.pydlutils.trace import TraceSet
from pydl.pydlutils.image import djs_maskinterp

#from pypit.idl_stats import djs_iterstat
from pypit import ginga
from pypit.extract_boxcar import extract_boxcar
from matplotlib import pyplot as plt
from astropy.stats import sigma_clipped_stats


from pydl.pydlutils.math import djs_median
from pydl.pydlutils.bspline import iterfit as bspline_iterfit

## Directory for IDL tests is /Users/joe/gprofile_develop/

path = '/Users/joe/REDUX/lris_redux/May_2008/0938+5317/Blue1200/'

# slitfile
slitfile = path + 'slits-lblue0059.fits.gz'
# science image
scifile = path + 'Science/sci-lblue0084.fits.gz'
wavefile = path + 'wave-lblue0028.fits.gz'



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
# Open  up the wavelength image
hdu_wave = fits.open(wavefile)
waveimg = hdu_wave[0].data


# Open up slitfile and read in tset_slits
tset_fits = Table.read(slitfile)
hdu_slit = fits.open(slitfile)
slitmask = hdu_slit[0].data
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
#viewer, ch = ginga.show_image(sciimg-sky_model)
#ginga.show_slits(viewer, ch, left_edge.T, righ_edge.T,np.arange(nslits+1))
#for iobj in range(nobj):
#    ginga.show_trace(viewer, ch, trace_in[iobj,:],id_str[iobj], color='green')

# parameters for extract_asymbox2
box_rad = 7
trace_in = trace_in[5:,:]
#right = trace_in[5:,:] + box_rad
image = sciimg - sky_model
ivar = sciivar*(slitmask == 3)
wave = objstruct['WAVE_BOX'][5]
flux = objstruct['FLUX_BOX'][5]
fluxivar = objstruct['IVAR_BOX'][5]
hwidth = objstruct['MASKWIDTH'][5]
thisfwhm = objstruct['FWHM'][5]

SN_GAUSS = None
MAX_TRACE_CORR = None
wvmnx = None

if SN_GAUSS == None: SN_GAUSS = 3.0
if thisfwhm == None: thisfwhm = 4.0
if hwidth == None: 3.0*(np.max(thisfwhm) + 1.0)
if MAX_TRACE_CORR == None:  MAX_TRACE_CORR = 2.0
if wvmnx == None: wvmnx = [2900., 30000]

thisfwhm = np.fmax(thisfwhm,1.0) # require the FWHM to be greater than 1 pixel

xnew = trace_in
dims = image.shape
nspat = dims[1]
nspec = dims[0]

#top = np.fmin(np.max(np.where(np.sum(ivar == 0,0) < nspec)),nspat)
#bot = np.fmax(np.min(np.where(np.sum(ivar == 0,0) < nspec)),0)
#min_column = np.fmax(np.min(trace_in - hwidth)),bot)
#max_column = long(max(trace_in + hwidth)) <  top

# create some images we will need
profile_model = np.zeros((nspec,nspat))
sub_obj = image
sub_ivar = ivar
sub_wave = waveimg
sub_trace = trace_in
sub_x = np.arange(nspat)
sn2_sub = np.zeros((nspec,nspat))
spline_sub = np.zeros((nspec,nspat))



flux_sm = djs_median(flux, width = 5, boundary = 'reflect')
fluxivar_sm =  djs_median(fluxivar, width = 5, boundary = 'reflect')
fluxivar_sm = fluxivar_sm*(fluxivar > 0.0)

indsp = (wave > wvmnx[0]) & (wave < wvmnx[1]) & \
         np.isfinite(flux_sm) & (flux_sm < 5.0e5) &  \
         (flux_sm > -1000.0) & (fluxivar_sm > 0.0)
nsp = np.sum(indsp)

# Not all the djs_reject keywords args are implemented yet
b_answer, bmask   = bspline_iterfit(wave[indsp], flux_sm[indsp], invvar = fluxivar_sm[indsp],everyn = 1.5)
b_answer, bmask2  = bspline_iterfit(wave[indsp], flux_sm[indsp], invvar = fluxivar_sm[indsp]*bmask,everyn = 1.5)
c_answer, cmask   = bspline_iterfit(wave[indsp], flux_sm[indsp], invvar = fluxivar_sm[indsp]*bmask2,everyn = 30)
spline_flux, _ = b_answer.value(wave[indsp])
cont_flux, _ = c_answer.value(wave[indsp])

sn2 = (np.fmax(spline_flux*(np.sqrt(np.fmax(fluxivar_sm[indsp], 0))*bmask2),0))**2
ind_nonzero = (sn2 > 0)
nonzero = np.sum(ind_nonzero)
if(nonzero >0):
    (mean, med_sn2, stddev) = sigma_clipped_stats(sn2)
else: med_sn2 = 0.0
sn2_med = djs_median(sn2, width = 9, boundary = 'reflect')
igood = (ivar > 0.0)
ngd = np.sum(igood)
if(ngd > 0):
    isrt = np.argsort(wave[indsp])
    sn2_sub[igood] = np.interp(sub_wave[igood],(wave[indsp])[isrt],sn2_med[isrt])
print('sqrt(med(S/N)^2) = ', np.sqrt(med_sn2))

min_wave = np.min(wave[indsp])
max_wave = np.max(wave[indsp])
spline_flux1 = np.zeros(nspec)
cont_flux1 = np.zeros(nspec)
sn2_1 = np.zeros(nspec)
ispline = (wave >= min_wave) & (wave <= max_wave)
spline_tmp, _ = b_answer.value(wave[ispline])
spline_flux1[ispline] = spline_tmp
cont_tmp, _ = c_answer.value(wave[ispline])
cont_flux1[ispline] = cont_tmp
sn2_1[ispline] = np.interp(wave[ispline], (wave[indsp])[isrt], sn2[isrt])
bmask = np.zeros(nspec,dtype='bool')
bmask[indsp] = bmask2
spline_flux1 = djs_maskinterp(spline_flux1,(bmask == False))
cmask2 = np.zeros(nspec,dtype='bool')
cmask2[indsp] = cmask
cont_flux1 = djs_maskinterp(cont_flux1,(cmask2 == False))

if(med_sn2 <= 2.0):
    (_, _, sigma1) = sigma_clipped_stats(flux[indsp])
    spline_sub[igood]= np.fmax(sigma1,0)
else:
    if((med_sn2 <=5.0) and (med_sn2 > 2.0)):
        spline_flux1 = cont_flux1
    # Interp over points <= 0 in boxcar flux or masked points using cont model
    badpix = (spline_flux1 <= 0.5) | (bmask == False)
    goodval = (cont_flux1 > 0.0) & (cont_flux1 < 5e5)
    indbad1 = badpix & goodval
    nbad1 = np.sum(indbad1)
    if(nbad1 > 0):
        spline_flux1[indbad1] = cont_flux1[indbad1]
    indbad2 = badpix & ~goodval
    nbad2 = np.sum(indbad2)
    ngood0 = np.sum(~badpix)
    if((nbad2 > 0) or (ngood0 > 0)):
        spline_flux1[indbad2] = djs_median(spline_flux1[~badpix])
    # take a 5-pixel median to filter out some hot pixels
    spline_flux1 = djs_median(spline_flux1,width=5,boundary ='reflect')
    # Create the normalized object image
    if(ngd > 0):
        isrt = np.argsort(wave)
        spline_sub[igood] = np.interp(sub_wave[igood],wave[isrt],spline_flux1[isrt])


#flux  = extract_boxcar(image,trace,box_rad)





# Omit the model functionality right now
#if model != None:
#    model = image*0
#    modelwt = image*0



#ncol =



