



import numpy as np


def extract_asymbox2(image,left,right,ycen = None,weight_image = None):
    """ Extract the total flux within a boxcar window at many positions.
    This routine will accept an asymmetric/variable window
    Traces are expected to run vertically to be consistent with other
    extract_  routines

    Parameters
    ----------
    det : int
      Detector index
    caps : bool, optional
      Return all caps?
    prefix : bool, optional
      Include the prefix?

    Returns
    -------
    dnum : str
      A string used by the settings dictionary
    """

    dim = left.shape
    ndim = len(dim)
    npix = dim[1]
    if (ndim == 1):
        nTrace = 1
    else:
        nTrace = dim[0]

    if ycen == None:
        if ndim == 1:
            ycen = np.arange(npix, dtype='int')
        elif ndim == 2:
            ycen = np.outer(np.ones(nTrace, dtype='int'), np.arange(npix, dtype='int'), )
        else:
            raise ValueError('left is not 1 or 2 dimensional')

    if np.size(left) != np.size(ycen):
        raise ValueError('Number of elements in LEFT and YCEN must be equal')

    idims = image.shape
    nx = idims[1]
    ny = idims[0]

    maxwindow = np.max(right - left)
    tempx = np.int(maxwindow + 3.0)

    bigleft = np.outer(left[:], np.ones(tempx))
    bigright = np.outer(right[:], np.ones(tempx))
    spot = np.outer(np.ones(npix * nTrace), np.arange(tempx)) + bigleft - 1
    bigy = np.outer(ycen[:], np.ones(tempx, dtype='int'))

    fullspot = np.array(np.fmin(np.fmax(np.round(spot + 1) - 1, 0), nx - 1), int)
    fracleft = np.fmax(np.fmin(fullspot - bigleft, 0.5), -0.5)
    fracright = np.fmax(np.fmin(bigright - fullspot, 0.5), -0.5)
    del bigleft
    del bigright
    bool_mask1 = (spot >= -0.5) & (spot < (nx - 0.5))
    bool_mask2 = (bigy >= 0) & (bigy <= (ny - 1))
    weight = (np.fmin(np.fmax(fracleft + fracright, 0), 1)) * bool_mask1 * bool_mask2
    del spot
    del fracleft
    del fracright
    bigy = np.fmin(np.fmax(bigy, 0), ny - 1)

    if weight_image != None:
        temp = np.array([weight_image[x1, y1] * image[x1, y1] for (x1, y1) in zip(bigy.flatten(), fullspot.flatten())])
        temp2 = np.reshape(weight.flatten() * temp, (nTrace, npix, tempx))
        fextract = np.sum(temp2, axis=2)
        temp_wi = np.array([weight_image[x1, y1] for (x1, y1) in zip(bigy.flatten(), fullspot.flatten())])
        temp2_wi = np.reshape(weight.flatten() * temp_wi, (nTrace, npix, tempx))
        f_ivar = np.sum(temp2_wi, axis=2)
        fextract = fextract / (f_ivar + (f_ivar == 0)) * (f_ivar > 0)
    else:
        # Might be more pythonic way to code this. I needed to switch the flattening order in order to get
        # this to work
        temp = np.array([image[x1, y1] for (x1, y1) in zip(bigy.flatten(), fullspot.flatten())])
        temp2 = np.reshape(weight.flatten() * temp, (nTrace, npix, tempx))
        fextract = np.sum(temp2, axis=2)

    # IDL version model functionality not implemented yet
    # At the moment I'm not reutnring the f_ivar for the weight_image mode. I'm not sure that this functionality is even
    # ever used

    return fextract
