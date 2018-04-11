# Module to run tests on arvcorr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from pypit import pyputils
msgs = pyputils.get_dummy_logger()

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

from pypit import arwave as py_arwave
from pypit import arutils
from pypit import arparse as settings

mjd = 57783.269661
RA = '07:06:23.45'
DEC = '+30:20:50.5'
hdr_equ = 2000.
lon = 155.47833            # Longitude of the telescope (NOTE: West should correspond to positive longitudes)
lat = 19.82833             # Latitude of the telescope
alt = 4160.0               # Elevation of the telescope (in m)

@pytest.fixture
def fitsdict():
    return arutils.dummy_fitsdict()


def test_geovelocity():
    """ Test the full geomotion velocity calculation
    """
    loc = (lon * u.deg, lat * u.deg, alt * u.m,)

    radec = SkyCoord(RA, DEC, unit=(u.hourangle, u.deg), frame='icrs')
    obstime = Time(mjd, format='mjd', scale='utc', location=loc)

    corrhelio = py_arwave.geomotion_velocity(obstime, radec, frame="heliocentric")
    corrbary = py_arwave.geomotion_velocity(obstime, radec, frame="barycentric")

    # IDL
    # vhel = x_keckhelio(106.59770833333332, 30.34736111111111, 2000., jd=2457783.769661)
    #    vrotate = -0.25490532
    assert np.isclose(corrhelio, -12.49764005490221, rtol=1e-5)
    assert np.isclose(corrbary, -12.510015817405023, rtol=1e-5)


def test_geocorrect(fitsdict):
    """
    """
    # Initialize some settings
    arutils.dummy_settings(spectrograph='shane_kast_blue')#, set_idx=False)
    # Load Dummy self
    slf = arutils.dummy_self(fitsdict=fitsdict)
    # Specobjs
    specobjs = arutils.dummy_specobj(fitsdict, extraction=True)
    slf._specobjs[0] = [specobjs]
    # Run
    # vhel = x_keckhelio(106.59770833333332, 30.34736111111111, 2000., jd=2457046.5036, OBS='lick')  9.3166 km/s
    helio, hel_corr = py_arwave.geomotion_correct(slf, 1, fitsdict)
    assert np.isclose(helio, -9.3350877, rtol=1e-5)  # Checked against x_keckhelio
    #assert np.isclose(helio, -9.3344957, rtol=1e-5)  # Original
    assert np.isclose(slf._specobjs[0][0][0].boxcar['wave'][0].value, 3999.8754558341816, rtol=1e-8)
