# Module to run tests on arvcorr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units

from pypeit.core import wave
from pypeit import specobjs
from pypeit.core import fsort
from pypeit.tests.tstutils import load_kast_blue_masters
from pypeit.spectrographs.util import load_spectrograph

mjd = 57783.269661
RA = '07:06:23.45'
DEC = '+30:20:50.5'
hdr_equ = 2000.
lon = 155.47833            # Longitude of the telescope (NOTE: West should correspond to positive longitudes)
lat = 19.82833             # Latitude of the telescope
alt = 4160.0               # Elevation of the telescope (in m)


@pytest.fixture
def fitstbl():
    return fsort.dummy_fitstbl()


def test_geovelocity():
    """ Test the full geomotion velocity calculation
    """
    loc = (lon * units.deg, lat * units.deg, alt * units.m,)

    radec = SkyCoord(RA, DEC, unit=(units.hourangle, units.deg), frame='icrs')
    obstime = Time(mjd, format='mjd', scale='utc', location=loc)

    corrhelio = wave.geomotion_velocity(obstime, radec, frame="heliocentric")
    corrbary = wave.geomotion_velocity(obstime, radec, frame="barycentric")

    # IDL
    # vhel = x_keckhelio(106.59770833333332, 30.34736111111111, 2000., jd=2457783.769661)
    #    vrotate = -0.25490532
    assert np.isclose(corrhelio, -12.49764005490221, rtol=1e-5)
    assert np.isclose(corrbary, -12.510015817405023, rtol=1e-5)


def test_geocorrect(fitstbl):
    """
    """
    # Spectrograph
    # (KBW) Had to change this to keck to match the telecope parameters,
    # then just changed to use definitions above directly.
#    spectrograph = load_spectrograph(spectrograph='keck_lris_blue')

    # Specobjs (wrap in a list to mimic a slit)
    sobj_list = specobjs.dummy_specobj(fitstbl, extraction=True)
    specObjs = specobjs.SpecObjs(sobj_list)
    scidx = 5
    tbname = fitstbl['date'][scidx]
    obstime = Time(tbname, format='isot')#'%Y-%m-%dT%H:%M:%S.%f')
    maskslits = np.array([False]*specObjs.nobj)

    helio, hel_corr = wave.geomotion_correct(specObjs, maskslits, fitstbl, scidx, obstime,
                                               lon, lat, alt, 'heliocentric')
    assert np.isclose(helio, -9.17461338, rtol=1e-5)  # Checked against x_keckhelio
    #assert np.isclose(helio, -9.3344957, rtol=1e-5)  # Original
    assert np.isclose(specObjs[0].boxcar['WAVE'][0].value, 3999.877589008, rtol=1e-8)

