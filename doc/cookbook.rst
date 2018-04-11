.. highlight:: rest

==============
PYPIT Cookbook
==============

This document gives an overview on
how to run PYPIT, i.e. minimal detail is provided.
Notes on :doc:`installing` are found elsewhere.

The following outlines the standard steps for running
PYPIT on a batch of data.  There are alternate ways to
run these steps, but non-experts should adhere to the
following approach.

Outline
+++++++

Here is the basic outline of the work flow.  The
following is for one instrument in one working directory.

1. Prepare your data

  - Identify folder(s) with raw images
  - The raw images can be gzip compressed although the Python FITS reader works much more slowly on gzipped files

2. Run the :ref:`pypit_setup` script one of the PYPIT-approved :doc:`instruments` (e.g. lris_blue, kast_red)

  - Generates a setup_files/ folder that holds a series of files
  - Generates the instrument PYPIT reduction file [not used further]
  - Generates the instrument .setups file (:doc:`setup`)
  - Generates a .sorted file which lists files sorted by setup
  - Scan the output WARNING messages which will indicate insufficient calibration files (e.g. missing arc frames)

3. Inspect the :ref:`setups-file` and :ref:`sorted-file` to confirm the instrument configurations
  - Modify the data and spect blocks in the instrument PYPIT file, as needed (see :ref:`setup-modifications`)

    - If changes were made, rerun :ref:`pypit_setup`

  - Now run :ref:`pypit_setup` with the --custom option to generates custom PYPIT reduction files and folders for each setup
  - This produces one folder per setup and a related custom :doc:`pypit_file`.

4. Prepare the custom :doc:`pypit_file` for reducing a given setup
  - Enter one of the setup folders (e.g. lris_blue_setup_A)
  - Modify the custom :doc:`pypit_file` as needed

    - e.g. trim/add calibration files, edit frametypes

5. Run :ref:`run-calcheck` on the custom PYPIT file(s) (described in :doc:`calcheck`)

  - Modify the spect block in the PYPIT file to specify calibrations
  - Inspect the .calibs file for your PYPIT file.
  - Confirm calibration, science and standard frames
  - Further customize your PYPIT file, as needed

6. Run the reduction (described in :doc:`running`)

  - :ref:`run-pypit` PYPIT_file
  - Hope for the best...  :)

7. Examine QA (:doc:`qa`)

  - When an exposure is fully reduced, a QA file (PDF) is generated in the QA folder
  - Examine the output as described in the :doc:`qa` documentation

8. Examine spectra
  - Examine the extracted 1D spectra with :ref:`pypit-1dspec`
  - Examine the extracted 2D spectra with :ref:`pypit-2dspec`

9. Coadd (see :doc:`coadding`)

10. Repeat steps 4-8 for additional setups, as desired



Old
+++

- In principle if the calibration and the science exposures are consistent you can pretty much prepare your *.pypit
file following one of the examples at: https://github.com/PYPIT/PYPIT/tree/master/test_suite

- In the case you have a different configuration in some of your calibration images, PYPIT might not run. You can
try to match the headers manually (copy and paste from your science image parts of the header such as ```SLITNAME```,
```SLITMASK```, etc). The results might not be optimal, it depends on which kind of differences you have
between the calibration images and the science images.

3. Prepare the *pypit file

- Before running you should prepare your *pypit file. As an example for LRISr:

```
run ncpus 1
run spectrograph lris_red
output verbosity 2
output overwrite True
output sorted lris_red_qpair

# Reduce
arc calibrate IDpixels 131.299,400.73,474.20,499.6,946.787,1423.99,2246.698,2723.903
arc calibrate IDwaves 5462.27,5771.21,5854.1101,5883.5252,6404.018,6967.35,7950.36,8523.78
trace dispersion direction 0
trace slits tilts params [1,1,1]
trace slits tilts method spca
reduce skysub method bspline
pixelflat combine method median
pixelflat combine reject level [10.0,10.0]
pixelflat norm recnorm False
bias useframe bias

spect read
  fits calwin 12.
  bias number 1
  arc number 1
  trace number 1
  pixelflat number 1
  standard number 1
  set bias your_bias_or_dark.fits
  set pixelflat your_flat.fits
  set trace your_flat.fits
  set arc your_arc.fits
  set standard your_standard.fits
spect end


# Read in the data
data read
 $PATH_TO_YOUR_DATA/*.fits
data end
```
The first section on the file includes the number of cpus used to run PYPIT, verbosity and some other running options.
The second section has details about the reduction and the calibration. In principle, for the LRIS red side you might have
to identify the lines manually and include them properly (try commenting those lines first and see if the code can identify
the lines automatically). It is possible that the program does not recognize some lines that you might input. In that case,
just try to choose another line or make sure that you got the correct edge of the line. For the LRIS blue side you can comment
the line identification part on the *pypit file since the code recognizes them automatically. The rest of paramenters can be
left as these default values. ```bias useframe bias``` can be accompanied by ```bias```, ```dark``` or ```overscan```. Even if you
are using dark exposures, ```bias``` will probably work (and better).

The ```data read... data end``` block includes the path (you should include the full path!) of your images. In principle, PYPIT
can recognize automatically the type of exposure and proceed from there automatically without further information. However, sometimes
PYPIT is not able to identify the files so you must use the block ```spect read...spect end```.

In the ```spect read... spect end``` block you can specify the type of image (standard, bias, dark, pixelflat, trace, science).

4. You are ready to ```run_pypit yourfile.pypit```

5. PYPIT will create a ```Science``` folder with your 1D and 2D spectra. A ```Plots``` folder with QA plots and a ```MasterFrame``` folder with
some calibration images. More details on the output can be found at: https://github.com/PYPIT/PYPIT/blob/master/doc/outputs.rst 
