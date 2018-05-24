#!/usr/bin/env python

"""
Script for fluxing PYPIT 1d spectra
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

# pypit_flux_spec sensfunc --std_file=spec1d_Feige66_KASTb_2015May20T041246.96.fits  --instr=shane_kast_blue --sensfunc_file=tmp.yaml
# pypit_flux_spec flux --sci_file=spec1d_J1217p3905_KASTb_2015May20T045733.56.fits --sensfunc_file=tmp.yaml --flux_file=tmp.fits


def parser(options=None):
    parser = argparse.ArgumentParser(description='Parse')
    parser.add_argument("steps", type=str, help="Steps to perform [sensfunc,flux]")
    parser.add_argument("--std_file", type=str, help="File containing the standard 1d spectrum")
    parser.add_argument("--std_obj", type=str, help="Standard star identifier, e.g. O479-S5009-D01-I0023")
    parser.add_argument("--sci_file", type=str, help="File containing the science 1d spectra")
    parser.add_argument("--instr", type=str, help="Instrument name (required to generate sensfunc)")
    parser.add_argument("--sensfunc_file", type=str, help="File containing the sensitivity function (input or output)")
    parser.add_argument("--flux_file", type=str, help="Output filename for fluxed science spectra")
    parser.add_argument("--plot", default=False, action="store_true", help="Show the sensitivity function?")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args, unit_test=False):
    """ Runs fluxing steps
    """
    import pdb

    from pypit import fluxspec

    # Parse the steps
    steps = args.steps.split(',')

    # Checks
    if 'sensfunc' in steps:
        if args.instr is None:
            raise IOError("You must set the instrument to generate the sensfunc")
        if args.std_file is None:
            raise IOError("You must input a spec1d file of the standard to generate the sensfunc")
        if args.sensfunc_file is None:
            raise IOError("You must give the output filename in --sensfunc_to generate the sensfunc")
    if 'flux' in steps:
        if args.sci_file is None:
            raise IOError("You must input a spec1d file of the science spectra to flux them")
        if args.flux_file is None:
            raise IOError("You must give the output filename in --flux_file for the fluxed spectra")
        if 'sensfunc' not in steps:
            if args.sensfunc_file is None:
                raise IOError("You must input a sensfunc in -sensfunc_file to flux.")

    # Instantiate
    FxSpec = fluxspec.FluxSpec(std_spec1d_file=args.std_file,
                               sci_spec1d_file=args.sci_file,
                               spectrograph=args.instr,
                               sens_file=args.sensfunc_file)
    # Step through
    if 'sensfunc' in steps:
        # Find the star automatically?
        if args.std_obj is None:
            _ = FxSpec.find_standard()
        else:
            _ = FxSpec._set_std_obj(args.std_obj)
        # Sensitivity
        _ = FxSpec.generate_sensfunc()
        # Output
        _ = FxSpec.save_master(outfile=args.sensfunc_file)
        # Show
        if args.plot:
            FxSpec.show_sensfunc()

    if 'flux' in steps:
        FxSpec.flux_science()
        FxSpec.write_science(args.flux_file)


