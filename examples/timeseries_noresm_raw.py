#!/usr/bin/env python3
"""
Created on Friday October 28 17:08:18 2021

@author: Ada Gjermundsen

Calculates timeseries of annual, global mean variables for atmosphere and ocean. Time is given in years

Sea ice variables (area and extent) are calculated for March and September 
(Please note that time is given for March and September such that time series for March will have nans in September and vica versa)
If you use xarray for plotting, that works just fine 
"""
from __future__ import annotations

from dask.diagnostics import ProgressBar
from pathlib import Path

import xarray as xr

import warnings

from xarray.core.duck_array_ops import last

warnings.simplefilter("ignore")

# recommend to make symbol links to general_util_funcs.py and reading_routines_noresm.py
# e.g. ln -s ../pyclim-noresm/reading_routines_noresm.py
import general_util_funcs as guf
from reading_routines_noresm import make_filelist_raw, read_noresm_raw


def atmos_timeseries(expid: str, path: str, varlist: list, first_year: int, last_year: int) -> xr.Dataset:
    """
    This function calculates annual and global mean timeseries of variables in varlist

    Parameters
    ----------
    expid:             str, case name (name of model simulation/experiment)
    path :             str, path to the data folders. E. g. '/projects/NS9560K/noresm/cases/'
    varlist :          str, list of all variables which are contained in the timeseries
    first_year:        int, first year of data to be read
    last_year:         int, last year of data to be read
    
    Returns
    -------
    ds_atm:            xarray.Dataset with global and annual mean values of the variables in varlist
  
    PLEASE NOTE that NorESM RAW cam.h0 files have incorrect time variable output,
    thus it's necessary to fix the time variable (fix_cam_time) on the WHOLE DATASET before any other functions involving time can be used! 
    If not done, the output is just WRONG! If you use CMORized data, it is not necessary, but it doesn't do any harm either

    """
    fnames = make_filelist_raw(expid, path, component="atmos", first_year=first_year, last_year=last_year)
    years = last_year - first_year + 1
    months = 12
    assert len(fnames) == years * months, f"only got {len(fnames)} files. Expected {years*months} files."
    ds = read_noresm_raw(
        fnames,
        data_vars=varlist,
        preprocess=lambda ds: guf.fix_cam_time(guf.consistent_naming(ds))[varlist],
        parallel=False,
    )
    if "FSNT" in list(ds.keys()) and "FLNT" in list(ds.keys()):
        ds["RESTOM"] = ds["FSNT"] - ds["FLNT"]
        ds["RESTOM"].attrs["long_name"] = "Net radiative flux at top of model"
        ds["RESTOM"].attrs["units"] = "W/m2"
    ds_atm = ds.map(guf.global_avg).map(guf.yearly_avg)
    return ds_atm


def ocean_timeseries(expid: str, path: str, varlist: list, first_year: int, last_year: int) -> xr.Dataset:
    """
    Parameters
    ----------
    expid:             str, case name (name of model simulation/experiment)
    path :             str, path to the data folders. Default is '/projects/NS9560K/noresm/cases/'
    varlist :          str, list of all variables which are contained in the timeseries
    first_year:        int, first year of data to be read
    last_year:         int, last year of data to be read

    Returns
    -------
    ds_ocn:            xarray.Dataset with global and annual mean values of the variables in varlist
  
    PLEASE NOTE that NorESM RAW cam.h0 time issue is not a problem in BLOM and for the ocean component output

    """
    fnames = make_filelist_raw(expid, path, component="ocean", first_year=first_year, last_year=last_year)
    years = last_year - first_year + 1
    months = 12
    assert len(fnames) == years * months, f"only got {len(fnames)} files. Expected {years*months} files."
    ds = read_noresm_raw(
        fnames, data_vars=varlist, preprocess=lambda ds: guf.consistent_naming(ds)[varlist], parallel=False,
    )
    # This is possible, but not necessary as global avg sst and sss are output; sstga and sssga
    areaavg = ds[["sss", "sst"]].map(guf.areaavg_ocn, cmor=False).map(guf.yearly_avg)
    # Area averaged values for a selected region constrained by lat_low, lat_high, lon_low, lon_high
    areaavg_masked = (
        ds[["sss", "sst"]].map(guf.regionalavg_ocn, lat_low=-90, lat_high=-35, cmor=False).map(guf.yearly_avg)
    )
    # NOTE! It is necessary to rename the regional avg variables so they don't overwrite the global mean variables already calculated in areaavg (e.g. sst, sss)
    areaavg_masked = areaavg_masked.rename({"sss": "sss_90S_35S"})
    areaavg_masked = areaavg_masked.rename({"sst": "sst_90S_35S"})
    yravg = ds[["sssga", "sstga", "salnga", "tempga"]].map(guf.yearly_avg)
    # AMOC @ 26N, 45N, and max(20N,60N)
    amoc = guf.amoc(ds["mmflxd"]).map(guf.yearly_avg)
    # Atlantic Ocean heat transport @ 26N, 45N, and max(20N,60N)
    oht = guf.atl_hfbasin(ds["mhflx"]).map(guf.yearly_avg)
    # sea-ice extent for March and September
    siext = guf.sea_ice_ext(ds["fice"], cmor=False)
    # sea-ice extent for March and September
    siarea = guf.sea_ice_area(ds["fice"], cmor=False)
    # combine all ocean datasets
    ds_ocn = xr.merge([areaavg_masked, amoc, oht, siext, siarea, yravg])
    return ds_ocn


if __name__ == "__main__":
    expids = ["N1850_f19_tn14_20190621"]  # name of experiment
    path = "/projects/NS9560K/noresm/cases/"  # path to experiment
    outdir = "/scratch/adagj/noresm_raw/"  # path to directory where output is stored

    # if you want all years in the NorESM simulation, you don't need to set start and end year
    # but since this reading script is bloody slow it's a good idea... or drink coffee while waiting
    first_year = 1600  # start year
    last_year = 1605  # end year
    # if yrs is not given, the first year will be the first year of the simulation
    # if yre is not given, the last year will be the last year of the simulation
    # e.g. ds_atm = atmos_timeseries(expid=expid, path=path, varlist = varlist)
    # e.g. ds_ocn = ocean_timeseries(expid = expid, path = path,  varlist = varlist)

    for expid in expids:
        # ATMOSPHERE
        print(expid)
        varlist = [
            "FSNT",
            "FSNTC",
            "FLNT",
            "FLNTC",
            "FSNS",
            "FSNSC",
            "FLNS",
            "FLNSC",
            "FLUT",
            "FLUTC",
            "FSNTOA",
            "FSNTOAC",
            "FLDS",
            "FSDS",
            "FSDSC",
            "AODVVOLC",
            "AOD_VIS",
            "CAODVIS",
            "QREFHT",
            "SWCF",
            "LWCF",
            "TREFHT",
            "CLDFREE",
            "CLDTOT",
            "CLDHGH",
            "CLDLOW",
            "CLDMED",
            "LHFLX",
            "SHFLX",
            "TS",
            "U10",
        ]
        ds_atm = atmos_timeseries(expid=expid, path=path, varlist=varlist, first_year=first_year, last_year=last_year,)
        print(ds_atm)
        # OCEAN AND SEA ICE
        varlist = [
            "dp",
            "sst",
            "sss",
            "mmflxd",
            "mhflx",
            "fice",
            "temp",
            "saln",
            "tempga",
            "salnga",
            "sssga",
            "sstga",
        ]
        ds_ocn = ocean_timeseries(expid=expid, path=path, varlist=varlist, first_year=first_year, last_year=last_year,)
        print(ds_ocn)
        # combine atmosphere and ocean datasets
        combined = xr.merge([ds_atm, ds_ocn])

        # this is just an example. please change to something you find useful.
        filename = (
            outdir
            + expid
            + ".%s_%s.timeseries.nc" % (str(combined.year.values[0]).zfill(4), str(combined.year.values[-1]).zfill(4),)
        )
        tmp = combined.to_netcdf(filename, compute=False)

        # writing files with progressbar fix most memory issues
        with ProgressBar():
            result = tmp.compute()
