#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday October 28 17:08:18 2021

@author: Ada Gjermundsen

Calculates timeseries of annual, global mean variables for atmosphere and ocean. Time is given in years

Sea ice variables (area and extent) are calculated for March and September 
(Please note that time is given for March and September such that time series for March will have nans in September and vica versa)
If you use xarray for plotting, that will work just fine 
"""


import sys
# path to the piclim-noresm folder
sys.path.insert(1, '~/pyclim-NorESM/pyclim-noresm/')
from reading_routines_noresm import make_filelist_raw, read_noresm_raw
import general_util_funcs as guf
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')


if __name__ == '__main__':
    expid = 'NCO2x4frc2_f09_tn14_keyclim_snow'
    path = '/cluster/work/users/adagj/archive/'
    outdir = ''
    yrs = 11 # start year
    yre = 16 # end year
    # make atmosphere timeseries
    varlist = ['FSNT', 'FSNTC', 'FLNT','FLNTC', 'FSNS','FSNSC','FLNS','FLNSC','FLUT','FLUTC', 'FSNTOA','FSNTOAC',
                'FLDS' , 'FSDS','FSDSC', 'AODVVOLC', 'AOD_VIS', 'CAODVIS','QREFHT',
                'SWCF','LWCF','TREFHT','CLDFREE', 'CLDTOT', 'CLDHGH', 'CLDLOW', 'CLDMED', 'LHFLX', 'SHFLX', 'TS', 'U10']
    # if yrs is not given, the first year will be the first year of the simulation
    # if yre is not given, the last year will be the last year of the simulation
    fnames = make_filelist_raw(expid, path, component='atmos', yrs = yrs, yre = yre)
    # PLEASE NOTE  NorESM RAW cam files have incorrect time variable output,
    # thus it is necessary to fix the time variable (fix_cam_time) on the WHOLE DATASET before any other functions involving time can be used! 
    # If not done, the output is just WRONG! If you use CMORized data, it is not necessary, but it doesn't do any harm either
    ds = read_noresm_raw(fnames, dim='time', transform_func=lambda ds: guf.fix_cam_time(guf.consistent_naming(ds))[varlist])
    ds['RESTOM'] = ds['FSNT'] - ds['FLNT']
    ds_atm = ds.map(guf.global_avg).map(guf.yearly_avg)
    # make ocean and sea ice timeseries
    varlist = ['dp','sst','sss','mmflxd', 'mhflx', 'fice', 'temp', 'saln', 'tempga', 'salnga', 'sssga', 'sstga']
    fnames = make_filelist_raw(expid, path, component='ocean', yrs = yrs, yre = yre)
    ds = read_noresm_raw(fnames, dim='time', transform_func=lambda ds: ds[varlist].map(guf.consistent_naming))
    # This is possible, but not necessary as global avg sst and sss are output; sstga and sssga
    areaavg =ds[['sss', 'sst']].map(guf.areaavg_ocn, cmor=False).map(guf.yearly_avg)
    areaavg_masked = ds[['sss', 'sst']].map(guf.regionalavg_ocn, lat_low=-90, lat_high=-35, cmor = False).map(guf.yearly_avg)
    # NOTE! It is necessary to rename the regional avg variables so they don't overwrite the global mean variables already calculated in areaavg (e.g. sst, sss) 
    areaavg_masked = areaavg_masked.rename({'sss':'sss_90S_35S'})
    areaavg_masked = areaavg_masked.rename({'sst':'sst_90S_35S'})
    yravg = ds[['sssga', 'sstga', 'salnga','tempga']].map(guf.yearly_avg)
    amoc = guf.amoc(ds['mmflxd']).map(guf.yearly_avg)
    oht = guf.atl_hfbasin(ds['mhflx']).map(guf.yearly_avg)
    siext = guf.sea_ice_ext(ds['fice'],cmor = False)
    siarea = guf.sea_ice_area(ds['fice'], cmor = False)
    ds_ocn = xr.merge([areaavg_masked, amoc, oht, siext, siarea, yravg])#, volumeavg])
    combined = xr.merge([ds_atm, ds_ocn])
    tmp = combined.to_netcdf(outdir + expid + '.%s_%s.timeseries.nc'%(str(combined.year.values[0]).zfill(4), str(combined.year.values[-1]).zfill(4)), compute = False)
    with ProgressBar():
         result = tmp.compute()

