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
# path to the pyclim-noresm folder
sys.path.insert(1, '~/pyclim-NorESM/pyclim-noresm/')
from reading_routines_noresm import read_noresm_cmor, Modelinfo, fx_files
import general_util_funcs as guf
import regrid_functions as rf
import xarray as xr
xr.set_options(enable_cftimeindex=True)
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter('ignore')
    
if __name__ == '__main__':
    expid = 'piControl'                     # name of experiment
    activity_id = 'CMIP'                    # activity id of experiment i.e. which MIP 
    modelname = 'NorESM2-LM'                # name of model
    realiz = 'r1i1p1f1'                     # ensemble member 
    outdir =  '/scratch/adagj/noresm_raw/'  # path to directory where output is stored
    # Create model object with experiment information (not file specific) as attributes
    # for details see Modelinfo in reading_routines_noresm.py
    model = Modelinfo(name = modelname, activity_id = activity_id, expid = expid, realiz=realiz)
    # REGRID info
    grid_weight_path = outdir
    regrid_mode = 'conservative'
    seaice = False
    reuse_weights = False

    # ATMOSPHERE
    var = 'tas'
    ds = read_noresm_cmor(model, varlist=[var], realm = 'Amon', transform_func=lambda ds: guf.consistent_naming(ds))
    # regrid NorESM2-LM to NorESM2-MM grid
    # this path only works on NIRD, but outgrid can easily be made from other grids or yo can use xesmf e.g.
    # import xesmf as xe
    # xe.util.grid_global(1,1)
    area = xr.open_dataset('/projects/NS9034K/CMIP6/CMIP/NCC/NorESM2-MM/piControl/r1i1p1f1/fx/areacella/gn/latest/areacella_fx_NorESM2-MM_piControl_r1i1p1f1_gn.nc')
    outgrid = rf.make_outgrid(area)
    dr_atm = rf.regrid_file(ds, var = var, outgrid=outgrid, grid_weight_path=grid_weight_path, regrid_mode = regrid_mode, curvilinear = False, seaice = seaice, reuse_weights=reuse_weights)  
  
    # OCEAN 
    var = 'thetao'
    ds = read_noresm_cmor(model, varlist=[var], realm = 'Omon', grid = 'gr',  dim='time', transform_func=lambda ds: guf.consistent_naming(ds))
    # The fx_files function only works on NIRD because NS9034K is not mounted to BETZY or FRAM
    # If you want to regrid to the same model grid, just the atmosphere grid, this can be done by:
    # fx_files(model, 'areacella') 
    # area = xr.open_dataset(model.fxfile)
    # outgrid = rf.make_outgrid(area)
    dr_ocn = rf.regrid_file(ds, var = var, outgrid=outgrid, grid_weight_path=grid_weight_path, regrid_mode = regrid_mode, curvilinear = True, seaice = seaice, reuse_weights=reuse_weights)
    
    #SEA-ICE timeseries
    var = 'siconc'
    ds = read_noresm_cmor(model, varlist=[var], realm = 'SImon', grid = 'gn',  dim='time', transform_func=lambda ds: guf.consistent_naming(ds))   
    # use outgrid from above
    dr_ice = rf.regrid_file(ds, var = var, outgrid=outgrid, grid_weight_path=grid_weight_path, regrid_mode = 'nearest_s2d' , curvilinear = True, seaice = True, reuse_weights=reuse_weights)
   
    # combine regridded atmosphere, ocean and sea-ice datasets 
    combined = xr.merge([dr_atm, dr_ocn, dr_ice])
    
    # this is just an example. please change to a filename you find useful 
    filename = outdir + modelname + '_' + expid + '_' + realiz + '.regridded.nc' 
    tmp = combined.to_netcdf(filename, compute = False)
    with ProgressBar():
         result = tmp.compute()

