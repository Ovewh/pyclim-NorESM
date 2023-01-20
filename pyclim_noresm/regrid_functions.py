#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used for regridding 

@author: Ada Gjermundsen 

Created on Wednesday November 3 2021
"""
#import sys
# path to the pyclim-noresm folder
# sys.path.insert(1, '~/pyclim-NorESM/pyclim-noresm/')
import general_util_funcs as guf
import xarray as xr
import xesmf as xe
import numpy as np
import warnings
warnings.simplefilter('ignore')

def noresm_bounds_seaice(grid):
    '''From Aleksi's great OMIP functions
       Return lon,lat bounds based on corner information clon,clat in grid.nc
       E.g  (this path only works on FRAM and BETZY)
       grid = xr.open_dataset('/cluster/shared/noresm/inputdata/ocn/blom/grid/grid_tnx1v4_20170622.nc')
    '''
    # NorESM grid
    mask_NorESM = grid.pmask.astype(float).rename('mask_NorESM')
    #
    ny,nx = mask_NorESM.shape
    lat_noresm = grid.plat.rename('lat').isel(y=slice(0,ny-1))
    lon_noresm = grid.plon.rename('lon').isel(y=slice(0,ny-1))
    #
    lon_b_noresm = xr.concat([grid.pclon.isel(nv=0),grid.pclon.isel(nv=1,x=-1)],dim='x')
    lon_b_noresm = xr.concat([lon_b_noresm,xr.concat([grid.pclon.isel(nv=3,y=-1),grid.pclon.isel(nv=2,y=-1,x=-1)],dim='x')],dim='y')
    lat_b_noresm= xr.concat([grid.pclat.isel(nv=0),grid.pclat.isel(nv=1,x=-1)],dim='x')
    lat_b_noresm = xr.concat([lat_b_noresm,xr.concat([grid.pclat.isel(nv=3,y=-1),grid.pclat.isel(nv=2,y=-1,x=-1)],dim='x')],dim='y')
    #
    lon_b_noresm = lon_b_noresm.isel(y=slice(0,ny)).rename('lon_b').rename({'y':'y_b','x':'x_b'})
    lat_b_noresm = lat_b_noresm.isel(y=slice(0,ny)).rename('lat_b').rename({'y':'y_b','x':'x_b'})
    #
    return xr.merge([lon_noresm,lat_noresm,lon_b_noresm,lat_b_noresm])
    

def make_bounds_ocean(ds, seaice=False):
    '''
    This function calculates latitude and longitude values and boundaries used for regridding
    from curvilinear grids to regular lat/lon grid (rectilinear grid)
    The Dataset generated is used as ds_in in the regridder function

    The ocean/sea-ice grid of NorESM2 is a tripolar grid with 360 and 384 unique grid cells in i- and j-direction, respectively. 
    Due to the way variables are staggered in the ocean model, an additional j-row is required explaining the 385 grid cells in the j-direction for the ocean grid. 
    The row with j=385 is a duplicate of the row with j=384, but with reverse i-index.
    Ocean variables are on i=360, j=385 grid
    Sea-ice variables are on i=360, j=384 grid

    Parameters
    ----------
    ds : xarray.DataSet, with model grid information for the data which need to be regridded
                         on (i,j,vertices) format where the 4 vertices give grid corner information

    Returns
    -------
    ds_in :  xarray.DataSet, with 2D model grid information for the data which need to be regridded
    '''

    ny,nx = ds.lat.shape # ny will be 384 for sea-ice variables
    if seaice:
        # There are problems with longitudes and latitudes in many of the cmorized sea-ice variables, so the
        # sea-ice dataset is not useful for regridding, e.g. all latitude values = 1 in the SH and fill values are incorrect
        # if sea ice variables are regridded and areacello or the ocean grid is used for setting the grid
        # drop the last row with j=385 of area when dealing with the sea ice variables.
        try:
            # This path only works on NIRD. NS9034 is not mounted in /trd-projects* so impossible to add a general path
            ds = xr.open_dataset('/projects/NS9034K/CMIP6/CMIP/NCC/NorESM2-MM/piControl/r1i1p1f1/Ofx/areacello/gn/latest/areacello_Ofx_NorESM2-MM_piControl_r1i1p1f1_gn.nc')
            ds = guf.consistent_naming(ds)
            ds = ds.isel(j = slice(0, ny))
        except:
            # This path only works on FRAM and BETZY
            grid = xr.open_dataset('/cluster/shared/noresm/inputdata/ocn/blom/grid/grid_tnx1v4_20170622.nc')
            return noresm_bounds_seaice(grid)
        # if sea ice variables are regridded and areacello or the ocean grid is used for setting the grid
        # ny = ny - 1, but if the sea ice grid is used, then ny = ny
        # drop the last row with j=385 of area when dealing with the sea ice variables.
    if 'lat' in ds.lon.coords:
        lat_model = ds.lat.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
        lon_model = ds.lon.isel(j=slice(0,ny)).rename({'i':'x','j':'y'}).drop('lon').drop('lat')
    else:
        lat_model = ds.lat.isel(j=slice(0,ny)).rename({'i':'x','j':'y'})
        lon_model = ds.lon.isel(j=slice(0,ny)).rename({'i':'x','j':'y'})
    lon_b_model = xr.concat([ds.vertices_longitude.isel(vertices=0), ds.vertices_longitude.isel(vertices=1,i=-1)],dim='i')
    lon_b_model = xr.concat([lon_b_model,xr.concat([ds.vertices_longitude.isel(vertices=3,j=-1), ds.vertices_longitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    lat_b_model = xr.concat([ds.vertices_latitude.isel(vertices=0), ds.vertices_latitude.isel(vertices=1,i=-1)],dim='i')
    lat_b_model = xr.concat([lat_b_model,xr.concat([ds.vertices_latitude.isel(vertices=3,j=-1), ds.vertices_latitude.isel(vertices=2,j=-1,i=-1)],dim='i')],dim='j')
    if 'lat' in lon_b_model.coords:
        lon_b_model = lon_b_model.isel(j=slice(0,ny+1)).rename('lon_b').rename({'j':'y_b','i':'x_b'}).drop('lon').drop('lat')
        lat_b_model = lat_b_model.isel(j=slice(0,ny+1)).rename('lat_b').rename({'j':'y_b','i':'x_b'}).drop('lon').drop('lat')
    else:
        lon_b_model = lon_b_model.isel(j=slice(0,ny+1)).rename('lon_b').rename({'j':'y_b','i':'x_b'})
        lat_b_model = lat_b_model.isel(j=slice(0,ny+1)).rename('lat_b').rename({'j':'y_b','i':'x_b'})
    lat_b_model = lat_b_model.where(lat_b_model<90.0,90.0)
    lat_b_model = lat_b_model.where(lat_b_model>-90.0,-90.0)
    return xr.merge([lon_model,lat_model,lon_b_model,lat_b_model])

def make_bounds_atmos(ds):
    '''
    This function calculates latitude and longitude values and boundaries used for regridding variables 
    from lat/lon grids to lat/lon grids (rectilinear grids)
    The Dataset generated is used as ds_in in the regridder function

    Parameters
    ----------
    ds : xarray.DataSet, with model grid information for the data which need to be regridded

    Returns
    -------
    ds_in :  xarray.DataSet, with 2D model grid information for the data which need to be regridded
    '''

    lon = ds.lon.rename({'lon':'x'})
    lat = ds.lat.rename({'lat':'y'})
    if 'lat_bnds' not in ds.variables or 'lon_bnds' not in ds.variables:
        ds = make_latlon_bounds(ds)
    lat_b = xr.concat([ds.lat_bnds.isel(bnds=0),ds.lat_bnds.isel(lat=-1).isel(bnds=1)],dim='lat').rename('lat_b').rename({'lat':'y_b'})
    lon_b = xr.concat([ds.lon_bnds.isel(bnds=0),ds.lon_bnds.isel(lon=-1).isel(bnds=1)],dim='lon').rename('lon_b').rename({'lon':'x_b'})
    if 'bnds' in lat_b.coords:
        lat_b = lat_b.drop('bnds')
        lon_b = lon_b.drop('bnds')
    if 'time' in lat_b.coords:
        lat_b = lat_b.isel(time=0).drop('time')
        lon_b = lon_b.isel(time=0).drop('time')
    return xr.merge([lon,lat,lon_b,lat_b])
   


def make_latlon_bounds(ds):
    '''
    This function creates latitude and longitude boundaries from lat and lon

    Parameters
    ----------
    ds : xarray.DataSet, with longitude (lon) and latitude (lat) information

    Returns
    -------
    ds:  xarray.DataSet, returns the same xarray.Dataset with lon/lat bounds added
    '''
    lon_b = np.concatenate((np.array([ds.lon[0].values - 0.5*ds.lon.diff(dim='lon').values[0]]),
                            0.5*ds.lon.diff(dim='lon').values + ds.lon.values[:-1],
                            np.array([ds.lon[-1].values + 0.5*ds.lon.diff(dim='lon').values[-1]])))
    lon_b = np.reshape(np.concatenate([lon_b[:-1],lon_b[1:]]),[2,len(ds.lon.values)]).T
    lon_b = xr.DataArray(lon_b, dims=('lon','bnds'),coords={'lon':ds.lon})
    lon_b.attrs['units']='degrees_east'
    lon_b.attrs['axis']='X'
    lon_b.attrs['bounds']='lon_bnds'
    lon_b.attrs['standard_name']='longitude_bounds'
    lon_b.attrs['long_name']='Longitude bounds'
    ds['lon_bnds'] = lon_b
    lat_b = np.concatenate((np.array([-90], float), 0.5*ds.lat.diff(dim='lat').values + ds.lat.values[:-1], np.array([90], float)))
    lat_b = np.reshape(np.concatenate([lat_b[:-1],lat_b[1:]]),[2,len(ds.lat.values)]).T
    lat_b = xr.DataArray(lat_b, dims=('lat','bnds'),coords={'lat':ds.lat})
    lat_b = lat_b.where(lat_b<90.0,90.0)
    lat_b = lat_b.where(lat_b>-90.0,-90.0)
    lat_b.attrs['long_name']='Latitude bounds'
    lat_b.attrs['units']='degrees_north'
    lat_b.attrs['axis']='Y'
    lat_b.attrs['bounds']='lat_bnds'
    lat_b.attrs['standard_name']='latitude_bounds'
    ds['lat_bnds'] = lat_b
    return ds

def make_outgrid(ds):
    '''Using xesmf output grids can easily be generated by the use of e.g xe.util.grid_2D, xe.util.grid_global
    But if a specific grid is preferable, it can be generated by the use of this function 
    
    Parameters
    ----------
    ds :      xarray.Dataset with grid information on 1D lat/lon grid

    Returns
    -------
    outgrid : xarray.DataSet with output grid information to be used for regridding files
    '''
    lon = ds.lon.rename({'lon':'x'})
    lat = ds.lat.rename({'lat':'y'})
    if 'lon_bnds' not in ds.variables or 'lat_bnds' not in ds.variables:
        ds = make_latlon_bounds(ds)
    lat_b = xr.concat([ds.lat_bnds.isel(bnds=0),ds.lat_bnds.isel(lat=-1).isel(bnds=1)],dim='lat').rename('lat_b').rename({'lat':'y_b'})
    lon_b = xr.concat([ds.lon_bnds.isel(bnds=0),ds.lon_bnds.isel(lon=-1).isel(bnds=1)],dim='lon').rename('lon_b').rename({'lon':'x_b'})
    outgrid = xr.merge([lon,lat,lon_b,lat_b])
    return outgrid


def make_regridder(ds, outgrid, grid_weight_path, regrid_mode = 'conservative', curvilinear = True, seaice = False, reuse_weights=False):
    ''' The first step of the regridding routine!
    There is an important reason why the regridding is broken into two steps
    (making the regridder and perform regridding). For high-resolution grids, 
    making the regridder (i.e. “computing regridding weights”) is quite computationally expensive, 
    but performing regridding on data (“applying regridding weights”) is still pretty fast.
    
    Parameters
    ----------
    ds :                 xarray.DataSet, with model grid information for the data which need to be regridded 
    outgrid :            xarray.DataSet, with output grid information which the data will be regridded to
    regrid_mode : str,   'bilinear', 'conservative', 'patch', 'nearest_s2d', 'nearest_d2s'
    grid_weight_path :   str, path to where the  regridder weight file will be stored
    curvilinear :         bool, True for ocean and se-ice variables. False for atmosphere
    seaice :             bool, set True for sea-ice variables and False for ocean variables (different grid size)
    reuse_weights :      bool, set to True to read existing weights from disk.
                               set to False if new weights need to be calculated
    Returns
    -------
    regridder : xarray.DataSet with regridder weight file information
    '''
    if curvilinear:
        ds_in = make_bounds_ocean(ds, seaice)
    else:
        ds_in = make_bounds_atmos(ds)
    regridder = xe.Regridder(ds_in, outgrid, regrid_mode,
                             filename=grid_weight_path + regrid_mode +'.nc',
                             reuse_weights=reuse_weights)#, ignore_degenerate=True)
    return regridder



def regrid_file(ds,var, outgrid, grid_weight_path, regrid_mode = 'conservative', curvilinear = True, seaice = False, reuse_weights=False):
    '''Second step of the regridding routine!
    
    Parameters
    ----------
    ds :                 xarray.DataSet, with model grid information for the data which need to be regridded 
    var :                str, name of varible
    outgrid :            xarray.DataSet, with output grid information which tif 'plev' in ds[var].dims:
                glb_mean = guf.global_avg(ds[var].isel(plev=-1).isel(time=0))ted
                           
    Returns
    -------
    dr : xarray.Dataset with regridded variable data and lon from 0-360
    '''
    regridder = make_regridder(ds, outgrid, grid_weight_path, regrid_mode, curvilinear, seaice, reuse_weights)
    dr = regridder(ds[var]) # needs DataArray
    dr = dr.rename({'x':'lon','y':'lat'})
    dr = dr.to_dataset(name = var) 
    if 'long_name'  in ds[var].attrs:
        dr[var].attrs['long_name']=ds[var].long_name
    if 'units'  in ds[var].attrs:
        dr[var].attrs['units']=ds[var].units
    if 'standard_name'  in ds[var].attrs:
        dr[var].attrs['standard_name']=ds[var].standard_name
    print('Regridding completed')
    # Print out gloabl mean values before and after regridding to see if the result makes sense 
    # may run into trouble here if there are e.g. other dimensions 
    # Also, this really slows things down as the data needs to be loaded into memory, so may be more efficient to skip this one.. I like it though
    try:
        if curvilinear:
            if 'lev' in ds[var].dims:
                glb_mean = guf.areaavg_ocn(ds[var].isel(lev=0).isel(time = 0)) 
            else:
                glb_mean = guf.areaavg_ocn(ds[var].isel(time = 0))
        else:
            if 'plev' in ds[var].dims:
                glb_mean = guf.global_avg(ds[var].isel(plev=-1).isel(time=0))
            elif 'lev' in ds[var].dims:
                glb_mean = guf.global_avg(ds[var].isel(lev=0).isel(time=0))
            else:
                glb_mean = guf.global_avg(ds[var].isel(time=0))
        if 'units'  in ds[var].attrs:
             print('%s : global mean value BEFORE regridding, first timestep:%f %s'%(var, np.round(glb_mean.values,4), ds[var].units))
        else:
             print('%s : global mean value BEFORE regridding, first timestep:%f'%(var, np.round(glb_mean.values,4)))
        if 'plev' in dr.dims:
            glb_mean = guf.global_avg(dr[var].isel(plev=-1).isel(time=0))
        elif 'lev' in dr.dims:
            glb_mean = guf.global_avg(dr[var].isel(lev=0).isel(time=0))
        else:
            glb_mean = guf.global_avg(dr[var].isel(time=0))
        if 'units'  in dr[var].attrs:
            print('%s : global mean value AFTER regridding, first timestep:%f %s'%(var, np.round(glb_mean.values,4),dr[var].units))
        else:
            print('%s : global mean value AFTER regridding, first timestep:%f'%(var, np.round(glb_mean.values,4)))
    except:
        print('Routine for writing global mean values before and after regridding failed in the regrid_file function in regrid_functions.py!\n Maybe some dimensions are not accounted for? Please check!')
    return dr


