#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:30:24 2021

@author: Ada Gjermundsen

General python functions for analyzing NorESM data 

"""
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter('ignore')

def global_mean(ds):
    '''Calculates globally averaged values

    Parameters
    ----------
    ds : xarray.DaraArray i.e. ds[var]

    Returns
    -------
    ds_out :  xarray.DaraArray with globally averaged values
    '''
    # to include functionality for subsets or regional averages:
    if 'time' in ds.dims:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat))*ds.notnull().mean(dim=('lon','time'))
    else:
        weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(ds.lat))*ds.notnull().mean(dim=('lon'))
    ds_out = (ds.mean(dim='lon')*weights).sum(dim='lat')/weights.sum()
    if 'long_name'  in ds.attrs:
        ds_out.attrs['long_name']= 'Globally averaged ' + ds.long_name
    if 'units'  in ds.attrs:
        ds_out.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_out.attrs['standard_name']=ds.standard_name
    return ds_out

def global_sum(ds,da):
    '''Calculates globally averaged values

    Parameters
    ----------
    ds : xarray.DaraSet i.e. ds[var]

    Returns
    -------
    ds_out :  xarray.DaraArray with globally averaged values
    '''
    area = define_area(ds)
    ds_out = (da*area).sum(dim=('lat','lon'),skipna=True)
    if 'long_name'  in da.attrs:
        ds_out.attrs['long_name']= 'Global sum ' + da.long_name
    if 'units'  in da.attrs:
        ds_out.attrs['units']=da.units
    if 'standard_name'  in da.attrs:
        ds_out.attrs['standard_name']=da.standard_name
    return ds_out

def yearly_avg(ds):
    ''' Calulates timeseries over yearly averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.

    Parameters
    ----------
    ds : xarray.DaraArray i.e. ds[var]

    Returns
    -------
    ds_weighted : xarray.DaraArray with yearly averaged values
    '''
    month_length = ds.time.dt.days_in_month
    weights = month_length.groupby('time.year') / month_length.groupby('time.year').sum()
    # Test that the sum of the weights for each year is 1.0
    np.testing.assert_allclose(weights.groupby('time.year').sum().values,
                               np.ones(len(np.unique(ds.time.dt.year))))
    # Calculate the weighted average:
    ds_weighted = (ds * weights).groupby('time.year').sum(dim='time')
    if 'long_name'  in ds.attrs:
        ds_weighted.attrs['long_name']= 'Annual mean ' + ds.long_name
    if 'units' in ds.attrs:
        ds_weighted.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_weighted.attrs['standard_name']=ds.standard_name

    return ds_weighted

def seasonal_avg_timeseries(ds, var=''):
    '''Calulates timeseries over seasonal averages from timeseries of monthly means
    The weighted average considers that each month has a different number of days.
    Using 'QS-DEC' frequency will split the data into consecutive three-month periods, 
    anchored at December 1st. 
    I.e. the first value will contain only the avg value over January and February 
    and the last value only the December monthly averaged value
    
    Parameters
    ----------
    ds : xarray.DaraArray i.e.  ds[var]
        
    Returns
    -------
    ds_out: xarray.DataSet with 4 timeseries (one for each season DJF, MAM, JJA, SON)
            note that if you want to include the output in an other dataset, e.g. dr,
            you should use xr.merge(), e.g.
            dr = xr.merge([dr, seasonal_avg_timeseries(dr[var], var)])
    '''
    month_length = ds.time.dt.days_in_month
    sesavg = ((ds * month_length).resample(time='QS-DEC').sum() /
              month_length.where(ds.notnull()).resample(time='QS-DEC').sum())
    djf = sesavg[0::4].to_dataset(name = var + '_DJF').rename({'time':'time_DJF'})
    mam = sesavg[1::4].to_dataset(name = var +'_MAM').rename({'time':'time_MAM'})
    jja = sesavg[2::4].to_dataset(name = var +'_JJA').rename({'time':'time_JJA'})
    son = sesavg[3::4].to_dataset(name = var +'_SON').rename({'time':'time_SON'})
    ds_out = xr.merge([djf, mam, jja, son])
    ds_out.attrs['long_name']= 'Seasonal mean ' + ds.long_name
    ds_out.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_out.attrs['standard_name']=ds.standard_name
    return ds_out

def seasonal_avg(ds):
    '''Calculates seasonal averages from timeseries of monthly means
    The time dimension is reduced to 4 seasons: 
        * season   (season) object 'DJF' 'JJA' 'MAM' 'SON'
    The weighted average considers that each month has a different number of days.
    
    Parameters
    ----------
    ds : xarray.DaraArray i.e.  ds[var]
        
    Returns
    -------
    ds_weighted : xarray.DaraArray 
    '''
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'.
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()
    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))
    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.season').sum(dim='time')
    ds_weighted.attrs['long_name']= 'Seasonal mean ' + ds.long_name
    ds_weighted.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_weighted.attrs['standard_name']=ds.standard_name
    return ds_weighted


def mask_region_latlon(ds, lat_low=-90, lat_high=90, lon_low=0, lon_high=360):
    '''Subtract data from a confined region
    Note, for the atmosphere the longitude values go from 0 -> 360.
    Also after regridding
    This is not the case for ice and ocean variables for which some cmip6 models
    use -180 -> 180
    
    Parameters
    ----------
    ds : xarray.DataArray or xarray.DataSet
    lat_low : int or float, lower latitude boudary. The default is -90.
    lat_high : int or float, lower latitude boudary. The default is 90.
    lon_low :  int or float, East boudary. The default is 0.
    lon_high : int or float, West boudary. The default is 360.
    
    Returns
    -------
    ds_out : xarray.DataArray or xarray.DataSet with data only for the selected region
    
    Then it is still possible to use other functions e.g. global_mean(ds) to 
    get an averaged value for the confined region 
    '''
    ds_out = ds.where((ds.lat>=lat_low) & (ds.lat<=lat_high))
    if lon_high>lon_low:
        ds_out = ds_out.where((ds_out.lon>=lon_low) & (ds_out.lon<=lon_high))
    else:
        boole = (ds_out.lon.values <= lon_high) | (ds_out.lon.values >= lon_low)
        ds_out = ds_out.sel(lon=ds_out.lon.values[boole])
    ds_out.attrs['long_name']= 'Regional subset (%i,%i,%i,%i) of '%(lat_low_lat_high,lon_low,lon_high) + ds.long_name 
    ds_out.attrs['units']=ds.units
    if 'standard_name'  in ds.attrs:
        ds_out.attrs['standard_name']=ds.standard_name
    return ds_out

def sea_ice_ext(ds, pweight):
    ''' 
    Calculates the sea ice extent from the sea ice concentration fice in BLOM
    Sea ice concentration (fice) is the percent areal coverage of ice within the ocean grid cell. 
    Sea ice extent is the integral sum of the areas of all grid cells with at least 15% ice concentration.
    Sea ice area is the integral sum of the product of ice concentration and area of all grid cells with at least 15% ice concentration. See sea_ice_area(ds))
    
    Parameters
    ----------
    ds : xarray.DaraArray i.e.  ds[var] (var = fice in BLOM)
    pweight : xarray.DataArray with area information
    
    Returns
    -------
    ds_out : xarray.DaraSet with sea-extent for each hemisphere, in March and in September

    '''
    ds_out = None
    if not isinstance(pweight,xr.DataArray):
        # only if pweight is not provided. Only works for 1deg ocean 
        grid = xr.open_mfdataset('/cluster/shared/noresm/inputdata/ocn/blom/grid/grid_tnx1v4_20170622.nc')
        pweight = grid.parea*grid.pmask
    for monthnr in [3, 9]:
        da = ds.groupby('time.month').sel(month=monthnr)
        parea = pweight.where(da>=15)
        SHout = parea.where(pweight.pclat <=0).sum(dim=('x','y'))/100/(1E6*(1000*1000))
        SHout.attrs['standard_name'] = 'siext_SH_0%i'%monthnr
        SHout.attrs['units'] = '10^6 km^2'
        SHout.attrs['long_name'] = 'southern_hemisphere_sea_ice_extent_month_0%i'%monthnr
        NHout = parea.where(pweight.pclat >=0).sum(dim=('x','y'))/100/(1E6*(1000*1000))
        NHout.attrs['standard_name'] = 'siext_NH_0%i'%monthnr
        NHout.attrs['units'] = '10^6 km^2'
        NHout.attrs['long_name'] = 'northern_hemisphere_sea_ice_extent_month_0%i'%monthnr
        if isinstance(ds_out,xr.Dataset):
            ds_out = xr.merge([ds_out, SHout.to_dataset(name = 'siext_SH_0%i'%monthnr), NHout.to_dataset(name = 'siext_NH_0%i'%monthnr)])
        else:
            ds_out = xr.merge([SHout.to_dataset(name = 'siext_SH_0%i'%monthnr), NHout.to_dataset(name = 'siext_NH_0%i'%monthnr)])
        
    return ds_out

def sea_ice_area(ds, pweight):
    ''' 
    Calculates the sea ice extent from the sea ice concentration fice in BLOM
    Sea ice concentration (fice) is the percent areal coverage of ice within the ocean grid cell. 
    Sea ice extent is the integral sum of the areas of all grid cells with at least 15% ice concentration.
    Sea ice area is the integral sum of the product of ice concentration and area of all grid cells with at least 15% ice concentration. See sea_ice_area(ds))
    
    Parameters
    ----------
    ds : xarray.DaraArray i.e.  ds[var] (var = fice in BLOM)
    pweight : xarray.DataArray with area information
    
    Returns
    -------
    ds_out : xarray.DaraSet with sea-extent for each hemisphere, in March and in September

    '''
    ds_out = None
    if not isinstance(pweight,xr.DataArray):
        # only if pweight is not provided. Only works for 1deg ocean 
        grid = xr.open_mfdataset('/cluster/shared/noresm/inputdata/ocn/blom/grid/grid_tnx1v4_20170622.nc')
        pweight = grid.parea*grid.pmask
    for monthnr in [3, 9]:
        da = ds.groupby('time.month').sel(month=monthnr)
        parea = (da*pweight).where(da>=15)
        SHout = parea.where(pweight.pclat <=0).sum(dim=('x','y'))/100/(1E6*(1000*1000))
        SHout.attrs['standard_name'] = 'siarea_SH_0%i'%monthnr
        SHout.attrs['units'] = '10^6 km^2'
        SHout.attrs['long_name'] = 'southern_hemisphere_sea_ice_area_month_0%i'%monthnr
        NHout = parea.where(pweight.pclat >=0).sum(dim=('x','y'))/100/(1E6*(1000*1000))
        NHout.attrs['standard_name'] = 'siarea_NH_0%i'%monthnr
        NHout.attrs['units'] = '10^6 km^2'
        NHout.attrs['long_name'] = 'northern_hemisphere_sea_ice_area_month_0%i'%monthnr
        if isinstance(ds_out,xr.Dataset):
            ds_out = xr.merge([ds_out, SHout.to_dataset(name = 'siarea_SH_0%i'%monthnr), NHout.to_dataset(name = 'siarea_NH_0%i'%monthnr)])
        else:
            ds_out = xr.merge([SHout.to_dataset(name = 'siarea_SH_0%i'%monthnr), NHout.to_dataset(name = 'siarea_NH_0%i'%monthnr)])

    return ds_out

        
    
    

       
    

