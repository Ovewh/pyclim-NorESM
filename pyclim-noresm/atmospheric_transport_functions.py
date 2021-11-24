#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday November 24 16:30:24 2021

@author: Ada Gjermundsen

python functions used for atmospheric heat transport calculations 

"""
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter('ignore')

def select_region(da, lat_lim=70):
    ''' select latitude circle'''
    da = da.sel(lat=slice(lat_lim,None))
    da = da.isel(lat=0)
    return da

def select_time(da, yr1=None, yr2=None):
    if yr1 and yr2:
        da = da.sel(time=slice(str(yr1)+"-01-01", str(yr2) +"-12-31"))
    elif yr1 and not yr2:
        da = da.sel(time=slice(str(yr1)+"-01-01",None))
    elif not yr1 and yr2:
        da = da.sel(time=slice(str(yr2) +"-12-31"))
    return da

def make_dx(da, lon_bnds):
    deg2m = 111132.945
    # assume that the Artic region already have been selected by the use of the "select_region" function
    if 'time' in lon_bnds.dims:
        lon_bnds = lon_bnds.isel(time=0).drop('time')
    dlon = np.diff(np.append(lon_bnds.sel(bnds=0).isel(lon=0).values, lon_bnds.sel(bnds=1).values))
    if 'lat' in da.dims:
        if len(da.lat.values) > 1:
            dx = deg2m*np.tile(dlon,[ len(da.lat.values),1]).T*np.tile(np.cos(da.lat.values*np.pi/180),[len(dlon), 1])
            dx = xr.DataArray(dx, dims=('lon', 'lat'), coords={'lon':da.lon, 'lat':da.lat})
        else:
            dx = deg2m*dlon*np.cos(da.lat.values*np.pi/180)
            dx = xr.DataArray(dx, dims=('lon'), coords={'lon':da.lon})
    else:
        dx = deg2m*dlon*np.cos(da.lat.values*np.pi/180)
        dx = xr.DataArray(dx, dims=('lon'), coords={'lon':da.lon})
    return dx

def make_pressure_bnds(ds):
    '''
    Parameters
    ----------
    ds :     xarray.Dataset with variables ('a_bnds','b_bnds','ps','p0') needed to calculate the pressure thickness. 
    
    Returns
    -------
    dp:     xarray.Dataset with pressure thickness 
    '''
    a_bnds = ds.a_bnds
    if 'time' in a_bnds.dims:
        a_bnds = a_bnds.isel(time=0).drop('time')
    abnds = np.append(a_bnds.sel(bnds=0).isel(lev=0).values, a_bnds.sel(bnds=1).values)
    b_bnds = ds.b_bnds
    if 'time' in b_bnds.dims:
        b_bnds = b_bnds.isel(time=0).drop('time')
    bbnds = np.append(b_bnds.sel(bnds=0).isel(lev=0).values, b_bnds.sel(bnds=1).values)
    # this will fail if the dataset has been reduced already, e.g. a given latitude has already been selected
    t, lats, lons =  ds.ps.shape
    # to get this correct you need to set the a_bnds dim last, and then transpose
    abnds = np.tile(abnds, [t, lats, lons, 1])
    # just set lev_bnds as a dummy variable - you don't need it in the end
    abnds = xr.DataArray(abnds, dims=('time', 'lat', 'lon' ,'lev_bnds'), coords={'time' : ds.time, 'lat':ds.lat, 'lon': ds.lon ,'lev_bnds': np.arange(0,len(a_bnds.values)+1)})
    # to get this correct you need to set the a_bnds dim last, and then transpose
    bbnds = np.tile(bbnds, [t, lats, lons, 1])
    # just set lev_bnds as a dummy variable - you don't need it in the end
    bbnds = xr.DataArray(bbnds, dims=('time', 'lat',  'lon' ,'lev_bnds'), coords={'time' : ds.time, 'lat':ds.lat, 'lon': ds.lon ,'lev_bnds': np.arange(0,len(a_bnds.values)+1)})
    ps = ds.ps*xr.ones_like(bbnds)
    bnds = ds.p0*abnds + bbnds*ps
    # need to remove mountain regions: set value to nan if p > ps
    bnds = xr.where(bnds<=ps, bnds, ps)
    bnds = -bnds.diff(dim='lev_bnds')
    bnds = bnds.rename({'lev_bnds':'lev'})
    bnds = bnds.assign_coords({"lev": ds.lev})
    bnds = bnds.transpose('time', 'lev', 'lat','lon')
    return bnds.to_dataset(name='dp')

def atmos_energy_flux(u, v, T, q, z, dp, lon_bnds):
    '''This function is used to calculate atmospheric northward energy flux'''
    # cp: specific heat of air at constant pressure
    cp = 1004 # J kg-1 K-1
    # g. acceleration of gravity 
    g = 9.81 # m s-2
    # L: latent heat of vaporization for water (@0°C)
    L = 2.5e6 # J kg-1

    F1 = cp*T          # internal energy flux
    F2 = g*z           # potential energy flux
    F3 = L*q           # latent energy flux
    F4 = 0.5*(u*u + v*v)             # kinetic energy flux
    F = F1 + F2 + F3 + F4  # total energy flux

    dx = make_dx(F, lon_bnds)
    dx = dx*xr.ones_like(F)
    energy_flux = (F*v*dx*dp/g)
    energy_flux_moist =  (F3*v*dx*dp/g)
    energy_flux_dry = (F1 + F2)*v*(dx*dp/g)
    energy_flux_kinetic = (F4*v*dx*dp/g)
    return energy_flux, energy_flux_moist, energy_flux_dry, energy_flux_kinetic

