#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:08:18 2020

@author: adag
"""
import glob
import numpy as np
import warnings
warnings.simplefilter('ignore')
import xarray as xr
xr.set_options(enable_cftimeindex=True)

def fx_files(self, var='areacella', path_to_data = '/projects/NS9252K/ESGF/CMIP6/'):
    '''
    '''
    if var in ['areacella', 'orog', 'sftgif', 'sftlf']:
        fx = 'fx'
    if var in ['areacello', 'deptho', 'thkcello', 'basin', 'volcello', 'sftof']:
        fx = 'Ofx'     
    if var in ['thkcello',  'volcello']:
    # This is not a very good solution, but it will work for NorESM2 though
    # but say we output data on native and latlon grid, then we will have gn and gr for the same variable...
        self.gridlabel = 'gr'
    else:
        self.gridlabel = 'gn'
    if self.name in ['NorESM2-LM', 'NorESM2-MM']:
         ofxpath = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/piControl/' + self.realiz + '/' + fx + '/' + var + '/'+ self.gridlabel +'/latest/'
         self.ofxfile = ofxpath + var + '_' + fx +  '_' + self.name + '_piControl_'+ self.realiz + '_' + self.gridlabel +'.nc'
    else:
         ofxpath = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/piControl/' + self.realiz + '/' + fx  +'/' + var + '/' + self.gridlabel+ '/latest/'
         self.ofxfile = ofxpath + var + '_'+ fx+'_' + self.name + '_piControl_' + self.realiz + '_' + self.gridlabel +'.nc'


# Need to add some sort of list available ocean grid function


#def noresm_cmip6():
#    '''
#    This function was very useful for analyzing 41 CMIP6 models, probably not necessary when only analysing NorESM models
#    you can rather give default values
#    '''
#    models={'NorCPM1':{'institute':'NCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
#          'NorESM2-LM':{'institute':'NCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':0},
#          'NorESM2-MM':{'institute':'NCC', 'grid_label_atmos':['gn'],  'grid_label_ocean':['gn','gr'], 'variant_labels':['r1i1p1f1'],'branch_yr':0}}
#    # the ocean grid should be input, I'll just save the information here for a while :)
#    # if self.name in ['NorESM2-LM', 'NorESM2-MM'] and var not in  ['hfbasin', 'so', 'wo','uo','msftmzmpa','msftmz','msftmrho','hfbasin', 'hfbasinpadv','hfbasinpmadv','hfbasinpsmadv', 'hfbasinpmdiff', 'thetao', 'thetaoga','tosga']:
#    # gridlabel = 'gn'
#    #elif self.name in ['NorESM2-LM', 'NorESM2-MM'] and var in[ 'so' , 'wo','uo', 'thetao']:
#    #            gridlabel = 'gr'
#    #        elif self.name in ['NorESM2-LM', 'NorESM2-MM'] and var in ['tosga', 'thetaoga']:
#    #            gridlabel = 'gm'
#    #            print(gridlabel)
#    #        elif self.name in ['NorESM2-LM', 'NorESM2-MM'] and var in [ 'msftmzmpa','msftmz','msftmrho','hfbasin', 'hfbasinpadv','hfbasinpmadv','hfbasinpsmadv', 'hfbasinpmdiff']:
#    #            gridlabel = 'grz'
#
#    return models


def make_filelist_cmor(self, var, component = 'atmos', activity_id='CMIP', path_to_data = '/projects/NS9034K/CMIP6/'):
        '''
        Function for providing a list of all files in order to read a cmorized variable for a given experiment. 
        There is usually only one grid for the atmosphere; gn (grid native)
        For the ocean and sea-ice files there are often both native (gn) and regridded files (gr, grz, gm). If not gn is used it is necessary state which grid label is wanted in self.grid_label_ocean

        Parameters
        ----------
        self: model object
        var :          str, name of variable
        component:     str, name of component (ocean, atmos, land, seaice). Default is atmos
        activity_id :  str, which MIP the experiment belongs to. Default is 'CMIP'    
        path_to_data : str, path to the data folders. Default is '/projects/NS9034K/CMIP6/'.

        Returns
        -------
        Sets a list if filename(s) as an attribute of the model object; self.filenames

        '''
        import glob
        self.variable = var
        self.activity_id = activity_id
        if component == 'atmos':
            self.gridlabel = self.grid_label_atmos
        if component in 'ocean':
            self.gridlabel = self.grid_label_ocean
        self.path = path_to_data + '/' + self.activity_id  + '/' + self.institute + '/' + self.name + '/' + self.expid + '/' + self.realiz + '/' + self.realm + '/' + self.variable + '/' + gridlabel+ '/' 
        # We need to fix the multiple version challenge. Not all files are necessarily located in the 'latest' folder
        versions = sorted(glob.glob(self.path +'*'))
        if versions:
            fnames = sorted(glob.glob(versions[0] +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz +'_' + gridlabel + '_*.nc'))
        else:
            fnames = []
        if len(versions)>1:
            for version in versions[1:]:
                files = sorted(glob.glob(version +'/' + self.variable +'_' + self.realm +'_' + self.name + '_' + self.expid + '_' + self.realiz +'_' + gridlabel + '_*.nc'))   
                for file in files:
                    if versions[0] +'/' +file.split('/')[-1] not in fnames:
                        fnames.append(file)              
        if fnames:
           if self.name=='NorESM2-MM' and self.realiz == 'r3i1p1f1':
              # There exists one extra file for all cmorized variables for NorESM2-MM, member r3i1p1f1. Need to remove the file from the filelist
              fnames.remove('/projects/NS9034K/CMIP6//CMIP/NCC/NorESM2-MM/historical/r3i1p1f1/Omon/' + var + '/' +gridlabel +'/latest/' + var +'_Omon_NorESM2-MM_historical_r3i1p1f1_'+gridlabel+'_186001-186105.nc')
           if len(fnames)>1:
               # test that the files contained in the filelist covers all years in consecutive order
               fnames = sorted(fnames ,key=lambda x: extract_number(x)) 
               checkConsecutive(fnames)
           self.filenames = fnames
           print('\n Final list of filenames:')
           print(self.filenames) 
        if not fnames:
           self.filenames = ''
        if not fnames:
            print('Variable %s not prestent in output folder for model %s\n'%(self.variable, self.name))
        #raise Exception

def extract_number(string):
    return string.split('_')[-1]

def extract_dates(string):
    return string.split('_')[-1].split('.')[0]


def checkConsecutive(fnames):
    sorteddates = [extract_dates(x) for x in fnames]
    for i in range(1,len(sorteddates)):
        if int(sorteddates[i].split('01-')[0]) != int(sorteddates[i-1].split('-')[1][:-2])+1:
            #print('NOTE! The files are not in consecutive order. Please check directory')
            print(fnames)
            raise Exception('NOTE! The files are not in consecutive order. Please check directory')

def make_filelist_raw(expid, path, component='atmos', yrs = None, yre = None):
    if component in ['atmos']:
        fnames = '%s/atm/hist/%s.cam.h0.*.nc'%(expid, expid)
    if component in ['ocean']:
        fnames = '%s/ocn/hist/%s.blom.hm.*.nc'%(expid, expid)
        if not sorted(glob.glob(path + fnames)):
            fnames = '%s/ocn/hist/%s.micom.hm.*.nc'%(expid, expid)
    if component in ['land']:
        fnames = '%s/lnd/hist/%s.clm2.h0.*.nc'%(expid, expid)
    if component in ['seaice']:
        fnames = '%s/ice/hist/%s.cice.h.*.nc'%(expid, expid)
    fnames = sorted(glob.glob(path + fnames))
    if yrs or yre:
        # all simulated years in experiment
        allyears = [int(fnames[i].split('/')[-1].split('.')[-2].split('-')[0].lstrip('0')) for i in range(0,len(fnames))]
        if yrs and yre:
             # create subset of filenames starting with year: yrs and ending with year: yre
            boole =  (np.array(allyears)<=yre)*(np.array(allyears)>=yrs)
        elif yrs and not yre:
            # create subset of filenames starting with year: yrs and to the end of the simulation
            boole =(np.array(allyears)>=yrs)
        elif not yrs and yre:
            # create subset of filenames from the start of the simulation and ending with year: yre
            boole =  (np.array(allyears)<=yre)
        fnames = [val for is_good, val in zip(boole, fnames) if is_good]
    # test that the files contained in the filelist covers all years in consecutive order
    if len(fnames)>1:
        for i in range(12,len(fnames),12):
            if int(fnames[i].split('/')[-1].split('.')[-2].split('-')[0].lstrip('0'))!=int(fnames[i-12].split('/')[-1].split('.')[-2].split('-')[0].lstrip('0'))+1:
                #print('NOTE! The files are not in consecutive order. Please check directory')
                print(fnames)
                raise Exception('NOTE! The files are not in consecutive order. Please check directory')
    return fnames

def read_noresm_raw(fnames, dim='time', transform_func=None):
    # if the reading crashes due to memory issues you may add chunks, i.e. parallel=True, chunks={"time":12}
    with xr.open_mfdataset(fnames, concat_dim="time", combine="nested",
                  data_vars='minimal', coords='minimal', compat='override') as ds:
        # transform_func should do some sort of selection on aggregation
        if transform_func is not None:
            ds = transform_func(ds)
    return ds
            

def read_noresm_cmor(model, varlist, component = 'atmos', dim='time', transform_func=None):
    ds = None
    for var in varlist:
        make_filelist(model, var,  component = component, activity_id= model.activity_id, path_to_data = '/projects/NS9034K/CMIP6/')
        # if the reading crashes due to memory issues you may add chunks, i.e. parallel=True, chunks={"time":12}
        with xr.open_mfdataset(fnames, concat_dim="time", combine="nested",
                  data_vars='minimal', coords='minimal', compat='override') as ds:
            # transform_func should do some sort of selection on aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            if isinstance(ds, xr.Dataset):
                ds = xr.merge([ds, xr.open_mfdataset(model.filenames, concat_dim="time", combine="nested",
                               data_vars='minimal', coords='minimal', compat='override')])
            else:
                ds = xr.merge([ds, xr.open_mfdataset(model.filenames, concat_dim="time", combine="nested",
                               data_vars='minimal', coords='minimal', compat='override')])
    return ds
delinfo

 
class Modelinfo:
    '''
    Sets the details of the model experiment, including filenames
    '''
    
    def __init__(self, name = 'NorESM2-LM', institute = 'NCC', expid='piControl', realm='Amon', 
                  realiz=['r1i1p1f1'], grid_atmos = 'gn', grid_ocean = 'gn', branchtime_year=0):
        '''

        Attributes
        ----------
        name :        str, name of the CMIP model - typically the Source ID
        institute :   str, Institution ID
        expid :       str, Experiment ID, e.g. piControl, abrupt-4xCO2
        realm :       str, which model domain and time frequency used, e.g. Amon, AERmon, Omon
        grid_labels : list, which grid resolutions are available. Modt useful for ocean and sea-ice variables. 
                      e.g. ['gn', 'gr'] gn: native grid, 'gr': regridded somehow - not obvious
        realiz :      str, variant labels saying something about which realization (ensemble member), initial conditions, forcings etc.
                      e.g. 'r1i1p1f1'
        version :     str, which version of the data is read. default is latest. 
        branchtime_year : int, when simulation was branched off from parent. 
                          Useful when anomalies are considered e.g. abrupt-4xCO2, historical 
                          then you only consider data from piCOntrol for the corresponding period i.e. piControl_data_array(time = branchtime_year:)
        
        '''
        self.name = name
        self.institute = institute
        self.expid = expid
        self.realm = realm
        self.realiz = realiz
        self.grid_label_atmos = grid_atmos
        self.grid_label_ocean = grid_ocean
        self.branchtime_year = branchtime_year

