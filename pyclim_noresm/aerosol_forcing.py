import xarray as xr
import xarray
import numpy as np
from .general_util_funcs import global_avg
from typing import Union


# Should I use xarray.dataset_accessor???

def merge_exp_ctrl(ds_control : xarray.Dataset,
                 ds_experiment : xarray.Dataset,
                 model_broadcast_equals=True,
                 varialbe_broadcast_equals=True) -> xarray.Dataset:
    """
    Merge a control and experiment CMIP6 into one dataset and makes 
    dimmensions and coordinates consistent between the control and 
    experiment runs

    Parameters
    ----------
        ds_control:     xarray.Dataset
                            CMIP6 data from control simulation.
        ds_experiment:  xarray.Dataset
                            CMIP6 data from experiment simulation.
        model_broadcast_equals: bool, default= True
                            Control and experiment have to be of the same model
        varialbe_broadcast_equals: bool, default=True
                            Control and experiment have be of the same varialbe 

    Return
    ------
        Merged Dataset containing both experiment and control simulation.
    
    """


    if ds_control.source_id != ds_experiment.source_id and model_broadcast_equals:
        raise AssertionError("""The control and experiment are not from the same source model""")
    if ds_control.variable_id != ds_experiment.variable_id and varialbe_broadcast_equals:
        raise AssertionError("""Control and experiment have to be of the same varialbe""")

    ctrl_lat_bnds, ctrl_lon_bnds = ds_control.lat_bnds, ds_control.lon_bnds
    exp_lat_bnds, exp_lon_bnds = ds_experiment.lat_bnds, ds_experiment.lon_bnds
    # Check if the lon lat bnds are equal between control and experiment
    np.testing.assert_allclose(ctrl_lat_bnds, exp_lat_bnds, atol=1e-4)
    np.testing.assert_allclose(ctrl_lon_bnds, exp_lon_bnds, atol=1e-4)
    np.testing.assert_allclose(ds_control.lon,ds_experiment.lon,atol=1e-4)
    np.testing.assert_allclose(ds_control.lat, ds_experiment.lat,atol=1e-4)

    ds_control = ds_control.reindex({'lon':ds_experiment.lon,'lat':ds_experiment.lat}, method='nearest')
    ds_control = ds_control.assign({'lon_bnds':ds_experiment.lon_bnds, 
                                    'lat_bnds':ds_experiment.lat_bnds})

    ds_control = ds_control.rename({ds_control['variable_id']: 'control_{}'.format(ds_control['variable_id'])})

    ds = xr.merge([ds_experiment,ds_control], compat='broadcast_equals', 
                    combine_attrs='drop_conflicts')

    return ds
    


def _check_consitancy_exp_control(experiment, control):
    """
    Checks if variables in the experiment and control simulations are the same 
    and that the typing is consititent. If the experiment and control are provided as 
    xarray.datasets it is returned as xarray.datasets 
    
    Parameters
    ----------
        experiment:   xarray.DataArray
                        Model output from the experiment simulation
        control:      xarray.DataArray
                        Model output from the control simulation

    """
    is_exp_da, is_cont_da = [isinstance(experiment, xr.DataArray), isinstance(control, xr.DataArray)]

    if is_exp_da and is_cont_da:
        varialbe = experiment.name
        if varialbe != control.name:
            raise AssertionError("Experiment and control dataset need to be of the same variables")

    else:
        raise AssertionError("Control and experiment have to be xarray.DataArray objects")
    

def calc_SW_ERF(experiment_downwelling: xarray.DataArray, 
                  experiment_upwelling: xarray.DataArray,
                  control_downwelling: xarray.DataArray, 
                  control_upwelling: xarray.DataArray, 
                ) -> xarray.DataArray:
    """
    Calculates SW ERF (direct aerosol focing) at surface or top of the atmosphere,
    depending on the provided input variables. Also makes sure that the calculated
    ERF are consitent with the provided input.

    Parameters
    ----------
        experiment_downwelling: xarray.DataArray
                                    The downwelling variable in the ERF calculation from the experiment. 
        experiment_upwelling:   xarray.DataArray
                                    The upwelling variable in the ERF calculation from the experiment.
        control_downwelling:    xarray.DataArray 
                                    The downwelling variable in the ERF calculation from the control.
        control_upwelling:      xarray.DataArray
                                    The upwelling variable in the ERF calculation from the control.
    Return
    ------
        erf: xarray.DataArray
                Calculated erf that is consitent with input data. The return DataArray includes new 
                metadata.

    
    """
    _check_consitancy_exp_control(experiment_downwelling, control_downwelling)
    _check_consitancy_exp_control(experiment_upwelling, control_upwelling)
    # Make sure that downwelling varialbe is paired with correst upwelling varialbe
    down_up_var_pairs = {'rsut':'rsdt',
                        'rsus':'rsds',
                        'rsuscs':'rsdscs',
                        'rsutcs':'rsdt'
                        }
    variable_down = experiment_downwelling.name
    variable_up = experiment_upwelling.name
    units = experiment_upwelling.units    
    if down_up_var_pairs[variable_down] != variable_up:
        raise ValueError(f'The combination {variable_down} and {variable_up} is invalid')
                            

    attrs = {'rsut': 
                {'varialbe_name': 'ERFtsw',
                'long_name':'Effective radiative forcing short wave at the top of the atmosphere', 
                'units': units},
            'rsus': 
                {'variable_name': 'ERFsurfsw',
                'long_name':'Effective radiative forcing short wave at the surface', 
                'units': units},
            'rsuscs':
                {'variable_name':'ERFsurfswcs',
                'long_name':'Effective clear sky radiative forcing short wave at the surface', 
                'units': units},
            'rsutcs':
                {'varialbe_name': 'ERFtswcs',
                'long_name':'Effective clear sky radiative forcing short wave at the top of the atmosphere', 
                'units': units}
    
            }
            
            

    erf = (np.abs(experiment_downwelling) - np.abs(experiment_upwelling))-(np.abs(control_downwelling)-np.abs(control_upwelling))
    erf = erf.rename(attrs[variable_down]['variable_name'])
    erf.attrs = attrs[variable_down]
    
    return erf



def calc_atm_abs(delta_rad_surf: xarray.DataArray, 
                delta_rad_toa : xarray.DataArray):
    """
    Calculates the atmospheric absorption as the difference between 
    the radiative imbalance at the top of the atmosphere and the surface. 

    
    """
    varialbe_pairs={
        'ERFtsw':'ERFsurfsw',
        'ERFtswcs' : 'ERFsurfswcs'

    }
    variable_toa = delta_rad_toa.name
    variable_surf = delta_rad_surf.name
    units = delta_rad_toa.units
    if varialbe_pairs[variable_toa] != variable_surf:
        raise ValueError(f'The combination {variable_toa} and {variable_surf} is invalid')

    attrs = {
        'ERFtsw': 
            {'variable_name':'atmabsSW',
            'long_name':'Atmospheric absorbtion of short wave radiation.', 
            'units': units
            },
        'ERFtswcs': 
            {'variable_name':'atmabsSWcs',
            'long_name':'Clear sky atmospheric absorbtion of short wave radiation.',
            'units': units    
            
            }
        

    }


    atm_abs = delta_rad_toa - delta_rad_surf
    atm_abs = atm_abs.rename(attrs[variable_toa]['variable_name'])
    atm_abs.attrs = {**atm_abs.attrs,**attrs}

    return atm_abs
