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
        experiment:   xarray.Dataset, xarray.DataArray
                        Model output from the experiment simulation
        control:      xarray.Dataset,xarray.DataArray
                        Model output from the control simulation
        
    Return
    ------
        da_exp:     xarray.DataArray
                        Model output from the experiment simulation (checked that it is consitent with control)
        da_control  xarray.Dataset
                        Model output from the control simulation (checked that it is consitent with experiment)

    """
    is_exp_ds, is_exp_da, is_cont_ds, is_cont_da = [isinstance(experiment, xr.Dataset), isinstance(experiment, xr.Dataset),
                                                    isinstance(control, xr.Dataset), isinstance(control, xr.DataArray)]


    if is_exp_ds and is_cont_ds:
        varialbe = experiment.variable_id
        if varialbe != control.variable_id:
            raise ValueError("Experiment and control dataset need to be of the same variables")
        da_exp = experiment[varialbe]
        da_cont = experiment[varialbe]
    elif is_exp_da and is_cont_ds or is_exp_ds and is_cont_da:
        if is_exp_ds:
            varialbe = experiment.variable_id
            da_exp = experiment[varialbe]
        else:
            varialbe = experiment.name
            da_exp = experiment
        if is_cont_ds:
            varialbe_cont_varialbe = control.variable_id
            da_cont = control[varialbe]
        else:
            varialbe_cont_varialbe = control.name
            da_cont = control
        if varialbe != varialbe_cont_varialbe:
            raise ValueError("Experiment and control dataset need to be of the same variables")
    elif is_exp_da and is_cont_da:
        varialbe = experiment.name
        if varialbe != control.name:
            raise ValueError("Experiment and control dataset need to be of the same variables")
        da_cont = control
        da_exp = control
    else:
        raise ValueError("Control and experiment have to be of type xarray.Dataset or xarray.DataArray")
    
    return da_exp, da_cont, varialbe


def calc_SW_ERF(experiment_downwelling: Union[xarray.Dataset,xarray.DataArray], 
                  experiment_upwelling: Union[xarray.Dataset,xarray.DataArray],
                  control_downwelling: Union[xarray.Dataset,xarray.DataArray], 
                  control_upwelling: Union[xarray.Dataset,xarray.DataArray], 
                ) -> xarray.DataArray:
    """
    Calculates SW ERF (direct aerosol focing) at surface or top of the atmosphere,
    depending on the provided input variables. Also makes sure that the calculated
    ERF are consitent with the provided input.

    Parameters
    ----------
        experiment_downwelling: Union[xarray.Dataset,xarray.DataArray]
                                    The downwelling variable in the ERF calculation from the experiment. 
        experiment_upwelling:   Union[xarray.Dataset,xarray.DataArray]
                                    The upwelling variable in the ERF calculation from the experiment.
        control_downwelling:    Union[xarray.Dataset,xarray.DataArray] 
                                    The downwelling variable in the ERF calculation from the control.
        control_upwelling:      Union[xarray.Dataset,xarray.DataArray]
                                    The upwelling variable in the ERF calculation from the control.
    Return
    ------
        erf: xarray.DataArray
                Calculated erf that is consitent with input data. The return DataArray includes new 
                metadata.

    
    """
    exp_down,control_down,varialbe_down = _check_consitancy_exp_control(experiment_downwelling, control_downwelling)
    exp_up, control_up, varialbe_up = _check_consitancy_exp_control(experiment_upwelling, control_upwelling)
    # Make sure that downwelling varialbe is paired with correst upwelling varialbe
    down_up_var_pairs = {'rsut':'rsdt',
                        'rsus':'rsds',
                        'rsuscs':'rsdscs',
                        'rsutcs':'rsdt'
                        }
    if down_up_var_pairs[varialbe_down] != varialbe_up:
        raise ValueError(f'The combination {varialbe_down} and {varialbe_up} is invalid')
                            

    attrs = {'rsut': 
                {'varialbe_name': 'ERFtsw',
                'long_name':'Effective radiative forcing short wave at the top of the atmosphere', 
                'units': exp_down.units},
            'rsus': 
                {'variable_name': 'ERFsurfsw',
                'long_name':'Effective radiative forcing short wave at the surface', 
                'units': exp_down.units},
            'rsuscs':
                {'variable_name':'ERFsurfswcs',
                'long_name':'Effective clear sky radiative forcing short wave at the surface', 
                'units': exp_down.units},
            'rsutcs':
                {'varialbe_name': 'ERFtswcs',
                'long_name':'Effective clear sky radiative forcing short wave at the top of the atmosphere', 
                'units': exp_down.units}
    
            }
            
            

    erf = (np.abs(exp_down) - np.abs(exp_up))-(np.abs(control_down)-np.abs(control_up))
    erf = erf.rename(attrs[varialbe_down]['variable_name'])
    erf.attrs = attrs[varialbe_down]
    
    return erf

def calc_atm_abs(delta_rad_surf: Union[xarray.Dataset,xarray.DataArray], 
                delta_rad_toa : Union[xarray.Dataset,xarray.DataArray]):
    """
    Calculates the atmospheric absorption as the difference between 
    the radiative imbalance at the top of the atmosphere and the surface. 

    
    """
    varialbe_pairs={
        'ERFtsw':'ERFsurfsw',
        'ERFtswcs' : 'ERFsurfswcs'

    }

    if isinstance(delta_rad_surf, xarray.DataArray):
        variable_surf = delta_rad_surf.name
    else:
        variable_surf = delta_rad_surf['variable_name']
        delta_rad_surf = delta_rad_surf['variable_name']

    if isinstance(delta_rad_toa, xarray.DataArray):
        variable_toa = delta_rad_toa.name
    else:
        variable_toa = delta_rad_toa['variable_name']
        delta_rad_toa = delta_rad_toa['variable_name']

    if varialbe_pairs[variable_toa] != variable_surf:
        raise ValueError(f'The combination {variable_toa} and {variable_surf} is invalid')

    attrs = {
        'ERFtsw': 
            {'variable_name':'AtmabsSW',
            'long_name':'Short wave atmospheric absorption', 
            'units': delta_rad_toa.uni

            }

    }


    atm_abs = delta_rad_toa - delta_rad_surf
