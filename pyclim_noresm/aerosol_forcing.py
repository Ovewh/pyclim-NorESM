from ast import Raise
import xarray as xr
import xarray
import numpy as np
from .general_util_funcs import global_avg
from typing import Union

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
                ):
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
    erf = erf.rename(attrs[varialbe_down])
    erf.attrs = attrs[varialbe_down]
    
    return erf
