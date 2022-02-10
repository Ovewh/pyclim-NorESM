import xarray as xr
import xarray
import numpy as np
from .regrid_functions import make_latlon_bounds

def merge_exp_ctrl(
    ds_control: xarray.Dataset,
    ds_experiment: xarray.Dataset,
    model_broadcast_equals=True,
    varialbe_broadcast_equals=True,
) -> xarray.Dataset:
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
        raise AssertionError(
            """The control and experiment are not from the same source model"""
        )
    if (
        ds_control.variable_id != ds_experiment.variable_id
        and varialbe_broadcast_equals
    ):
        raise AssertionError(
            """Control and experiment have to be of the same varialbe"""
        )

    # Check if the lon lat bnds are equal between control and experiment
    if 'lat_bnds' not in ds_control.variables or 'lon_bnds' not in ds_control.variables:
        ds_control = make_latlon_bounds(ds_control)
    if 'lat_bnds' not in ds_experiment.variables or 'lon_bnds' not in ds_experiment.variables:
        ds_experiment = make_latlon_bounds(ds_experiment)
    try:
        np.testing.assert_allclose(ds_control.lon_bnds, ds_experiment.lon_bnds, atol=1e-4)
        np.testing.assert_allclose(ds_control.lat_bnds, ds_experiment.lat_bnds, atol=1e-4)
        np.testing.assert_allclose(ds_control.lat, ds_experiment.lat, atol=1e-4)
    except AssertionError:
        ds_control = make_latlon_bounds(ds_control)
        ds_experiment = make_latlon_bounds(ds_experiment)
        np.testing.assert_allclose(ds_control.lon_bnds, ds_experiment.lon_bnds, atol=1e-4)
        np.testing.assert_allclose(ds_control.lat_bnds, ds_experiment.lat_bnds, atol=1e-4)
        np.testing.assert_allclose(ds_control.lat, ds_experiment.lat, atol=1e-4)
        

    ds_control = ds_control.reindex(
        {"lon": ds_experiment.lon, "lat": ds_experiment.lat}, method="nearest"
    )
    ds_control = ds_control.assign(
        {"lon_bnds": ds_experiment.lon_bnds, "lat_bnds": ds_experiment.lat_bnds}
    )
    ds_control = ds_control.rename(
        {
            ds_control.attrs["variable_id"]: "control_{}".format(
                ds_control.attrs["variable_id"]
            )
        }
    )

    if len(ds_control.time) > len(ds_experiment.time):
        ds_control = ds_control.sel(time=ds_experiment.time)
    else:
        ds_experiment = ds_experiment.sel(time=ds_control.time)
    ds = xr.merge(
        [ds_experiment, ds_control],
        compat="broadcast_equals",
        combine_attrs="drop_conflicts",
    )

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
    is_exp_da, is_cont_da = [
        isinstance(experiment, xr.DataArray),
        isinstance(control, xr.DataArray),
    ]

    if is_exp_da and is_cont_da:
        varialbe = experiment.name
        if varialbe != control.name:
            raise AssertionError(
                "Experiment and control dataset need to be of the same variables"
            )

    else:
        raise AssertionError(
            "Control and experiment have to be xarray.DataArray objects"
        )


def calc_SW_ERF(
    experiment_downwelling: xarray.DataArray,
    experiment_upwelling: xarray.DataArray,
    ctrl_downwelling: xarray.DataArray,
    ctrl_upwelling: xarray.DataArray,
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
        ctrl_downwelling:    xarray.DataArray
                                    The downwelling variable in the ERF calculation from the control.
        ctrl_upwelling:      xarray.DataArray
                                    The upwelling variable in the ERF calculation from the control.
    Return
    ------
        erf: xarray.DataArray
                Calculated erf that is consitent with input data. The return DataArray includes new
                metadata.


    """
    _check_consitancy_exp_control(experiment_downwelling, ctrl_downwelling)
    _check_consitancy_exp_control(experiment_upwelling, ctrl_upwelling)
    # Make sure that downwelling varialbe is paired with correst upwelling varialbe
    down_up_var_pairs = {
        "rsut": "rsdt",
        "rsus": "rsds",
        "rsuscs": "rsdscs",
        "rsutcs": "rsdt",
    }
    variable_down = experiment_downwelling.name
    variable_up = experiment_upwelling.name
    units = experiment_upwelling.units
    if down_up_var_pairs[variable_up] != variable_down:
        raise ValueError(
            f"The combination {variable_down} and {variable_up} is invalid"
        )

    attrs = {
        "rsut": {
            "variable_name": "ERFtsw",
            "long_name": "Effective radiative forcing short wave at the top of the atmosphere",
            "units": units,
        },
        "rsus": {
            "variable_name": "ERFsurfsw",
            "long_name": "Effective radiative forcing short wave at the surface",
            "units": units,
        },
        "rsuscs": {
            "variable_name": "ERFsurfswcs",
            "long_name": "Effective clear sky radiative forcing short wave at the surface",
            "units": units,
        },
        "rsutcs": {
            "variable_name": "ERFtswcs",
            "long_name": "Effective clear sky radiative forcing short wave at the top of the atmosphere",
            "units": units,
        },
    }

    erf = (-np.abs(experiment_downwelling) + np.abs(experiment_upwelling)) - (
        -np.abs(ctrl_downwelling) + np.abs(ctrl_upwelling)
    )
    erf = erf.rename(attrs[variable_up]["variable_name"])
    erf.attrs = attrs[variable_up]

    return erf


def calc_total_ERF_surf(
    experiment_downwelling_SW_surf: xarray.DataArray,
    experiment_upwelling_SW_surf: xarray.DataArray,
    experiment_upwelling_LW_surf: xarray.DataArray,
    experiment_downwelling_LW_surf: xarray.DataArray,
    ctrl_downwelling_SW_surf: xarray.DataArray,
    ctrl_upwelling_SW_surf: xarray.DataArray,
    ctrl_upwelling_LW_surf: xarray.DataArray,
    ctrl_downwelling_LW_surf: xarray.DataArray,
   
) -> xarray.DataArray:

    """
    Calculate the surface ERF between control and experiment CMIP6 simulation.
    While making sure that the varialbe used are consistent with the derived ERF.
    Can derive both total surface ERF and clear sky ERF

    Paramters
    ---------
        experiment_downwelling_SW_surf:      xarray.DataArray
                                            Experiment downwelling SW radiation at surface (e.g. rsds).
        experiment_upwelling_SW_surf:        xarray.DataArray
                                            Experiment upwelling SW radiation at surface (e.g. rsus).
        experiment_upwelling_LW_surf:        xarray.DataArray
                                            Experiment upwelling LW radiation at surface (e.g. ruls).
        experiment_downwelling_LW_sur:       xarray.DataArray
                                            Experiment downwelling LW radiation at surface.
        ctrl_downwelling_SW_surf:            xarray.DataArray
                                            Control downwelling SW radiation at surface.
        ctrl_upwelling_SW_surf:              xarray.DataArray
                                            Control upwelling SW radiation at surface.
        ctrl_upwelling_LW_surf:              xarray.DataArray
                                            Control upwelling LW radiation at surface.
        ctrl_downwelling_LW_surf:            xarray.DataArray
                                            Control downwelling LW radiation at surface.
    Return
    ------
        xarray.DataArray : Containing calculated ERF.

    """

    _check_consitancy_exp_control(
        experiment_downwelling_SW_surf, ctrl_downwelling_SW_surf
    )
    _check_consitancy_exp_control(experiment_upwelling_SW_surf, ctrl_upwelling_SW_surf)
    _check_consitancy_exp_control(experiment_upwelling_LW_surf, ctrl_upwelling_LW_surf)
    _check_consitancy_exp_control(
        experiment_downwelling_LW_surf, ctrl_downwelling_LW_surf
    )

    down_up_var_pairs = {
        "rsus": ["rsds", "rlus", "rlds"],
        "rsuscs": ["rsds","rluscs", "rldscs"],
    }

    lookup_var = experiment_upwelling_SW_surf.name

    corresponding_vars = down_up_var_pairs[lookup_var]
    for var, name in zip(
        corresponding_vars,
        [experiment_downwelling_SW_surf.name, experiment_upwelling_LW_surf.name],
    ):
        if var != name:
            raise AssertionError(
                f"{lookup_var} not consitent with {name} for ERF TOA calculation"
            )
    units = ctrl_downwelling_SW_surf.units
    attrs = {
        "rsus": {
            "variable_name": "ERFsurf",
            "long_name": "Effective radiative forcing at the surface",
            "units": units,
        },
        "rsuscs": {
            "variable_name": "ERFsurfcs",
            "long_name": "Clear sky effective radiative forcing at the surface",
            "units": units,
        },
    }
    
    rsns_exp = - np.absolute(experiment_downwelling_SW_surf) + np.absolute(
        experiment_upwelling_SW_surf
    )
    rsns_ctrl =  - np.absolute(ctrl_downwelling_SW_surf) + np.absolute(
        ctrl_upwelling_SW_surf
    )
    rlns_exp = - np.absolute(experiment_downwelling_LW_surf) + np.absolute(
        experiment_upwelling_LW_surf
    )
    rlns_ctrl = - np.absolute(ctrl_downwelling_LW_surf) + np.absolute(
        ctrl_upwelling_LW_surf
    )
    erf = (rsns_exp - rlns_exp) - (rsns_ctrl - rlns_ctrl)
    erf = erf.rename(attrs[lookup_var]["variable_name"])
    erf.attrs = {**erf.attrs, **attrs[lookup_var]}

    return erf


def calc_total_ERF_TOA(
    experiment_downwelling_SW: xarray.DataArray,
    experiment_upwelling_SW: xarray.DataArray,
    experiment_upwelling_LW: xarray.DataArray,
    ctrl_downwelling_SW: xarray.DataArray,
    ctrl_upwelling_SW: xarray.DataArray,
    ctrl_upwelling_LW: xarray.DataArray,
) -> xarray.DataArray:
    """
    Calculate the TOA ERF between control and experiment CMIP6 simulation.
    Also make sure that the varialbe used are consistent with the derived ERF.

    Paramters
    ---------
        experiment_downwelling_SW:      xarray.DataArray
                                            Experiment downwelling SW radiation at TOA (e.g. rsdt).
        experiment_upwelling_SW:        xarray.DataArray
                                            Experiment upwelling SW radiation at TOA (e.g. rsut).
        experiment_upwelling_LW:        xarray.DataArray
                                            Experiment upwelling LW radiation at TOA (e.g. rult).
        ctrl_downwelling_SW:            xarray.DataArray
                                            Control downwelling SW radiation at TOA
        ctrl_upwelling_SW:              xarray.DataArray
                                            Control upwelling SW radiation at TOA
        ctrl_upwelling_LW:              xarray.DataArray
                                            Control upwelling LW radiation at TOA
    Return
    ------
        xarray.DataArray : Containing calculated ERF.

    """

    _check_consitancy_exp_control(experiment_downwelling_SW, ctrl_downwelling_SW)
    _check_consitancy_exp_control(experiment_upwelling_SW, ctrl_upwelling_SW)
    _check_consitancy_exp_control(experiment_upwelling_LW, ctrl_upwelling_LW)

    down_up_var_pairs = {
        "rsut": ["rsdt", "rlut"],
        "rsutcs": ["rsdt", "rlutcs"],
    }

    lookup_var = experiment_upwelling_SW.name

    corresponding_vars = down_up_var_pairs[lookup_var]
    for var, name in zip(
        corresponding_vars,
        [experiment_downwelling_SW.name, experiment_upwelling_LW.name],
    ):
        if var != name:
            raise AssertionError(
                f"{lookup_var} not consitent with {name} for ERF TOA calculation"
            )
    units = ctrl_downwelling_SW.units
    attrs = {
        "rsut": {
            "variable_name": "ERFt",
            "long_name": "Effective radiative forcing at the top of the atmosphere",
            "units": units,
        },
        "rsutcs": {
            "variable_name": "ERFtcs",
            "long_name": "Clear sky effective radiative forcing at the top of the atmosphere",
            "units": units,
        },
    }

    rsnt_exp = - np.absolute(experiment_downwelling_SW) + np.absolute(
        experiment_upwelling_SW
    )
    rsnt_ctrl = - np.absolute(ctrl_downwelling_SW) + np.absolute(
        ctrl_upwelling_SW
    )
    rlnt_exp = np.absolute(experiment_upwelling_LW)
    rlnt_ctrl = np.absolute(ctrl_upwelling_LW)
    # Take Net LW + SW TOA in experiment - control
    erf = (rsnt_exp + rlnt_exp) - (rsnt_ctrl + rlnt_ctrl)
    erf = erf.rename(attrs[lookup_var]["variable_name"])
    erf.attrs = {**erf.attrs, **attrs[lookup_var]}

    return erf


def calc_atm_abs(delta_rad_surf: xarray.DataArray, delta_rad_toa: xarray.DataArray):
    """
    Calculates the atmospheric absorption as the difference between
    the radiative imbalance at the top of the atmosphere and the surface.


    """
    varialbe_pairs = {"ERFtsw": "ERFsurfsw", "ERFtswcs": "ERFsurfswcs","ERFt":"ERFsurf","ERFtcs":"ERFsurfcs"}
    variable_toa = delta_rad_toa.name
    variable_surf = delta_rad_surf.name
    units = delta_rad_toa.units
    if varialbe_pairs[variable_toa] != variable_surf:
        raise ValueError(
            f"The combination {variable_toa} and {variable_surf} is invalid"
        )

    attrs = {
        "ERFtsw": {
            "variable_name": "atmabsSW",
            "long_name": "Atmospheric absorbtion of short wave radiation.",
            "units": units,
        },
        "ERFtswcs": {
            "variable_name": "atmabsSWcs",
            "long_name": "Clear sky atmospheric absorbtion of short wave radiation.",
            "units": units,
        },
        "ERFt": {
            "variable_name": "atmabs",
            "long_name": "Total atmospheric absorbtion",
            "units": units,
        },
        "ERFtcs": {
            "variable_name": "atmabscs",
            "long_name": "Total atmospheric absorbtion assuming clear sky.",
            "units": units,
        },
    }

    atm_abs = delta_rad_toa - delta_rad_surf
    atm_abs = atm_abs.rename(attrs[variable_toa]["variable_name"])
    atm_abs.attrs = {**atm_abs.attrs, **attrs[variable_toa]}

    return atm_abs
