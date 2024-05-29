!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday November 4 17:08:18 2021

@author: Ada Gjermundsen

Reading routines for NorESM raw and cmorized output
"""
from __future__ import annotations

import glob
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np

warnings.simplefilter("ignore")
import xarray as xr


def fx_files(self, var="areacella", path_to_data="/projects/NS9034K/CMIP6/"):
    """ """
    if var in ["areacella", "orog", "sftgif", "sftlf"]:
        fx = "fx"
    if var in ["areacello", "deptho", "thkcello", "basin", "volcello", "sftof"]:
        fx = "Ofx"
    if var in ["thkcello", "volcello"]:
        # This is not a very good solution, but it will work for NorESM2 though
        # but say we output data on native and latlon grid, then we will have gn and gr for the same variable...
        self.gridlabel = "gr"
    else:
        self.gridlabel = "gn"
    fxpath = (
        path_to_data
        + "/"
        + self.activity_id
        + "/"
        + self.institute
        + "/"
        + self.name
        + "/piControl/"
        + self.realiz
        + "/"
        + fx
        + "/"
        + var
        + "/"
        + self.gridlabel
        + "/latest/"
    )
    self.fxfile = (
        fxpath
        + var
        + "_"
        + fx
        + "_"
        + self.name
        + "_piControl_"
        + self.realiz
        + "_"
        + self.gridlabel
        + ".nc"
    )

def make_filelist_cmor(
    self,
    var,
    activity_id="CMIP",
    realm="Amon",
    grid="gn",
    path_to_data="/projects/NS9034K/CMIP6/",
):
    """
    Function for providing a list of all files in order to read a cmorized variable for a given experiment.
    There is usually only one grid for the atmosphere; gn (grid native)
    For the ocean and sea-ice files there are often both native (gn) and regridded files (gr, grz, gm). If not gn is used it is necessary state which grid label is wanted in self.grid_label_ocean

    Parameters
    ----------
    self: model object
    var :          str, name of variable
    realm:         str, e.g. Amon, AERmon, Omon, SImon
    grid:          str, e.g. gn, gr, grz
    activity_id :  str, which MIP the experiment belongs to. Default is 'CMIP'
    path_to_data : str, path to the data folders. Default is '/projects/NS9034K/CMIP6/'.

    Returns
    -------
    Sets a list if filename(s) as an attribute of the model object; self.filenames

    """
    import glob

    self.variable = var
    self.activity_id = activity_id
    self.realm = realm
    self.grid = grid
    # This is kind of a selection function since the ocean grid names are kind of variable, can also have multiple grids for one given variable:
    grid_label_paths = sorted(
        glob.glob(
            path_to_data
            + "/"
            + self.activity_id
            + "/"
            + self.institute
            + "/"
            + self.name
            + "/"
            + self.expid
            + "/"
            + self.realiz
            + "/"
            + self.realm
            + "/"
            + self.variable
            + "/*"
        )
    )
    grid_labels = [
        grid_label_paths[i].split("/")[-1] for i in range(0, len(grid_label_paths))
    ]
    if not grid_labels:
        print(
            "No grid labels found for the given path: %s. Please chekc variable name and path"
            % (
                path_to_data
                + "/"
                + self.activity_id
                + "/"
                + self.institute
                + "/"
                + self.name
                + "/"
                + self.expid
                + "/"
                + self.realiz
                + "/"
                + self.realm
                + "/"
                + self.variable
                + "/*"
            )
        )
        self.filenames = None
    else:
        if self.grid in grid_labels:
            self.gridlabel = self.grid
        else:
            self.gridlabel = grid_labels[0]
            print(
                "\nPLEASE NOTE! Grid label is set to %s for variable %s. If not correct, please reset the grid info. Available grid(s): "
                % (self.gridlabel, self.variable)
                + " ".join(map(str, grid_labels))
            )
        self.path = (
            path_to_data
            + "/"
            + self.activity_id
            + "/"
            + self.institute
            + "/"
            + self.name
            + "/"
            + self.expid
            + "/"
            + self.realiz
            + "/"
            + self.realm
            + "/"
            + self.variable
            + "/"
            + self.gridlabel
            + "/"¨
        )
        # We need to fix the multiple version challenge. Not all files are necessarily located in the 'latest' folder
        versions = sorted(glob.glob(self.path + "*"))
        if versions:
            fnames = sorted(
                glob.glob(
                    versions[0]
                    + "/"
                    + self.variable
                    + "_"
                    + self.realm
                    + "_"
                    + self.name
                    + "_"
                    + self.expid
                    + "_"
                    + self.realiz
                    + "_"
                    + self.gridlabel
                    + "*.nc" #original: "_*.nc"
                )
            )
        else:
            fnames = None
        if len(versions) > 1:
            for version in versions[1:]:
                files = sorted(
                    glob.glob(
                        version
                        + "/"
                        + self.variable
                        + "_"
                        + self.realm
                        + "_"
                        + self.name
                        + "_"
                        + self.expid
                        + "_"
                        + self.realiz
                        + "_"
                        + self.gridlabel
                        + "*.nc" #original: "_*.nc"
                    )
                )
                for file in files:
                    if versions[0] + "/" + file.split("/")[-1] not in fnames:
                        fnames.append(file)

        if fnames:
            if (self.name == "NorESM2-MM" and self.realiz == "r3i1p1f1" and self.expid == "historical"):
                # There exists one extra file for all cmorized variables for NorESM2-MM, member r3i1p1f1.
                # # Need to remove the file from the filelist            
                temp = path_to_data + "/" + self.activity_id + "/" + self.institute + "/" + self.name + "/" + self.expid + "/" + self.realiz + "/" + self.realm + "/" + self.variable + "/" + self.gridlabel + "/latest/" + self.variable + "_" + self.realm + "_" + self.name + "_" + self.expid + "_" + self.realiz + "_" + self.gridlabel + "_186001-186105.nc"
                if temp in fnames:
                    fnames.remove(temp)
                temp = path_to_data + "/" + self.activity_id + "/" + self.institute + "/" + self.name + "/" + self.expid + "/" + self.realiz + "/" + self.realm + "/" + self.variable + "/" + self.gridlabel + "/latest/" + "rldscs_CFday_NorESM2-MM_historical_r3i1p1f1_gn_19400101-19430623.nc"
                if temp in fnames:
                    fnames.remove(temp)
            if len(fnames) > 1 and self.activity_id[-3:] == "mon":
                # test that the files contained in the filelist covers all years in consecutive order
                # only works for monthly files
                fnames = sorted(fnames, key=lambda x: extract_number(x))
                checkConsecutive(fnames)
            self.filenames = fnames
            # print('\n Final list of filenames:')
            # print(self.filenames)
        if not fnames:
            self.filenames = None
        #if not fnames:
        #    raise Exception(
        #        "Variable %s not prestent in output folder for model %s\n"
        #        % (self.variable, self.name)
        #    )


def extract_number(string):
    return string.split("_")[-1]


def extract_dates(string):
    return string.split("_")[-1].split(".")[0]


def checkConsecutive(fnames):
    sorteddates = [extract_dates(x) for x in fnames]
    for i in range(1, len(sorteddates)):
        if (
            int(sorteddates[i].split("01-")[0])
            != int(sorteddates[i - 1].split("-")[1][:-2]) + 1
        ):
            print(fnames)
            raise Exception(
                "NOTE! The files are not in consecutive order. Please check directory"
                 )


def read_noresm_cmor(
    model,
    varlist,
    realm="Amon",
    grid="gn",
    dim="time",
    transform_func=None,
    path_to_data="/projects/NS9034K/CMIP6/",
):
    """
    This function reads cmorized data from NorESM.
    Please NOTE: This is not a general CMIP6 reading routine as it contains NorESM specific bug fixes
    which may mess up model results from other CMIP6 models.

    Parameters
    ----------
    model :          python object, with experiment details as attributes (generated by class Modelinfo )
    varlist:         list, list of variable names which will be read and loaded into one xarray.Dataset
    realm:           str, Realm: e.g. Amon, AERmon, CFmon, Omon, SImon
    grid :           str, which grid resolution should be used.  e.g. 'gn', 'gr', 'gm', 'grz'
    gn: native grid, 'gr': regridded somehow - not obvious
                     The grid is not really needed unless you want to specify one particular grid (out of several options)
    dim:             str, concatenate files by this dimension (if there are several files for one single variable) . Default is 'time'
    transform_func:  python function(s), which will be applied to all variables in the variable list

    Returns
    -------
    ds_out : xarray.Dataset

    """
    
    ds_out = None
    for var in varlist:
        make_filelist_cmor(
            model,
            var,
            activity_id=model.activity_id,
            realm=realm,
            grid=grid,
            path_to_data=path_to_data,
        )
        if isinstance(model.filenames,list):
            # NOTE! if the reading crashes due to memory issues you may add chunks, i.e. parallel=True, chunks={"time":12}
            with xr.open_mfdataset(
                model.filenames,
                concat_dim=dim,
                combine="nested",
                data_vars="minimal",
                coords="minimal",
                join="override",
                compat="override", #added this because of the merge problem with sea ice (ANDREA)
            ) as ds:
                if "height" in ds.variables:
                    # Datasets for variables @ a given height e.g. 2 m temperature, 10 m surface temperature
                    # contain a 'height' variable which causes problems when combining datasets.
                    # i.e. xarray.core.merge.MergeError: conflicting values for variable 'height' on objects to be combined.
                    # The height is not really used for anything, so it is safe to drop it
                    ds = ds.drop("height")
                if "lon_bnds" in ds.variables and model.name in [
                    "NorESM2-LM",
                    "NorESM2-MM",
                    "NorCPM1",
                ]:
                    if ds.lon_bnds.isel(bnds=0).values[0] == 0:
                        # many of the cmorized variables contain wrong longitude boundary values
                        # which causes problems when combining datasets.
                        # Wrong boundary values can potentially really mess up regridding as well
                        # Over write with correct boundary values
                        lon_b = np.concatenate(
                            (
                                np.array(
                                    [
                                        ds.lon[0].values
                                        - 0.5 * ds.lon.diff(dim="lon").values[0]
                                    ]
                                ),
                                0.5 * ds.lon.diff(dim="lon").values + ds.lon.values[:-1],
                                np.array(
                                    [
                                        ds.lon[-1].values
                                        + 0.5 * ds.lon.diff(dim="lon").values[-1]
                                    ]
                                ),
                            )
                        )
                        lon_b = np.reshape(
                            np.concatenate([lon_b[:-1], lon_b[1:]]), [2, len(ds.lon.values)]
                        ).T
                        ds["lon_bnds"] = xr.DataArray(
                            lon_b, dims=("lon", "bnds"), coords={"lon": ds.lon}
                        )
                if "sector" in ds.variables:
                    # the problem here is that we operate with a very variable number of sting lenghts (from 70 to 1100)
                    # make them identical by getting rid of white spaces
                    ds["sector"] = ds.sector.str.strip()
                # transform_func should do some sort of selection on aggregation
                if transform_func is not None:
                    ds = transform_func(ds)
                if isinstance(ds_out, xr.Dataset):
                    ds_out = xr.merge([ds, ds_out],compat="override")
                else:
                    ds_out = ds
        else:
            print("No files found for variable %s \n"%var)
            continue
    return ds_out

def date_from_path(path: Path, format: str = "%Y-%m") -> date:
    """This function finds and returns the date in the NorESM raw output files

    Parameters
    ----------
    path :         Path object, contains one file path
    format:        str, format of date in file name. Default is year-month -> "%Y-%m"

    Returns
    -------
    the date of the file as a datetime.date object
    """
    date = path.stem.split(".")[
        -1
        ][:7]  # Returns the date part of the filename e.g. 0001-01
    return datetime.strptime(date, format).date()

def make_filelist_raw(
    expid: str,
    path: Path | str = "/projects/NS9560K/noresm/cases/",
    *,
    component: str = "atmos",
    first_year: int | None = None,
    last_year: int | None = None,
    month: int | None = None,
) -> list[Path]:

    """This function finds and returns the paths of NorESM raw output monthly files either as:
    - all files between first and last year
    - files for a given month (e.g. January files) between first and last year

    Parameters
    ----------
    expid:        str, name of experiment
    path:         str or Path, the path to where the experiment is stored
    component:    str, name of model component; atmos, ocean, land, seaice. Default is  "atmos",
    first_year:   int, first year of data to be read
    last_year:    int, last year of data to be read
    month:        int or None, None: includes all months in paths, int: only include files with months = month. Default is None.

    Returns
    -------
    paths :       list, contains all file paths as Path objects
    """
    if isinstance(path, str):
        path = Path(path)

    pattern = dict(
        atmos=f"{expid}/atm/hist/{expid}.cam.h0.*.nc",
        ocean=f"{expid}/ocn/hist/{expid}.*.hm.*.nc",
        land=f"{expid}/lnd/hist/{expid}.clm2.h0.*.nc",
        seaice=f"{expid}/ice/hist/{expid}.cice.h.*.nc",
    )[component]
    paths = sorted(path.glob(pattern))
    assert paths, f"no files found for {expid} '/' {component}"
    if first_year is not None:
        paths = [
            path for path in paths if first_year <= date_from_path(path).year]
    else:
        print(date_from_path(paths[0]))
        print(date_from_path(paths[0]).year)
        first_year = date_from_path(paths[0]).year
        print(first_year)
        print(type(first_year))
    if last_year is not None:
        paths = [
            path for path in paths if last_year >= date_from_path(path).year]
    else:
        last_year = date_from_path(paths[-1]).year
    assert (
        paths
    ), f"no files found for {expid} between the years: {first_year}--{last_year}"

    if month is not None:
        paths = [path for path in paths if date_from_path(path).month == month]
        assert (
            paths
        ), f"no files found for {expid} between {first_year}--{last_year} for month nr.: {month}"
    return paths

def read_noresm_raw(
    fnames: list[str | Path],
    *,
    data_vars: str | list[str] = "minimal",
    preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
    parallel: bool = False,
) -> xr.Dataset:
    """This function finds and returns the paths of NorESM raw output files either as:

    Parameters
    ----------
    fnames :       list or path object, the list contains all file paths as Path objects
    data_vars :    str or list, list of variables to be read from file or "minimal". Default is "minimal"
    preprocess:    python function(s), functions to be used on all variables in the Dataset . Default is None.
    parallel:      bool, use dask to read the data in paralell. Can slow things down quite a bit. Default is False.

    Returns
    -------
          :        xarray.Dataset
    """
    # if the reading crashes due to memory issues you may add chunks, i.e. parallel=True, chunks={"time":12}
    return xr.open_mfdataset(
        fnames,
        concat_dim="time",
        combine="nested",
        data_vars=data_vars,
        coords="minimal",
        compat="override",
        preprocess=preprocess,
        parallel=parallel,
    )

def make_filelist_raw_highres(
    expid: str,
    path: str | Path,
    hnumber: str,
    *,
    component: str = "atmos",
    first_year: int | None = None,
    last_year: int | None = None,
    month: int | None = None,
) -> list[Path]:
    """This function finds and returns the paths of NorESM raw output files:
    - for a given month
    - or all files between first and last year
    Parameters
    ----------
    expid:        str, name of experiment
    path:         str or Path, the path to where the experiment is stored
    hnumber:      str, file output number (which is not really consistent in NorESM output)
    component:    str, name of model component; atmos, ocean, land, seaice. Default is  "atmos",
    first_year:   int, first year of data to be read
    last_year:    int, last year of data to be read
    month:        int or None, None: includes all months in paths, int: only include files with months = month. Default is None.

    Returns
    -------
    paths :       list, contains all file paths as Path objects
    """

    if isinstance(path, str):
        path = Path(path)

    pattern = dict(
        atmos=f"{expid}/atm/hist/{expid}.cam.{hnumber}.*.nc",
        ocean=f"{expid}/ocn/hist/{expid}.*.{hnumber}.*.nc",
        land=f"{expid}/lnd/hist/{expid}.clm2.{hnumber}.*.nc",
        seaice=f"{expid}/ice/hist/{expid}.cice.{hnumber}.*.nc",
    )[component]

    paths = sorted(path.glob(pattern))
    assert paths, f"no files found for {expid} '/' {component}"

    def date_from_path_highres(path: Path, format: str = "%Y-%m") -> date:
        """This function finds and returns the date in the NorESM raw output files

        Parameters
        ----------
        path :         Path object, contains one file path
        format:        str, format of date in file name. Default is year-month -> "%Y-%m"

        Returns
        -------
        the date of the file as a datetime.date object
        """
        date = path.stem.split(".")[-1][
            :7
        ]  # Returns the date part of the filename e.g. 0001-01
        return datetime.strptime(date, format).date()

    paths = [
        path
        for path in paths
        if first_year <= date_from_path_highres(path).year <= last_year
    ]
    print(paths)
    assert (
        paths
    ), f"no files found for {expid} between the years: {first_year}--{last_year}"

return paths


class Modelinfo:
    """
    Sets the details of the model experiment
    """

    def __init__(
        self,
        name="NorESM2-LM",
        institute="NCC",
        activity_id="CMIP",
        expid="piControl",
        realiz=["r1i1p1f1"],
        branchtime_year=0,
    ):
        """
        The attributes need to be general and not file specific thus detail like Amon, grid label should not be included

        Attributes
        ----------
        name :        str, name of the CMIP model - typically the Source ID
        institute :   str, Institution ID
        activity_id:  str, activity ID for experiment
        expid :       str, Experiment ID, e.g. piControl, abrupt-4xCO2
        realiz :      str, variant labels saying something about which realization (ensemble member), initial conditions, forcings etc.
                      e.g. 'r1i1p1f1'
        branchtime_year : int, when simulation was branched off from parent.
                          Useful when anomalies are considered e.g. abrupt-4xCO2, historical
                          then you only consider data from piControl for the corresponding period
                          e.g. xarray.DataArray(year = slice(model.branchtime_year,None))

        """
        self.name = name
        self.institute = institute
        self.activity_id = activity_id
        self.expid = expid
        self.realiz = realiz
        self.branchtime_year = branchtime_year








                
                    

            



