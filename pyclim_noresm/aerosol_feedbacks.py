import xarray as xr
from .general_util_funcs import global_avg

def partialDC_partialDT(deltaT: xr.DataArray, deltaC: xr.DataArray
                            , area_cello: xr.DataArray=None, 
                            sum_deltaC: bool = True,
                            per_year: bool=False, 
                            time_average_first=False):
    
    deltaC_name = deltaC.name

    if isinstance(deltaT, xr.Dataset):
        deltaT = deltaT[deltaT.variable_id]
    if isinstance(deltaC, xr.Dataset):
        deltaC = deltaC[deltaC.variable_id]

    if 'year' in deltaT.dims:
        deltaT = deltaT.rename({'year':'time'})
    if 'year' in deltaC.dims:
        deltaC = deltaC.rename({'year':'time'})

    if time_average_first:
        with xr.set_options(keep_attrs=True):
            deltaC = deltaC.mean(dim='time')
            deltaT = deltaT.mean(dim='time')

    if per_year:
        with xr.set_options(keep_attrs=True):
            deltaC = deltaC*365*24*60*60
            deltaC.attrs['units'] = '{} year-1'.format(' '.join(deltaC.attrs['units'].split(' ')[:-1]))
    
    if area_cello:
        with xr.set_options(keep_attrs=True):
            deltaC = deltaC*area_cello['cell_area']
            deltaC.attrs['units'] = deltaC.attrs['units'].split(' ')[0]+ ' ' + deltaC.attrs['units'].split(' ')[-1]

    with xr.set_options(keep_attrs=True):
        deltaT = global_avg(deltaT)
        if sum_deltaC:
            deltaC = deltaC.sum(dim=['lon','lat'])
        else:
            deltaC = global_avg(deltaC)
    pDC_pDT = deltaC/deltaT.values
    pDC_pDT.attrs['units'] = '{}/{}'.format(deltaC.attrs['units'],deltaT.attrs['units'])
    pDC_pDT.attrs['long_name'] = '{} divided by {}'.format(deltaT.attrs['long_name'],deltaC.attrs['long_name'])
    pDC_pDT = pDC_pDT.rename('{}_{}'.format('deltaT', deltaC_name))

    return pDC_pDT, deltaC, deltaT

def partialDF_partialDC(forcing: xr.DataArray, 
                        deltaC: xr.DataArray, 
                        area_cello: xr.DataArray=None,
                        per_year: bool =True,
                        sum_deltaC: bool = True,
                        time_average_first=False):
    """
    Calculates the change in TOA imbalance as per change in emission
    
    Parameters:
    ----------
        Forcing: xr.DataArray     forcing at the TOA, yearly average (W m-2)
        deltaC: xr.DataArray      change in some quantity between experiment and control (mass ) 
        area_cello : xr.Dataset   area of each gridcell
        per_year : bool           change time unit to year-1

    Returns:
    --------
        pDF_pDC : forcing per emission change

    """
    
    if isinstance(forcing, xr.Dataset):
        forcing = forcing[forcing.variable_id]
    if isinstance(deltaC, xr.Dataset):
        deltaC = deltaC[deltaC.variable_id]

    forcing_name = forcing.name
    deltaC_name = deltaC.name

    if 'year' in forcing.dims:
        forcing = forcing.rename({'year':'time'})
    if 'year' in deltaC.dims:
        deltaC = deltaC.rename({'year':'time'})

    if time_average_first:
        with xr.set_options(keep_attrs=True):
            deltaC = deltaC.mean(dim='time')
            
            forcing = forcing.mean(dim='time')


    if per_year:
        with xr.set_options(keep_attrs=True):
            deltaC = deltaC*365*24*60*60
            deltaC.attrs['units'] = '{} year-1'.format(' '.join(deltaC.attrs['units'].split(' ')[:-1]))

    if area_cello:
        with xr.set_options(keep_attrs=True):
            deltaC = deltaC*area_cello['cell_area'].values
            deltaC.attrs['units'] = deltaC.attrs['units'].split(' ')[0]+ ' ' + deltaC.attrs['units'].split(' ')[-1]

    with xr.set_options(keep_attrs=True):
        forcing = global_avg(forcing)
        if sum_deltaC:
            deltaC = deltaC.sum(dim=['lon','lat'])
        else:
            deltaC = global_avg(deltaC)
    pDF_pDC = forcing/deltaC.values
    pDF_pDC.attrs['units'] = '{}/{}'.format(forcing.attrs['units'], deltaC.attrs['units'])
    pDF_pDC.attrs['long_name'] = '{} divided by {}'.format(forcing.attrs['long_name'],deltaC.attrs['long_name'])
    pDF_pDC = pDF_pDC.rename('{}_{}'.format(forcing_name, deltaC_name))

    return pDF_pDC, deltaC, forcing


    