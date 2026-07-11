
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.interpolate import griddata
from pykrige.uk import UniversalKriging


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def f_interp(punti, lat, lon, ds_orog_lsm):
    griglia_orog = ds_orog_lsm.mterh.values[::-1, :]
    lat2D = ds_orog_lsm.latitude.values[::-1, :]
    lon2D = ds_orog_lsm.longitude.values[::-1, :]

    valid_src = ~np.isnan(griglia_orog.ravel())
    
    elev_at_points = griddata(
        (lon2D.ravel()[valid_src], lat2D.ravel()[valid_src]),
        griglia_orog.ravel()[valid_src],
        (lon.values, lat.values),
        method='linear'
    )
    nan_mask = np.isnan(elev_at_points)
    if nan_mask.any():
        elev_at_points[nan_mask] = griddata(
            (lon2D.ravel()[valid_src], lat2D.ravel()[valid_src]),
            griglia_orog.ravel()[valid_src],
            (lon.values[nan_mask], lat.values[nan_mask]),
            method='nearest'
        )
        
    uk = UniversalKriging(
        lon.values,
        lat.values,
        punti.values,
        variogram_model='linear',
        drift_terms=['specified'],
        specified_drift=[elev_at_points],
    )
    
    valid_tgt = ~np.isnan(griglia_orog.ravel())
    z_flat = np.full(lon2D.size, np.nan)
    ss_flat = np.full(lon2D.size, np.nan)
    z_flat[valid_tgt], ss_flat[valid_tgt] = uk.execute(
        'points',
        lon2D.ravel()[valid_tgt],
        lat2D.ravel()[valid_tgt],
        specified_drift_arrays=[griglia_orog.ravel()[valid_tgt]]
    )
    
    return z_flat.reshape(lat2D.shape)


def f_plot_coste(ax, area, regioni):
    ax.coastlines(resolution='10m', lw=0.75)
    ax.add_feature(cfeature.BORDERS, lw=0.75)
    ax.set_extent(area, crs=ccrs.PlateCarree())
    for r in regioni.records():
       if r.attributes['NAME_1'] == 'Liguria':
           liguria = r.geometry
           ax.add_geometries([liguria], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.5)
