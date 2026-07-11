
import os
import ast
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from metpy.calc import windchill
from metpy.units import units
from cartopy.io.shapereader import Reader
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely import contains_xy
from datetime import datetime, timezone
import json
import configparser
from PIL import Image
from pyproj import Transformer
from shapely.ops import transform as shapely_transform

from danilib import f_settaggio_db_arpal
connessione = f_settaggio_db_arpal()

plt.rc('font', weight='normal', size=6)

os.chdir('/run/media/daniele.carnevale/Daniele2TB/repo/campi2D_obs')
# os.chdir('/media/daniele/Daniele2TB/repo/campi2D_obs')

from funzioni import _hex_to_rgb
from funzioni import f_interp

regioni = Reader('./shapefile/gadm41_ITA_shp/gadm41_ITA_1.shp')

# %%
config = configparser.ConfigParser()
config.read('./config.ini')

area = ast.literal_eval(config.get('COMMON', 'area'))
R_TERRA = 6378137.0  # raggio sferico Web Mercator (EPSG:3857)
cartella_destinazione = f"{config.get('COMMON', 'cartella_destinazione')}/windchill2D_obs"
freq = config.get('COMMON', 'freq')

ds_orog_lsm = xr.open_dataset('./moloch_domain_orogr_lsm.grib2', engine='cfgrib')
crs_moloch = ccrs.RotatedPole(pole_longitude=9, pole_latitude=135.000004, central_rotated_longitude=8.634001)

######################
######################
######################

adesso_0_UTC = pd.to_datetime(datetime.now(timezone.utc)).tz_localize(None).round(freq)

# lista_tempi = [adesso_0_UTC - pd.Timedelta(freq)]
lista_tempi = pd.date_range('2026-06-28 00:00', adesso_0_UTC, freq=freq)

albero_3857 = None

for tempo in lista_tempi:
    print(tempo)

    cartella_file = f"{cartella_destinazione}/{tempo.strftime('%Y/%m/%d')}"
    nome_base = f"windchill2D_obs_{tempo.strftime('%Y-%m-%d_%H%M')}"
    if os.path.exists(f'{cartella_file}/{nome_base}.png') and not config.getboolean('COMMON', 'sovrascrivi'):
        print('Esiste già il file. Esco.\n')
        continue

    query_TMEAN = f"""
    SELECT
        TO_CHAR(data.dtrf, 'YYYY-MM-DD HH24:MI:SS') AS tempo,
        anag.code,
        anag.lon/1e5 AS lon,
        anag.lat/1e5 AS lat,
        anag.elev AS elev,
        anag.name AS name,
        tempm/10 AS TMEAN
    FROM
        data
    JOIN
        anag ON data.code = anag.code
    WHERE
        tempm IS NOT NULL
        AND data.dtrf = TO_DATE('{tempo:%Y%m%d%H%M}', 'YYYYMMDDHH24MI')
    ORDER BY
        data.code
    """
    
    query_WIND = f"""
    SELECT
        TO_CHAR(data.dtrf, 'YYYY-MM-DD HH24:MI:SS') AS tempo,
        anag.code,
        anag.lon/1e5 AS lon,
        anag.lat/1e5 AS lat,
        anag.elev AS elev,
        anag.name AS name,
        tempm/10 as TMEAN,
        wspdm/10*3.6 as WIND
    FROM
        data
    JOIN
        anag ON data.code = anag.code
    WHERE
        wspdm IS NOT NULL
        AND tempm IS NOT NULL
        AND data.dtrf = TO_DATE('{tempo:%Y%m%d%H%M}', 'YYYYMMDDHH24MI')
    ORDER BY
        data.code
    """

    print('Query della temperatura...')
    df_obs_TMEAN = pd.read_sql(query_TMEAN, con=connessione).dropna()
    print('Query del vento...')
    df_obs_WIND = pd.read_sql(query_WIND, con=connessione).dropna()

    df_obs_TMEAN["theta_TMEAN"] = df_obs_TMEAN["TMEAN"].add(df_obs_TMEAN["ELEV"] * float(config.get('WINDCHILL2D_OBS', 'lapse_rate_T')), axis=0)

    ######################
    ######################
    ######################
    
    ds_orog_lsm = ds_orog_lsm.where(
        (ds_orog_lsm.longitude >= area[0]) & (ds_orog_lsm.longitude <= area[1]) &
        (ds_orog_lsm.latitude  >= area[2]) & (ds_orog_lsm.latitude  <= area[3]),
        drop=True
    )
    
    """ Prima riportavo tutto a theta
    # print('Interpolo TMEAN_grigliata_h...')
    # TMEAN_grigliata_h = f_interp(df_obs_TMEAN['TMEAN'], df_obs_TMEAN['LAT'], df_obs_TMEAN['LON'], ds_orog_lsm)
    print('Interpolo THETAMEAN_grigliata_sfc...')
    THETAMEAN_grigliata_sfc = f_interp(df_obs_TMEAN['theta_TMEAN'], df_obs_TMEAN['LAT'], df_obs_TMEAN['LON'], ds_orog_lsm)
    print('Interpolo TMEAN_grigliata_h_nuova...')
    TMEAN_grigliata_h_nuova = THETAMEAN_grigliata_sfc - float(config.get('WINDCHILL2D_OBS', 'lapse_rate_T')) * ds_orog_lsm.mterh.values[::-1, :]
    
    print('Interpolo WIND_grigliata_h...')
    WIND_grigliata_h = f_interp(df_obs_WIND['WIND'], df_obs_WIND['LAT'], df_obs_WIND['LON'], ds_orog_lsm)
    """
    
    """ Adesso vado dritto con il Kriging con l'orografia """
    TMEAN_grigliata_h_nuova = f_interp(df_obs_TMEAN['TMEAN'], df_obs_TMEAN['LAT'], df_obs_TMEAN['LON'], ds_orog_lsm)
    WIND_grigliata_h = f_interp(df_obs_WIND['WIND'], df_obs_WIND['LAT'], df_obs_WIND['LON'], ds_orog_lsm)
    
    print('Calcolo WC...')
    wc = windchill(TMEAN_grigliata_h_nuova * units.degC, WIND_grigliata_h * units('m/s'), face_level_winds=False, mask_undefined=True)
    wc = np.ma.filled(wc.magnitude, np.nan)
    wc = np.where(ds_orog_lsm.lsm.values[::-1, :] < 1, np.nan, wc)
    wc = np.where(wc > -5, np.nan, wc)

    # %% Plot WC
    print('Plot...')

    for r in regioni.records():
        if r.attributes['NAME_1'] == 'Liguria':
            liguria = r.geometry
    
    mask = contains_xy(
        liguria,
        ds_orog_lsm.longitude.values[::-1, :],
        ds_orog_lsm.latitude.values[::-1, :]
    )
    
    wc = np.where(mask, wc, np.nan)
    # wc = np.where(mask, wc, 1)

    livelli = np.arange(-20, -3, 1)

    colori = [
        '#ffffff',
        ####
        '#00ffff',
        '#00e5e5',
        '#00cccc',
        '#00b2b2',
        '#009999',
        ####
        '#0000ff',
        '#0000e5',
        '#0000cc',
        '#0000b2',
        '#000099',
        ####
        '#800080',
        '#730073',
        '#660066',
        '#590059',
        '#4c004c',
        ####
        '#FFC0CB',
        ][::-1]
    
    cmap = mcolors.ListedColormap(colori[1:])
    cmap.set_under(colori[0])
    norm = mcolors.BoundaryNorm(livelli, cmap.N)
    
    ### Scommenta per vedere il plot
    
    # from funzioni import f_plot_coste
    # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    # f_plot_coste(ax, aree, regioni)
    # pcm = ax.contourf(
    #         ds_orog_lsm.longitude.values[::-1, :],
    #         ds_orog_lsm.latitude.values[::-1, :],
    #         wc,
    #         levels=livelli,
    #         cmap=cmap,
    #         norm=norm,
    #         extend='min',
    #         transform=ccrs.PlateCarree()
    #     )
    # ax.set_title(f'wc {str(tempo)}')
    # cbar = fig.colorbar(pcm, ax=ax, shrink=0.30, pad=0.02)
    # cbar.set_ticks([-5, -10, -15, -20])
    # cbar.ax.tick_params(which='minor', length=0)
    # plt.show()
    # plt.close()
    # sss

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon2D = ds_orog_lsm.longitude.values[::-1, :]
    lat2D = ds_orog_lsm.latitude.values[::-1, :]
    x_pts, y_pts = transformer.transform(lon2D.ravel(), lat2D.ravel())

    valid = ~np.isnan(wc.ravel())
    
    res = 500
    west, east = x_pts.min(), x_pts.max()
    south, north = y_pts.min(), y_pts.max()
    width_3857 = int(np.ceil((east - west) / res))
    height_3857 = int(np.ceil((north - south) / res))

    x_grid_3857 = west + (np.arange(width_3857) + 0.5) * res
    y_grid_3857 = north - (np.arange(height_3857) + 0.5) * res
    X_grid_3857, Y_grid_3857 = np.meshgrid(x_grid_3857, y_grid_3857)

    if valid.sum() < 3:
        print(f'Troppo pochi punti validi ({valid.sum()}) per {tempo}, produco immagine vuota.\n')
        banda_3857 = np.full(X_grid_3857.shape, np.nan)
        # banda_3857 = np.full(X_grid_3857.shape, 1)
    else:
        if albero_3857 is None:
            albero_3857 = cKDTree(np.column_stack((x_pts, y_pts)))

        _, idx_3857 = albero_3857.query(
            np.column_stack((X_grid_3857.ravel(), Y_grid_3857.ravel()))
        )
        banda_3857 = wc.ravel()[idx_3857].reshape(X_grid_3857.shape)

    liguria_3857 = shapely_transform(transformer.transform, liguria)
    mask_3857 = geometry_mask(
        [liguria_3857],
        out_shape=banda_3857.shape,
        transform=from_bounds(west, south, east, north, width_3857, height_3857),
        invert=True
    )
    banda_3857 = np.where(mask_3857, banda_3857, np.nan)
    # banda_3857 = np.where(mask_3857, banda_3857, 1)

    lon_3857 = np.degrees(x_grid_3857 / R_TERRA)
    lat_3857 = np.degrees(2 * np.arctan(np.exp(y_grid_3857 / R_TERRA)) - np.pi / 2)

    COLORI_RGB = np.array([_hex_to_rgb(c) for c in ['#000000'] + colori], dtype=np.uint8)

    mancanti = np.isnan(banda_3857)
    idx = np.searchsorted(livelli, banda_3857, side="right")
    idx = np.clip(idx, 0, len(COLORI_RGB) - 1)
    rgb = COLORI_RGB[idx]
    alpha = np.where(idx == 0, 0, 255).astype(np.uint8)
    alpha[mancanti] = 0
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)

    os.makedirs(cartella_file, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(f"{cartella_file}/{nome_base}.png")

    with open(f"{cartella_file}/{nome_base}.json", "w") as f:
        json.dump({
            "bounds": [
                [float(lat_3857[0]), float(lon_3857[0])],
                [float(lat_3857[-1]), float(lon_3857[-1])],
            ],
        }, f)
        
    # sss
    
# %% Plot della colorbar

# import matplotlib.colors as mcolors
# import matplotlib.patheffects as path_effects


# livelli = np.arange(-20, -4, 1)
# labels = [str(x) if x % 5 == 0 else '' for x in livelli]
# colori = [
#     '#00ffff',
#     '#00e5e5',
#     '#00cccc',
#     '#00b2b2',
#     '#009999',
    
#     '#0000ff',
#     '#0000e5',
#     '#0000cc',
#     '#0000b2',
#     '#000099',
    
#     '#800080',
#     '#730073',
#     '#660066',
#     '#590059',
#     '#4c004c',
    
#     '#FFC0CB',
#     ][::-1]

# cmap = mcolors.ListedColormap(colori[1:])
# cmap.set_under(colori[0])
# norm = mcolors.BoundaryNorm(livelli, cmap.N)

# ###################

# fig, ax = plt.subplots(figsize=(10, 0.3))

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])

# cbar = plt.colorbar(
#     sm,
#     cax=ax,
#     orientation="horizontal",
#     extend="min"
# )

# # niente ticks
# cbar.ax.set_xticks([])                 # major ticks OFF
# cbar.ax.set_xticks([], minor=True)     # minor ticks OFF
# cbar.ax.tick_params(which='both', length=0)
# cbar.ax.minorticks_off()

# label_fontsize = 9
# unit_fontsize = 11

# # label valori (bold)
# for i, (val, lab) in enumerate(zip(livelli, labels)):
#     x = i / (len(livelli) - 1)
#     cbar.ax.text(
#         x, -0.25, lab,
#         transform=cbar.ax.transAxes,
#         ha='center',
#         va='top',
#         fontsize=label_fontsize,
#         fontweight='bold',
#         color='black',
#         path_effects=[
#             path_effects.withStroke(linewidth=3, foreground="white")
#         ]
#     )

# # unità a destra (bold)
# cbar.ax.text(
#     1.02, 0.5, "°C",
#     transform=cbar.ax.transAxes,
#     ha='left',
#     va='center',
#     fontsize=unit_fontsize,
#     fontweight='bold',
#     color='black',
#     path_effects=[
#         path_effects.withStroke(linewidth=3, foreground="white")
#     ]
# )

# # estetica pulita
# cbar.outline.set_visible(True)
# cbar.outline.set_edgecolor("black")
# cbar.outline.set_linewidth(1.0)
# fig.patch.set_alpha(0)
# ax.patch.set_alpha(0)

# plt.savefig(
#     "./../MeteoBricchi/static/icone/colorbar_windchill2D_obs.png",
#     dpi=600,
#     transparent=True,
#     bbox_inches="tight",
#     pad_inches=0.1
# )

# plt.show()
# plt.close()
    
print('\n\nDone')
