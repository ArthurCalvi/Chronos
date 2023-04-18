import geopandas as gpd
import phoenix as phx
import rasterio
from rasterio.warp import transform_geom, reproject
from rasterio.features import rasterize
from shapely.geometry import shape 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import dotenv
from datetime import datetime, timedelta
import json 
from pansharpening import pansharpen, histogram_match_pansharpen_visual
from skimage.filters import unsharp_mask
from skimage.exposure import match_histograms
import numpy as np 
import rasterio
from copy import deepcopy
dotenv.load_dotenv('/Users/arthurcalvi/Documents/PhD/Kayrros/test/phx.env')

priority = ['sentinel-2-l2a', 'sentinel-2-l1c', 'landsat-8', 'landsat-5', 'landsat-7']

url = "https://prod-catalog.internal.kayrros.org/"
auth = phx.HTTPBasicAuth(os.getenv("LDAP_USERNAME"),
                         os.getenv("LDAP_PASSWORD"))
client = phx.catalog.Client(url)

dplatform = {
    'sentinel-2a':'sentinel-2',
    'sentinel-2b':'sentinel-2',
    'landsat-5':'landsat-5',
    'landsat-7':'landsat-7',
    'landsat-8':'landsat-8',
    }

config = {
    'landsat-8': {
        # 'collection':client.get_collection('nasa-landsat-8-oli+tirs-c2-l2-t1').at('aws:s3:usgs-landsat'),
        'collection': client.get_collection('nasa-landsat-8-oli+tirs-c2-l2-t1').at('aws:proxima:usgs-landsat'),
        'instruments': ['oli', 'tirs'],
        'all_bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'SR_B6', 'ST_B10'],
        'rgb_bands': ['SR_B2', 'SR_B3', 'SR_B4'],
        'nir_bands': ['SR_B4'],
        'pan': {'collection':client.get_collection('nasa-landsat-8-oli+tirs-c2-l1-t1').at('aws:proxima:usgs-landsat'), 
                'pan_band': 'B8',
                'from':'L2SP',
                'to':'L1TP'},
        'pan_coverage': ['SR_B2', 'SR_B3', 'SR_B4'],
        'swir_bands': ['SR_B6', 'SR_B7'],
        'ndvi_bands': ['SR_B5', 'SR_B4'],
        'ndwi_bands': ['SR_B3', 'SR_B5'],
        'evi_bands': ['SR_B5', 'SR_B4', 'SR_B2'],
        'nbr_bands': ['SR_B5', 'SR_B7'], 
        'crswir_bands':['SR_B6', 'SR_B6', 'SR_B7'],
        'crswir_coeffs':[(1650-865)/(2220-865)],
        #https://www.sciencedirect.com/science/article/pii/S0034425718304139
        # 'landsat-8-to-sentinel-2' : {
        #     'SR_B2': [-0.00411, 0.977],
        #     'SR_B3': [-0.00093, 1.005],
        #     'SR_B4': [0.00094, 0.982], 
        #     'SR_B5': [-0.00029, 1.001],
        #     'SR_B6': [-0.00015, 1.001],
        #     'SR_B7': [-0.00097, 0.996], 
        # },
        'qa': ['QA_PIXEL'],
        'qa_cloud' : [22280],
        'start': datetime(2013, 1, 1),
        'end': datetime(2030, 1, 1),
        'reflectance': lambda x: (x * 2.75e-5 - 0.2),
        'resolution': {'SR_B1': 30,
                       'SR_B2': 30,
                       'SR_B3': 30,
                       'SR_B4': 30,
                       'SR_B5': 30,
                       'SR_B7': 30,
                       'SR_B6': 30,
                       'ST_B10': 30,
                       'QA_PIXEL': 30,
                       'pan': 15},
    },

    'landsat-7': {
        # 'collection':client.get_collection('nasa-landsat-7-etm-c2-l2-t1').at('aws:s3:usgs-landsat'),
        'collection': client.get_collection('nasa-landsat-7-etm-c2-l2-t1').at('aws:proxima:usgs-landsat'),
        'instruments': ['etm'],
        'all_bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'],
        'rgb_bands': ['SR_B1', 'SR_B2', 'SR_B3'],
        'nir_bands': ['SR_B4'],
        'swir_bands': ['SR_B5', 'SR_B7'],
        'pan': {'collection':client.get_collection('nasa-landsat-7-etm-c2-l1-t1').at('aws:proxima:usgs-landsat'), 
                'pan_band': 'B8',
                'from':'L2SP',
                'to':'L1TP'},
        'pan_coverage': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'],
        'ndvi_bands': ['SR_B4', 'SR_B3'],
        'ndwi_bands': ['SR_B2', 'SR_B4'],
        'evi_bands': ['SR_B4', 'SR_B3', 'SR_B1'],
        'nbr_bands': ['SR_B4', 'SR_B7'],
        'ndre_bands': ['SR_B4', 'SR_B3'],
        'crswir_bands':['SR_B4', 'SR_B5', 'SR_B7'],
        'crswir_coeffs':[(1650-835)/(2215-835)],
        'qa': ['QA_PIXEL'],
        'qa_cloud' : [5896],
        'start': datetime(2000, 1, 1),
        'end': datetime(2014, 1, 1),
        'reflectance': lambda x: (x * 2.75e-5 - 0.2),
        'resolution': {'SR_B1': 30,
                       'SR_B2': 30,
                       'SR_B3': 30,
                       'SR_B4': 30,
                       'SR_B5': 30,
                       'SR_B7': 30,
                       'ST_B6': 30,
                       'QA_PIXEL': 30, 
                       'pan': 15},
    },
    'landsat-5': {
        # 'collection':client.get_collection('nasa-landsat-5-mss+tm-c2-l2-t1').at('aws:s3:usgs-landsat'),
        'collection': client.get_collection('nasa-landsat-5-mss+tm-c2-l2-t1').at('aws:proxima:usgs-landsat'),
        'instruments': ['tm'],
        'all_bands': ['SR_T1', 'SR_T2', 'SR_T3', 'SR_T4', 'SR_T5', 'SR_T7', 'ST_T6'],
        'rgb_bands': ['SR_T1', 'SR_T2', 'SR_T3'],
        'nir_bands': ['SR_T4'],
        'swir_bands': ['SR_T5', 'SR_T7'],
        'pan': None,
        'pan_coverage': None,
        'ndvi_bands': ['SR_T4', 'SR_T3'],
        'ndwi_bands': ['SR_T3', 'SR_T5'],
        'evi_bands': ['SR_T4', 'SR_T3', 'SR_T1'],
        'nbr_bands': ['SR_T4', 'SR_T7'],
        'ndre_bands': ['SR_T4', 'SR_T3'],
        'crswir_bands':['SR_T4', 'SR_T5', 'SR_T7'],
        'crswir_coeffs':[(1650-830)/(2215-830)],
        'qa': ['QA_PIXEL'],
        'qa_cloud' : [5896],
        'start': datetime(1984, 1, 1),
        'end': datetime(2014, 1, 1),
        'reflectance': lambda x: (x * 2.75e-5 - 0.2),
        'resolution': {'SR_T1': 30,
                       'SR_T2': 30,
                       'SR_T3': 30,
                       'SR_T4': 30,
                       'SR_T5': 30,
                       'SR_T7': 30,
                       'ST_T6': 30,
                       'QA_PIXEL': 30},
    },

    'sentinel-2': {
        'collection': client.get_collection('esa-sentinel-2-msi-l2a').at('aws:proxima:sentinel-cogs'),
        'all_bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A'],
        'rgb_bands': ['B02', 'B03', 'B04'],
        'nir_bands': ['B05', 'B06', 'B07', 'B08', 'B8A'],
        'swir_bands': ['B11', 'B12'],
        'pan': None,
        'pan_coverage': None,
        'ndwi_bands': ['B03', 'B08'],
        'ndvi_bands': ['B08', 'B04'],
        'evi_bands': ['B08', 'B04', 'B02'],
        'ndre_bands': ['B09', 'B05'],
        'nbr_bands': ['B08', 'B12'],
        'crswir_bands':['B08', 'B11', 'B12'],
        'crswir_coeffs':[(1610-842)/(2190-842)],
        'qa': ['SCL'],
        'qa_cloud' : [9],
        'start': datetime(2017, 1, 1),
        'end': datetime(2030, 1, 1),
        'reflectance': lambda x: (x * 1e-4),
        'resolution': {'B01': 60,
                       'B02': 10,
                       'B03': 10,
                       'B04': 10,
                       'B05': 20,
                       'B06': 20,
                       'B07': 20,
                       'B08': 10,
                       'B09': 60,
                       'B11': 20,
                       'B12': 20}
    },
    # 'sentinel-2-l1c': {
    #     'collection': client.get_collection('esa-sentinel-2-msi-l1c').at('aws:proxima:sentinel-s2-l1c'),
    #     'all_bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A'],
    #     'rgb_bands': ['B02', 'B03', 'B04'],
    #     'nir_bands': ['B05', 'B06', 'B07', 'B08', 'B8A'],
    #     'swir_bands': ['B11', 'B12'],
    #     'ndwi': ['B03', 'B08'],
    #     'ndvi': ['B08', 'B04'],
    #     'evi': ['B08', 'B04', 'B02'],
    #     'ndre': ['B08', 'B05'],
    #     'nbr': ['B08', 'B12'],
    #     'qa': ['SCL'],
    #     'start': datetime(2015, 1, 1),
    #     'end': datetime(2017, 1, 1),
    #     'reflectance': lambda x: (x * 1e-4),
    #     'resolution': {'B01': 60,
    #                     'B02': 10,
    #                     'B03': 10,
    #                     'B04': 10,
    #                     'B05': 20,
    #                     'B06': 20,
    #                     'B07': 20,
    #                     'B08': 10,
    #                     'B09': 60,
    #                     'B11': 20,
    #                     'B12': 20}
    # }
}

#utils
def write_tif(arr, transfo, crs, direction, item, dtype='uint16', normalization=True):
    pr  = {
            'transform': transfo,
            'crs': crs,
            'width': arr.shape[2],
            'height': arr.shape[1],
            'count': arr.shape[0],
            'dtype': dtype, 
            'driver': 'GTiff'
            
        }
    
    arr = arr.clip(0,1)
    if normalization is not None:
        arr = ((arr - normalization[0]) /  (normalization[1]- normalization[0])).clip(0,1)

    if dtype == 'uint16':
        arr *= 65535
    elif dtype == 'uint8':
        arr *= 255 


    name = '{}_{}.tif'.format(item.properties['datetime'].strftime("%Y-%m-%d_%H%M%S"), item.properties['platform'])
    with rasterio.open(os.path.join(direction, name),'w', **pr) as ff:
        ff.write(arr)

def get_asset(item, sat, band, bbox):
    
    crs, transfo, arr, _ = item.assets.crop_as_array(band, bbox = bbox)

    return crs, transfo, arr 

    
def get_pansharpened(item, config, arr, transfo, geometry_buffer):
    sat = dplatform[item.properties['platform']]
    collection = config[sat]['pan']['collection']

    try :
        item_pan = collection.get_item( item.id.replace(config[sat]['pan']['from'], config[sat]['pan']['to']) )
        #cst
        metadata = json.loads(item_pan.assets.download_as_str('MTL_JSON'))
        cst = metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']
        add = float(cst['REFLECTANCE_ADD_BAND_8'])
        mult = float(cst['REFLECTANCE_MULT_BAND_8'])
        pan = item_pan.assets.crop_as_array(config[sat]['pan']['pan_band'], bbox= shape(geometry_buffer).bounds)
        pan = (mult * (pan[2].squeeze()) + add) 
        #pan-sharpening
        pansharpened = pansharpen(arr, pan, method='pca',interpolation_order=3 ,with_nir=False)
        transfo = transfo * transfo.scale((arr.shape[1] / pansharpened.shape[1]),\
                                        (arr.shape[2] / pansharpened.shape[2]))
        pansharpened = histogram_match_pansharpen_visual(pansharpened, arr)
        return pansharpened, transfo, True
    except:
        return arr, transfo, False
    
def get_sats(start, end, config, priority):
    sats = []
    for key in config:
        start_sat = config[key]['start']
        end_sat = config[key]['end']
        if (start_sat < start and end_sat > start) or (start_sat < end and end_sat > end):
            sats.append(key)

    return [x for _, x in sorted(zip(priority, sats))]

def research_items(geometry, start_date, end_date, sats, config, cc):
    filter_ = [
    phx.catalog.Field('eo:cloud_cover') <= cc, 
    phx.catalog.Field('datetime') <= end_date, 
    phx.catalog.Field('datetime') >= start_date, 
    ]

    items = []
    for sat in sats:
        items_sat = list(config[sat]['collection'].search_items(shape(geometry).envelope, filters=filter_,))
        # print(f'nbr items for {sat}: {len(items_sat)}')
        items.extend(items_sat)

    return sorted(items, key=lambda x:x.properties['datetime'])

def select_items(df, items, delta_min=20, show=False):
    df['gap'] = (df.date - df.date.shift()).dt.days
    df['sat_change'] = abs(df.sat - df.sat.shift()).astype(bool)
    df['index_correction_priority'] = -(df.sat < df.sat.shift()).astype(int)
    df['delete'] = (df['gap'] < delta_min).values 
    index = df.delete.idxmax()
    index_corrected = index + df.iloc[index].index_correction_priority
    df.drop(index=index_corrected, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'index_items'}, inplace=True)
    while df.delete.sum() > 0:
        index = df.delete.idxmax()
        index_corrected = index + df.iloc[index].index_correction_priority
        df.drop(index=index_corrected, inplace=True)
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)
        df['gap'] = (df.date - df.date.shift()).dt.days
        df['sat_change'] = abs(df.sat - df.sat.shift()).astype(bool)
        df['delete'] = (df['gap'] < delta_min).values 
        if show:
            df.plot.scatter(x='date', y='sat', c=df['delete'], figsize=(8,1), marker='+')

    return [items[index] for index in df.index_items.to_list()]

def date_isin(date, start_end, delta_min):
    delta_min = timedelta(days=delta_min)
    if date > (start_end[0] + delta_min) and date < (start_end[1] - delta_min):
        return True
    return False

def check_intervals(date, dates_hole, delta_min):
    for date_hole in dates_hole:
        if date_isin(date, date_hole, delta_min):
            return True
    
    return False 
    
def get_cc(item, geometry, disturbance, config):
    sat = dplatform[ item.properties['platform'] ]
    try:
        crs, transfo, qa, _ = item.assets.crop_as_array(config[sat]['qa'][0], bbox= shape(geometry).bounds)
        qa_on_disturbance = qa[:,disturbance.astype('bool')].reshape(-1)
        if dplatform[item.properties['platform']] == 'landsat-7':
            coeff = (qa_on_disturbance == 1).mean() #SLC failure taken as cloud-cover
        else :
            coeff = 0
        return ((qa_on_disturbance == config[sat]['qa_cloud'][0]).sum() / qa_on_disturbance.shape[0] + coeff ) * 100
    except:
        return 100
    
from skimage.transform import resize


dfunc = {
    'rgb': lambda x,c: x[[2,1,0], :, :], 
    'ndvi' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'ndwi' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'nbr' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'ndre' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'evi' : lambda x,c: 2.5 * (x[0] - x[1]) / (x[0] + 6. * x[1] - 7.5 * x[2] + 1).clip(-1, 1) / 2 + 0.5,
    'crswir' : lambda x,c: (x[1] / ( (x[2] - x[0]) * c[0] + x[0] )) / 5 ,
}

def check_transfo_crs(ibands):
    itransfo = [x[1] for x in ibands]
    icrs = [x[0] for x in ibands]
    return (itransfo.count(itransfo[0]) == len(itransfo)) * (icrs.count(icrs[0]) == len(icrs))

def get_indices(folder, item, config_sat, geometry, indices=['rgb'], target=None, aoi=None, \
                force_reproject=False, pansharpening=True, sharpening=True, normalization=None):

    #retrieve bands
    bands = []
    for indice in indices:
        bands.extend(config_sat["_".join([indice, 'bands'])])
    bands = set(bands)

    #download bands
    dbands = dict()
    try :
        for key in bands:
            dbands[key] = get_asset(item, config_sat, key, shape(geometry).bounds)
    except :
        return 0


    #compute indices
    r = 0
    for indice in indices:
        bands = config_sat["_".join([indice, 'bands'])]
        ibands = [ dbands[key] for key in bands ]
        #same spatial resolution and same crs 
        if check_transfo_crs(ibands):
            pass
        else :
            index_pre_target = np.argmax([ band[2].shape[1:] for band in ibands])
            pt_crs, pt_transfo, pt_arr = ibands[index_pre_target]
            ## scale all bands and then stack and preprocessing
            for i in range(len(ibands)):
                if i != index_pre_target:
                    arr = resize(ibands[i][2], pt_arr.shape, preserve_range=True, order=3)
                    ibands[i] = (ibands[i][0], pt_transfo, arr)

        crs = ibands[0][0]
        transfo = ibands[0][1]
        arr = np.stack([b[2].squeeze() for b in ibands], axis=0)
        #preprocessing (PanSharpening, Sharpening, Reprojecting, Histogram Matching)
        crs, transfo, arr = preprocessing(crs, transfo, arr, item, bands, config_sat, geometry, target=target,\
                                            force_reproject=force_reproject, pansharpening=pansharpening, sharpening=sharpening)
        
        #computing indice :
        c = None
        if indice == 'crswir' :
            c = config_sat['crswir_coeffs']
        arr = dfunc[indice](arr, c)

        
        #writing 
        if arr is not None:
            if aoi is not None:
                if len(arr.shape) == 2:
                    arr = np.expand_dims(arr, axis=0)

                if arr[0].shape != aoi.shape:
                    print('gi-arr :', arr.shape)
                    print('gi-aoi :', aoi.shape)
                    print('indice: ', indice)
                    print('ibands: ', ibands)
                    print('target: ', target)
                    print('item properties: ', item.properties)

                arr[:, aoi == 1] = 1
            direction = os.path.join(folder, indice)
            os.makedirs(direction, exist_ok=True)
            write_tif(arr, transfo, crs, direction, item, normalization=normalization)
            r += 1
        
    return r/len(indices)


def preprocessing(crs, transfo, arr, item, bands, config_sat, geometry, target=None, force_reproject=False, pansharpening=False, sharpening=True):
    #Conv to reflectance
    arr = config_sat['reflectance'](arr) 

    #Calibration reflectance 
    #TO-DO : use RNN
    # if target is not None:
    #     key = '-to-'.join([dplatform[ item.properties['platform'] ], target['sat']])
    #     if key in config_sat:
    #         arr = spectral_calibration(arr, bands, config_sat[key])

    r_arr = arr.copy()

    #Pansharpening (optional)
    if pansharpening and config_sat['pan'] is not None:
        arr, transfo, pansharpening = get_pansharpened(item, config, arr, transfo, geometry)

    #Reprojection 
    if target is not None and (arr[0].shape < target['shape'] or force_reproject):
        arr, transfo = reproject(arr, destination=np.zeros((arr.shape[0],*target['shape'])),\
                                  src_transform=transfo, src_crs=crs, dst_transform=target['transform'], dst_crs=target['crs'], \
                                    resampling=rasterio.enums.Resampling.cubic)
        crs = target['crs']
        
        #sharpening
        if sharpening and 'landsat' in dplatform[item.properties['platform']].lower():
            if pansharpening:
                radius = 1.5
                amount = 1.
            else:
                radius = 3.
                amount = 1. 

            arr = unsharp_mask(arr, radius=radius, amount=amount)

        #histogram matching
        arr = match_histograms(arr, r_arr, channel_axis=0)

    return crs, transfo, arr

def spectral_calibration(arr, bands, coeffs):
    
    f = lambda x,a,b: (x-b)/a 

    for i in range(len(bands)):
        b, a = coeffs[bands[i]]
        arr[i] = f(arr[i], a, b)

    return arr 

def wrapper_item_target(item_target, band, geometry, config):
    
    if band == 'pan':
        sat = dplatform[item_target.properties['platform']]
        collection = config[sat]['pan']['collection']
        item_target = collection.get_item( item_target.id.replace(config[sat]['pan']['from'], config[sat]['pan']['to']) )
        return item_target.assets.crop_as_array(config[sat]['pan']['pan_band'], bbox=shape(geometry).bounds)
    else:
        return item_target.assets.crop_as_array(band, bbox=shape(geometry).bounds)
    
class Chronos():

    def __init__(self, delta_min:int, cc1=5, cc2=75, buffer=3000,\
                 pansharpening=True, sharpening=True, force_reproject=False,\
                 normalization=None, verbose=0, n_jobs=-1, prefer='threads') -> None:
        self.delta_min = delta_min
        self.cc1 = cc1
        self.cc2 = cc2
        self.buffer = buffer
        self.pansharpening = pansharpening
        self.sharpening = sharpening
        self.force_reproject = force_reproject
        self.normalization = normalization 
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.prefer = prefer
        self.count_sat = dict()
        

    def research(self, geometry, start_date, end_date):
        """Research items

        Args:
            geometry (_type_): in crs 4326
            start_date (_type_): _description_
            end_date (_type_): _description_
            buffer (int, optional): in meters. Defaults to 3000.
        """
        show = False
        if self.verbose > 2:
            show = True


        #geometry
        old_geometry = deepcopy(geometry)
        geometry = transform_geom('epsg:3857', 'epsg:4326', shape(old_geometry).convex_hull)
        geometry_buffer = transform_geom('epsg:3857', 'epsg:4326', shape(old_geometry).convex_hull.buffer(self.buffer))

        #sats 
        self.sats = get_sats(start_date, end_date, config, priority)
        for sat in self.sats:
            self.count_sat[sat] = 0

        if self.verbose > 0:
            print('satellites used: ', self.sats)

        items1 = research_items(geometry, start_date, end_date, self.sats, config, self.cc1)
        if self.verbose > 0:
            print(f'Items found for research 1 with cc={self.cc1} : {len(items1)}')

        x = [item.properties['datetime'] for item in items1]
        y = [self.sats.index( dplatform[ item.properties['platform'] ]) for item in items1]

        #gap
        df = pd.DataFrame(data=np.array([x,y]).T, columns=['date', 'sat'])
        df['gap'] = (df.date - df.date.shift()).dt.days
        df['hole'] = (df.gap > self.delta_min * 2)
        dates_hole = [[df.date.iloc[x-1], df.date.iloc[x]] for x in df.index.to_numpy()[df.hole]]

        #research 2 
        items2 = research_items(geometry, start_date, end_date, self.sats, config, self.cc2)
        dates = [item.properties['datetime'] for item in items2]
        indexes_prob = [check_intervals(date, dates_hole, self.delta_min//2) for date in dates]
        items_prob = [items2[i] for i in range(len(items2)) if indexes_prob[i]]

        
        step = False
        i = 0 
        while not step:
            item = items2[i]
            try :
                crs, transfo, arr, _ = item.assets.crop_as_array(config[ dplatform[ item.properties['platform'] ] ]['all_bands'][0], bbox= shape(geometry).bounds)
                step = True
            except :
                pass
            i += 1

        if step : 
            aoi = rasterize([transform_geom('epsg:3857', crs, old_geometry.buffer(100).convex_hull)], out_shape = arr.shape[1:], transform=transfo, fill=np.nan, all_touched=False)
            cloud_cover = Parallel(n_jobs=self.n_jobs, prefer=self.prefer, verbose=self.verbose)(delayed(get_cc)(item, geometry, aoi, config) for item in items_prob)
            indexes_cc_ok = (np.array(cloud_cover) < self.cc1)
            cloud_cover = np.array(cloud_cover)[indexes_cc_ok]
            items_cc_ok = [items_prob[i] for i in range(len(items_prob)) if indexes_cc_ok[i]]

            #update CC:
            for i,item in enumerate(items_cc_ok):
                item.properties['eo:cloud_cover'] = cloud_cover[i]

            if self.verbose > 0:
                print(f'Items found for research 2 with cc={self.cc1} (on aoi) : {len(items2)}')

            #assembling
            items1.extend(items_cc_ok)
            items1 = sorted(items1, key=lambda x:x.properties['datetime'])
            if self.verbose > 0:
                print('Items found for combined research: ', len(items1))

            #filtering
            x = [item.properties['datetime'] for item in items1]
            cc = [item.properties['eo:cloud_cover'] for item in items1]
            y = [self.sats.index(dplatform[ item.properties['platform'] ]) for item in items1]
            df = pd.DataFrame(data=np.array([x,y,cc]).T, columns=['date', 'sat', 'cc'])
            items_kept = select_items(df, items1, self.delta_min, show)
            if self.verbose > 0:
                print('Items selected to evenly space the data: ', len(items_kept))
            
            #count per satellite
            for item in items_kept:
                self.count_sat[ dplatform[ item.properties['platform'] ] ] = self.count_sat[ dplatform[ item.properties['platform'] ] ] + 1

            if self.verbose>1:
                fig, ax = plt.subplots(figsize=(8,2))
                for sat in df['sat'].unique():
                    dfplot = df[ df.sat == sat ]
                    dfplot.plot.scatter(ax=ax, x='date',y='cc', c=f'C{sat}', marker='+', label=self.sats[sat], legend=True)
                # ax.set_yticks(range(len(self.sats)))
                # ax.set_yticklabels(self.sats)
                ax.grid(True, linestyle='--', lw=0.5)
                plt.show()

            return items_kept, geometry, geometry_buffer, old_geometry
        
        else :
            'research failed'


    def download(self, folder:str, geometry:dict, start_date, end_date, indices=['rgb'], show_aoi=True):
        """Downlaod items

        Args:
            folder (str): folder to download 
            geometry (dict): _description_
            start_date (_type_): _description_
            end_date (_type_): _description_
            indices (list, optional): _description_. Defaults to ['rgb'].
            aoi (_type_, optional): _description_. Defaults to None.
        """
        #research
        items, geometry, geometry_buffer, old_geometry = self.research(geometry, start_date, end_date)

        #target
        target = {'sat':None, 'band':None, 'res':100, 'crs':None, 'transform':None, 'shape':None}
        for sat in self.sats: #reverse to not have L5 as target 
            res = min(config[sat]['resolution'].values())
            if res < target['res'] or ( res == target['res'] and self.count_sat[sat] > self.count_sat.get(target['sat'], 0) ):
                target['sat'] = sat
                target['res'] = res
                target['band'] = min(config[sat]['resolution'], key=config[sat]['resolution'].get)
                

        i = 0
        target_state = False
        p_targets = [item for item in items if dplatform[item.properties['platform']] == target['sat']]
        while not target_state and i < self.count_sat[target['sat']]:
            item_target = p_targets[i]
            try :
                crs_target, transfo_target, arr_target, _ = wrapper_item_target(item_target, target['band'], geometry_buffer, config)
                target_state = True
                
            except:
                i+=1
        
        if target_state:
            target['crs'] = crs_target
            target['transform'] = transfo_target
            target['shape'] = arr_target.squeeze().shape

            if self.verbose > 0:
                print('target:', target)

            #show aoi
            aoi = None 
            if show_aoi:
                aoi = rasterize([transform_geom('epsg:3857', crs_target, old_geometry.buffer(100).convex_hull.boundary)], \
                                out_shape = arr_target.shape[1:], transform=transfo_target, fill=np.nan, all_touched=False)

            #download 
            if self.verbose > 0:
                print('Starting download at: ', folder)

            return Parallel(n_jobs=self.n_jobs, prefer=self.prefer, verbose=self.verbose)\
                (delayed(get_indices)(folder, item, config[dplatform[item.properties['platform']]],\
                                    shape(geometry_buffer).envelope, indices=indices, target=target, aoi=aoi, \
                                        force_reproject=self.force_reproject, pansharpening=self.pansharpening, \
                                            sharpening=self.sharpening, normalization=self.normalization) for item in items)
        else :
            print('target not available')
            return None
        

