
import dotenv
import phoenix as phx
import os
from datetime import datetime

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

dsatellite = {
    'sentinel-2a':'S2A',
    'sentinel-2b':'S2B',
    'landsat-5':'L5',
    'landsat-7':'L7',
    'landsat-8':'L8',
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
        'qa_bands': ['QA_PIXEL'],
        'qa_cloud' : [22280, 22080, 21826, 21890, 54596, 22144, 54852, 55052], #22280= High cloud proba, 22080=Medium cloud proba, 21826= over land proba, 21890=over water proba, 54596=high cirrus proba
        'qa_cloud_shadow': [22280, 22080, 21826, 21890, 54596, 22144, 54852, 55052, 23888, 23952, 24088, 24216, 24344, 24472], #+shadow
        'qa_nodata' : [1],
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
        'qa_bands': ['QA_PIXEL'],
        'qa_cloud' : [5896, 5696, 5442, 5506, 5760, 7960, 8088], #CLOUDS H & M proba + Cirrus
         # https://www.usgs.gov/media/files/landsat-4-7-collection-2-level-2-science-product-guide
        'qa_cloud_shadow' : [5896, 5696, 5442, 5506, 5760, 7824, 7960, 8088, 7440, 7568, 7696], #+ shadow
        'qa_nodata' : [1],
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
        'qa_bands': ['QA_PIXEL'],
        'qa_cloud' : [5896, 5696, 5442, 5506, 5760, 7960, 8088], #CLOUDS H & M proba + Cirrus
         # https://www.usgs.gov/media/files/landsat-4-7-collection-2-level-2-science-product-guide
        'qa_cloud_shadow' : [5896, 5696, 5442, 5506, 5760, 7824, 7960, 8088, 7440, 7568, 7696], #+ shadow
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
        'qa_bands': ['SCL'],
        'qa_cloud' : [8,9,10], #https://rasterframes.io/masking.html 8=cloud medium proba, 9=cloud high proba, 10=thin cirrus
        'qa_cloud_shadow' : [3,8,9,10], #+ shadow
        'qa_nodata' : [0],
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
    #     'qa_bands': ['SCL'],
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