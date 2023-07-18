import phoenix as phx
import rasterio
from rasterio.warp import reproject
from shapely.geometry import shape 
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import json 
from pansharpening import pansharpen, histogram_match_pansharpen_visual
from skimage.filters import unsharp_mask
from skimage.exposure import match_histograms
import numpy as np 
import rasterio

from config import dplatform, config, dsatellite

#registration
# from itsr.registration import main
# from itsr.utils import group_by_n 
from registration import main

def _registration_(folder, outdir, extra_inputs=None, extra_outputs=None):
    inputs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]
    outputs = [os.path.join(outdir, f) for f in os.listdir(folder) if f.endswith('.tif')]
    print('inputs:', inputs[0])
    print('outputs:', outputs[0])
    print('extra_outputs:')
    for i in extra_outputs:
        print(i[0])
    print('extra_inputs:')
    for i in extra_inputs:
        print(i[0])
    #switch to main in registration.py 
    main(inputs, outputs, extra_inputs=extra_inputs, extra_outputs=extra_outputs)

#utils
def write_tif(arr, transfo, crs, direction, item, indice, dtype='uint16', normalization=None, scaling=True):
    """Write a GeoTiFF from an array

    Args:
        arr (np.array): array to write
        transfo (_type_): transform of the GeoTiFF
        crs (_type_): crs of the GeoTiFF
        direction (_type_): direction to write the GeoTiFF
        item (_type_): item from catalog
        dtype (str, optional): dtype for writing the GeoTiFF ; 'uint8' or 'uint16'. Defaults to 'uint16'.
        normalization (list, optional): a and b coeff for (x-a)/(b-a) normalization. Defaults to None.
    """
    pr  = {
            'transform': transfo,
            'crs': crs,
            'width': arr.shape[2],
            'height': arr.shape[1],
            'count': arr.shape[0],
            'dtype': dtype, 
            'driver': 'GTiff'
            
        }
    
    if normalization is not None:
        arr = ((arr - normalization[0]) /  (normalization[1]- normalization[0])).clip(0,1)

    if scaling:
        arr = arr.clip(0,1)
        if dtype == 'uint16':
            arr *= 65535
        elif dtype == 'uint8':
            arr *= 255 
        else:
            raise ValueError('dtype should be uint8 or uint16')


    name = '{}_{}_{}.tif'.format(item.properties['datetime'].strftime("%Y-%m-%d_%H%M%S"), dsatellite[item.properties['platform']], indice.upper())
    with rasterio.open(os.path.join(direction, name),'w', **pr) as ff:
        ff.write(arr)

def get_asset(item, band, bbox):
    """Get desired band from item and crop it to bbox

    Args:
        item (dict): item from catalog
        band (string): name of the band
        bbox (tuple): coordinates of the bounding box (minx, miny, maxx, maxy) in EPSG:4326

    Returns:
        crs, transfo, array 
    """
    crs, transfo, arr, _ = item.assets.crop_as_array(band, bbox = bbox)

    return crs, transfo, arr 

    
def get_pansharpened(item, config, arr, transfo, geometry_buffer):
    """Get pansharpened image from item and geometry

    Args:
        item (_type_): item from catalog
        config (dict): config with all satellite parameters and information
        arr (np.array):  array to pansharpen
        transfo (_type_): transform of the array
        geometry_buffer (_type_): geometry with buffer of the area of interest

    Returns:
        arr, transfo, bool (pansharpenned or not)
    """
    sat = dplatform[item.properties['platform']]
    collection = config[sat]['pan']['collection']

    try :
        item_pan = collection.get_item( item.id.replace(config[sat]['pan']['from'], config[sat]['pan']['to']) )
        #cst
        metadata = json.loads(item_pan.assets.download_as_str('MTL_JSON'))
        cst = metadata['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']
        add = float(cst['REFLECTANCE_ADD_BAND_8'])
        mult = float(cst['REFLECTANCE_MULT_BAND_8'])
        crs_pan, transfo_pan, arr_pan, _ = item_pan.assets.crop_as_array(config[sat]['pan']['pan_band'], bbox= shape(geometry_buffer).bounds)
        pan = (mult * (arr_pan.squeeze()) + add) 
        #pan-sharpening
        pansharpened = pansharpen(arr, pan, method='pca',interpolation_order=3 ,with_nir=False)
        transfo = transfo * transfo.scale((arr.shape[1] / pansharpened.shape[1]),\
                                        (arr.shape[2] / pansharpened.shape[2]))
        # print('transfo: ', transfo)
        # print('transfo_pan: ', transfo_pan)
        pansharpened = histogram_match_pansharpen_visual(pansharpened, arr)
        return pansharpened, transfo_pan, True
    except:
        return arr, transfo, False
    
def get_sats(start, end, config, priority):
    """Get satellites available for a given time interval"""
    sats = []
    start = datetime(year=start.year, month=start.month, day=start.day)
    end = datetime(year=end.year, month=end.month, day=end.day)
    for key in config:
        start_sat = config[key]['start']
        end_sat = config[key]['end']
        if (start_sat <= end) and (start <= end_sat):
            sats.append(key)

    return [x for _, x in sorted(zip(priority, sats))]

def research_items(geometry, start_date, end_date, sats, config, cc):
    """Research items for a given geometry, time interval, satellites and cloud cover"""
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
    """Select items for a given dataframe and delta_min"""

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
    
def get_cc(item, geometry, daoi:dict, config, shadow=False):

    sat = dplatform[ item.properties['platform'] ]
    disturbance = daoi[sat]
    try:
        #retrieve QA band
        crs, transfo, qa, _ = item.assets.crop_as_array(config[sat]['qa_bands'][0], bbox= shape(geometry).bounds)
        qa_on_disturbance = qa[:,disturbance.astype('bool')].reshape(-1)

        #compute nodata and cloud cover
        nodata = (qa_on_disturbance == config[sat]['qa_nodata'][0]).mean() * 100
        if shadow:
            key = 'qa_cloud_shadow'
        else:
            key = 'qa_cloud'
        cc = np.isin(qa_on_disturbance, config[sat][key]).astype(int).mean() * 100

        return cc, nodata 
    except:
        return 100, 100 

import json 

def get_nd(item, daoi, geometry, shadow=False):
    """Get nodata value for a given item and geometry

    Args:
        item (dict): item from catalog
        daoi (_type_): _description_
        geometry (dict): _description_

    Returns:
        _type_: _description_
    """
    if dplatform[ item.properties['platform'] ] == 'landsat-7' or dplatform[ item.properties['platform'] ] == 'sentinel-2' :
        _, nd = get_cc(item, geometry, daoi, config, shadow=shadow)
        item.properties['nodata'] = nd 
    return None

def is_pan_available(item, geometry):
    """Check is panchromatic band is available for a given item and geometry
    It loads metadata and check if panchromatic band is available with Phoenix API

    Args:
        item (dict): item from catalog
        geometry (dict): geometry of the area of interest

    Returns:
        bool: True if panchromatic band is available, False otherwise
    """
    sat = dplatform[item.properties['platform']]
    try : 
        collection = config[sat]['pan']['collection']
        item_pan = collection.get_item( item.id.replace(config[sat]['pan']['from'], config[sat]['pan']['to']) )
        metadata = json.loads(item_pan.assets.download_as_str('MTL_JSON'))
        pan = item_pan.assets.crop_as_array(config[sat]['pan']['pan_band'], bbox= shape(geometry).bounds)
        return True
    except : 
        return False
    
def get_res(item, geometry):
    sat = dplatform[item.properties['platform']]
    band = config[sat]['rgb_bands'][0]
    res = config[sat]['resolution'][band]
    if 'pan' in config[sat].keys() and is_pan_available(item, geometry):
        res = config[sat]['resolution']['pan']
    item.properties['resolution'] = res



#GRAPH
import networkx as nx
import statistics
import math
import matplotlib.pyplot as plt

def score(node, cc_ub, sr_ub, nd_ub):

    return 1/cc_ub * node['cloud_cover'] + 1/sr_ub * node['spatial_res'] + 1/nd_ub * node['nodata']

def get_scores(graph, nodes, cc_ub, sr_ub, nd_ub):
    #score 
    attribute1_list = [graph.nodes[n]['cloud_cover'] for n in nodes]
    attribute2_list = [graph.nodes[n]['spatial_res'] for n in nodes]
    attribute3_list = [graph.nodes[n]['nodata'] for n in nodes]
    return [1/cc_ub * x + 1/sr_ub * y + 1/nd_ub * z for x, y, z in zip(attribute1_list, attribute2_list, attribute3_list)], (attribute1_list, attribute2_list, attribute3_list)

def get_metrics(graph, shortest_path, cc_ub=10, sr_ub=30, nd_ub=25):
    """_summary_
    
    Args:
        graph (_type_): _description_
        shortest_path (_type_): _description_
        a (int, optional): _description_. Defaults to 1.
        b (int, optional): _description_. Defaults to 1.
        c (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """

    #score 
    scores, (attribute1_list, attribute2_list, attribute3_list) = get_scores(graph, shortest_path, cc_ub, sr_ub, nd_ub)
    avg_score = statistics.mean(scores)
    avg_cloud_cover = statistics.mean(attribute1_list)
    avg_spatial_res = statistics.mean(attribute2_list)
    avg_nodata = statistics.mean(attribute3_list)

    #weight
    weights = [graph.edges[n]['weight'][0] for n in zip(shortest_path, shortest_path[1:])]
    avg_weight = statistics.mean(weights)
    std_dev_weight = statistics.stdev(weights)

    return avg_score, avg_cloud_cover, avg_spatial_res, avg_nodata, avg_weight, std_dev_weight

def weight_function(u, v, graph, alpha=0.1):
    """
    weight[1] between 0 and 3
    weight[0] between 0 and 30 
    To equally weight the two attributes, we divide the weight[0] by 10
    """
    return graph[u][v]['weight'][0] * alpha + graph[u][v]['weight'][1] 

def select_dates(df, target_weight=30, lower_bound=20, alpha=0.1, verbose=0, cc_ub=10, sr_ub=30, nd_ub=25):
    
    """_summary_
    Inputs :
        - df : dataframe with columns 'date', 'sat', 'cc', 'nodata', 'resolution'
        - target_weight : target weight for the path
        - lower_bound : lower bound for the time difference between two acquisitions
        - alpha : weight of the temporal dif in the weight function
        - verbose : verbosity level
        - cc_ub : upper bound for the cloud cover
        - sr_ub : upper bound for the spatial resolution
        - nd_ub : upper bound for the nodata percentage

    """

    df = df.fillna(0)
    timestamps = (df['date'] - df['date'].min() ).dt.days.to_numpy()


    #build timestamp difference matrix (adjencty matrix)
    N = len(timestamps)
    x = np.ones(N)
    deltaT = np.outer(x, timestamps) - np.outer(timestamps, x)
    ub = (df['date'] - df['date'].shift() ).dt.days.max() * 1.25

    if verbose > 0:
        print('upper bound : ', ub)
    deltaT_cond = deltaT.copy()
    deltaT_cond[ (abs(deltaT) > ub) | (abs(deltaT) < lower_bound) ] = np.nan
    deltaT_cond[ deltaT < 0 ] = np.nan

    if verbose > 1:
        plt.figure(figsize=(10, 10))
        plt.title('adjency matrix')
        plt.imshow(deltaT_cond)
        plt.colorbar(shrink=0.6, label='difference in days')
        plt.xlabel('nodes')
        plt.ylabel('nodes')
        plt.show()

    #build graph
    # Create an empty graph
    graph = nx.DiGraph()

    # Add nodes to the graph
    for i in range(len(timestamps)):
        row = df.iloc[i]
        attributes = {
            'cloud_cover': row['cc'],
            'spatial_res': row['resolution'],
            'nodata' : row['nodata'], 
            'date' : timestamps[i]
        }
        graph.add_node(i, **attributes)

    # Add edges to the graph based on the time differences
    for i in range(len(timestamps)):
        for j in range(i+1, len(timestamps)):
            time_diff = deltaT_cond[i][j]
            
            if lower_bound < time_diff < ub:
                if i < j:
                    graph.add_edge(i, j, weight = (abs(time_diff-target_weight), score(graph.nodes[j], cc_ub, sr_ub, nd_ub)))
                else:
                    graph.add_edge(j, i, weight = (abs(time_diff-target_weight), score(graph.nodes[i], cc_ub, sr_ub, nd_ub)))

    if verbose > 0:
        print('is strongly connected : ', nx.is_strongly_connected(graph))
        print('is weakly connected : ', nx.is_weakly_connected(graph))

    # Find the path that minimizes the sum of scores while considering weights with approximately the same weight
    start_node = 0
    end_node = len(timestamps) - 1
    shortest_path = None
    min_score = (math.inf, math.inf)
    buffer = 15 #days

    # Iterate over possible start and end nodes within the buffer time period
    for i in range(0, 10):
        start_date = graph.nodes[start_node]['date']
        possible_start = graph.nodes[i]
        possible_start_date = possible_start['date']
        time_diff_start = abs(possible_start_date - start_date)

        if time_diff_start <= buffer:
            for j in range(len(timestamps) - 10, len(timestamps)):
                end_date = graph.nodes[end_node]['date']
                possible_end = graph.nodes[j]
                possible_end_date = possible_end['date']
                time_diff_end = abs(possible_end_date - end_date)

                if time_diff_end <= buffer:

                    try :
                        path = nx.shortest_path(graph, i, j, weight=lambda u, v, d: weight_function(u, v, graph, alpha))
                    except:
                        continue

                    if path is not None:
                        #compute metrics
                        avg_score, avg_cloud_cover, avg_spatial_res, avg_nodata, avg_weight, std_dev_weight = get_metrics(graph, path)
                        tmp_score = (avg_score, abs(target_weight - avg_weight))
                        if tmp_score < min_score:
                            min_score = tmp_score
                            shortest_path = path
                            print(f'Avg Score: {avg_score:.2f}, Avg Cloud cover: {avg_cloud_cover:.2f}, Avg Spatial res: {avg_spatial_res:.2f}, Avg nodata: {avg_nodata:.2f}, Avg Weight: {avg_weight:.2f}, Std Dev Weight: {std_dev_weight:.2f}')

    # Create a scatter plot of nodes
    scores, _  = get_scores(graph, graph.nodes(), cc_ub=cc_ub, sr_ub=sr_ub, nd_ub=nd_ub)
    x = [graph.nodes[n]['date'] for n in graph.nodes()]

    if shortest_path is None:
        print('No path found')
        return None
    
    if verbose > 1:
        plt.figure(figsize=(20, 10))
        plt.scatter(x, scores, s=15, color='green', alpha=0.5)
        plt.title('Graph Visualization')
        plt.xlabel('Time')
        plt.ylabel('Score')

        # Add arrows to the plot for the shortest path
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i+1]
            plt.arrow(x[u], scores[u], x[v] - x[u], scores[v] - scores[u], color='red', alpha=0.5, width=0.01, head_width=0.01, length_includes_head=True, linestyle='solid')

        scores_s, _ = get_scores(graph, shortest_path, cc_ub=10, sr_ub=30, nd_ub=25)
        plt.scatter([graph.nodes[n]['date'] for n in shortest_path], scores_s, marker='x', color='red')
        #compute metrics
        avg_score, average_attribute1, average_attribute2, average_attribute3, average_weight, std_dev_weight = get_metrics(graph, shortest_path)

        # Add arrows to the plot for directed edges
        plt.text(0.05, 0.75, f'Average Cloud cover: {average_attribute1:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.7, f'Average Spatial res: {average_attribute2:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.65, f'Average nodata: {average_attribute3:.2f}', transform=plt.gca().transAxes)

        plt.text(0.05, 0.9, f'Avg Score: {avg_score:.2f}', transform=plt.gca().transAxes)

        plt.text(0.05, 0.85, f'Average Weight: {average_weight:.2f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.8, f'Std Dev Weight: {std_dev_weight:.2f}', transform=plt.gca().transAxes)
        plt.show()

    selectionned_dates = np.zeros(N)
    selectionned_dates[shortest_path] = 1
    df['selected'] = selectionned_dates
    return df

from skimage.transform import resize


dfunc = {
    'rgb': lambda x,c: x[[2,1,0], :, :], 
    'ndvi' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'ndwi' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'nbr' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'ndre' : lambda x,c: (x[0] - x[1]) / (x[0] + x[1]).clip(-1, 1) / 2 + 0.5,
    'evi' : lambda x,c: 2.5 * (x[0] - x[1]) / (x[0] + 6. * x[1] - 7.5 * x[2] + 1).clip(-1, 1) / 2 + 0.5,
    'crswir' : lambda x,c: (x[1] / ( (x[2] - x[0]) * c[0] + x[0] )) / 5 ,
    'qa' : lambda x,c: x, 
    'cloud_mask' : lambda x,c: x.astype(int)
}

def check_transfo_crs(ibands):
    itransfo = [x[1] for x in ibands]
    icrs = [x[0] for x in ibands]
    return (itransfo.count(itransfo[0]) == len(itransfo)) * (icrs.count(icrs[0]) == len(icrs))

def get_indices(folder, item, config_sat, geometry, log_path=None, indices=['rgb'], target=None, aoi=None, \
                force_reproject=False, pansharpening=True, sharpening=True, normalization=None, \
                    dtype='uint16', shadow=False):

    #retrieve bands
    bands = []
    for indice in indices:
        if indice == 'cloud_mask':
            indice = 'qa'
        bands.extend(config_sat["_".join([indice, 'bands'])])
    bands = set(bands)

    #download bands
    dbands = dict()
    try :
        for key in bands:
            dbands[key] = get_asset(item, key, shape(geometry).bounds)
    except :
        with open(log_path, 'a') as f:
            f.write(f'item {item.id} not available \n')
        return 0


    #compute indices
    r = 0
    for indice in indices:
        if indice == 'cloud_mask':
            bands = config_sat["_".join(['qa', 'bands'])]
        else:
            bands = config_sat["_".join([indice, 'bands'])]
        ibands = [ dbands[key] for key in bands ]

        #cloud_mask
        if indice == 'cloud_mask':
            if shadow:
                key = 'qa_cloud_shadow'
            else:
                key = 'qa_cloud'
            ibands[0] = (ibands[0][0], ibands[0][1], np.isin(ibands[0][2].astype(int), config_sat[key]).astype(int) )

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
        if indice == 'qa' or indice == 'cloud_mask':
            only_projection = True
            resampling = rasterio.enums.Resampling.nearest
            scaling = False
            dtype = 'uint16' 
        else :
            only_projection = False
            resampling = rasterio.enums.Resampling.cubic
            scaling = True

        crs, transfo, arr, log = preprocessing(crs, transfo, arr, item, config_sat, geometry, only_projection=only_projection, target=target,\
                                            force_reproject=force_reproject, pansharpening=pansharpening, sharpening=sharpening, resampling=resampling)

        if log_path is not None:
            with open(log_path, 'a') as f:
                f.write('{:<25s}'.format(f'indice: {indice}') + log + '\n')
        
        #computing indice :
        c = None
        if indice == 'crswir' :
            c = config_sat['crswir_coeffs']
        
        arr = dfunc[indice](arr, c)
        
        #writing 
        if arr is not None:
            if len(arr.shape) == 2:
                    arr = np.expand_dims(arr, axis=0)
            if aoi is not None:
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
            write_tif(arr, transfo, crs, direction, item, indice, dtype=dtype, scaling=scaling, normalization=normalization)
            r += 1
        
    return r/len(indices)


def preprocessing(crs, transfo, arr, item, config_sat, geometry, only_projection=False, target=None, \
                   force_reproject=False, pansharpening=False, sharpening=True, resampling=rasterio.enums.Resampling.cubic):
    
    osr = transfo.a 

    #Conv to reflectance
    if not only_projection:
        arr = config_sat['reflectance'](arr) 

    r_arr = arr.copy()

    #Pansharpening (optional)
    if pansharpening and config_sat['pan'] is not None and not only_projection:
        arr, transfo, pansharpening = get_pansharpened(item, config, arr, transfo, geometry)
    else :        
        pansharpening = False

    #Reprojection 
    if target is not None and (arr[0].shape < target['shape'] or force_reproject):
        arr, transfo = reproject(arr, destination=np.zeros((arr.shape[0],*target['shape'])),\
                                  src_transform=transfo, src_crs=crs, dst_transform=target['transform'], dst_crs=target['crs'], \
                                    resampling=resampling)
        crs = target['crs']
        
        #sharpening
        if sharpening and 'landsat' in dplatform[item.properties['platform']].lower() and not only_projection:
            if pansharpening:
                radius = 1.5
                amount = 1.
            else:
                radius = 3.
                amount = 1. 

            arr = unsharp_mask(arr, radius=radius, amount=amount)
        else :
            sharpening = False

        #histogram matching
        if not only_projection:
            arr = match_histograms(arr, r_arr, channel_axis=0)

    log = f"satellite: {item.properties['platform']}, \
        date: {item.properties['datetime']}, \
        pansharpening: {pansharpening}, \
        sharpening: {sharpening}, \
        original spatial resolution: {osr}"
    
    log = "{:<25s} | {:<25s} | {:<25s} | {:<25s} | {:<25s}".format(*log.split(','))

    return crs, transfo, arr, str(log)

def wrapper_item_target(item_target, band, geometry, config):
    
    if band == 'pan':
        sat = dplatform[item_target.properties['platform']]
        collection = config[sat]['pan']['collection']
        item_target = collection.get_item( item_target.id.replace(config[sat]['pan']['from'], config[sat]['pan']['to']) )
        return item_target.assets.crop_as_array(config[sat]['pan']['pan_band'], bbox=shape(geometry).bounds)
    else:
        return item_target.assets.crop_as_array(band, bbox=shape(geometry).bounds)
    