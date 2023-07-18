from rasterio.warp import transform_geom
from rasterio.features import rasterize
from shapely.geometry import shape 
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os 
from copy import deepcopy
import shutil
import rasterio

from config import config, priority, dplatform
from utils import research_items, get_sats, get_nd, get_cc, get_res, select_dates, check_intervals, wrapper_item_target, get_indices, _registration_


class Chronos():

    def __init__(self, delta_min:int, cc1=5, cc2=75, buffer=3000, nodata=10, crs=None, \
                 alpha = 0.1, show_aoi = False, dtype='uint16', satellites=None, \
                 shadow = False, registration=True, 
                 pansharpening=True, sharpening=True, force_reproject=False,\
                 normalization=None, verbose=0, n_jobs=-1, prefer='threads') -> None:
        
        """
        Chronos class

        
        Inputs:
        - delta_min: int, minimum number of days between two acquisitions
        - cc1: int, cloud cover threshold for the first research
        - cc2: int, cloud cover threshold for the second research
        - buffer: int, buffer in meters around the geometry
        - nodata: int, nodata threshold 
        - crs: str, crs used for the geometry
        - alpha: float, parameter to weight between the two objectives in the Dijsktra algoritm, 
                alpha=0.1 is equally balancing the two objectives : minimizing spatial resolution, 
                cloud cover and no data percentage & having a dense time serie (fix period between two acquisitions)
        - pansharpening: bool, pansharpening for the second research
        - sharpening: bool, sharpening for the second research
        - force_reproject: bool, force reproject for the second research
        - normalization: str, normalization for the second research
        - verbose: int, verbose level
        - n_jobs: int, number of jobs for parallel computing
        - prefer: str, prefered parallel computing method
        """

        self.delta_min = delta_min
        self.crs = crs
        self.satellites = satellites 
        self.cc1 = cc1
        self.cc2 = cc2
        self.nodata = nodata
        self.buffer = buffer
        self.alpha = alpha
        self.show_aoi = show_aoi
        self.shadow = shadow
        self.pansharpening = pansharpening
        self.sharpening = sharpening
        self.force_reproject = force_reproject
        self.normalization = normalization 
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.prefer = prefer
        self.dtype = dtype 
        self.registration = registration
        self.count_sat = dict()

        assert self.dtype in ['uint8', 'uint16'], 'dtype should be either uint8 or uint16'
        assert self.crs is not None, "Please specify a crs used for the geometry with crs='epsg:XXXX'"
        
        
    def research(self, geometry, start_date, end_date):

        #geometry
        old_geometry = deepcopy(geometry)
        geometry = transform_geom(self.crs, 'epsg:4326', shape(old_geometry).convex_hull)
        geometry_buffer = transform_geom(self.crs, 'epsg:4326', shape(old_geometry).convex_hull.buffer(self.buffer))

        #sats 
        self.count_sat = {}
        self.sats = get_sats(start_date, end_date, config, priority)
        if self.satellites is not None:
            self.sats = [sat for sat in self.sats if sat in self.satellites]

        if self.verbose > 0:
            print('satellites used:', self.sats)

        for sat in self.sats:
            self.count_sat[sat] = 0

        #RESEARCH 1
        #TODO: add a check on the number of items found for the first research, SHOULD BE > 0
        items1 = research_items(geometry, start_date, end_date, self.sats, config, self.cc1)

        #retrieve crs, transform and array for each sat

        items1_sat = set([dplatform[item.properties['platform']] for item in items1])
        items1_prop = {sat:None for sat in items1_sat}
        for sat in items1_sat:
            i = 0 
            temp_items1 = [item for item in items1 if dplatform[item.properties['platform']] == sat]
            while items1_prop[sat] is None and i < len(temp_items1):
                item = temp_items1[i]
                try :
                    crs, transfo, arr, _ = item.assets.crop_as_array(config[ dplatform[ item.properties['platform'] ] ]['qa_bands'][0], bbox= shape(geometry).bounds)
                    items1_prop[sat] = {'crs':crs, 'transfo':transfo, 'arr':arr}
                except :
                    pass
                i += 1

        items1_sat = [sat for sat in items1_sat if items1_prop[sat] is not None]

        #retrieve aoi for each sat
        daoi = {}
        for sat in items1_sat:
            daoi[sat] = rasterize([transform_geom(self.crs, items1_prop[sat]['crs'] , old_geometry.buffer(100).convex_hull)], out_shape = items1_prop[sat]['arr'].shape[1:], transform=items1_prop[sat]['transfo'], fill=np.nan, all_touched=False)

        #research and filter for landsat-7
        _ = Parallel(n_jobs=self.n_jobs, prefer=self.prefer, verbose=self.verbose)(delayed(get_nd)(item, daoi, geometry, shadow=self.shadow) for item in items1)
        items1 = [item for item in items1 if ('nodata' not in item.properties or item.properties['nodata'] <= self.nodata)]
        if self.verbose > 0:
                    print(f'Items found for research 1 with cc={self.cc1} : {len(items1)}')
        x = [start_date] + [item.properties['datetime'] for item in items1] + [end_date]
        y = [None] + [self.sats.index( dplatform[ item.properties['platform'] ]) for item in items1] + [None]

        #gap
        df = pd.DataFrame(data=np.array([x,y]).T, columns=['date', 'sat'])
        df['gap'] = (df.date - df.date.shift()).dt.days
        df['hole'] = (df.gap > self.delta_min)
        dates_hole = [[df.date.iloc[x-1], df.date.iloc[x]] for x in df.index.to_numpy()[df.hole]]

        #RESEARCH 2 
        items2 = research_items(geometry, start_date, end_date, self.sats, config, self.cc2)
        dates = [item.properties['datetime'] for item in items2]
        indexes_prob = [check_intervals(date, dates_hole, 1) for date in dates]
        items_prob = [items2[i] for i in range(len(items2)) if indexes_prob[i]]

        #retrieve crs, transform and array for each sat 
        items2_sat = set([dplatform[item.properties['platform']] for item in items2])
        items2_prop = items1_prop
        for sat in items2_sat:
            items2_prop.setdefault(sat, None)
        

        if not sorted(items2_sat) == sorted(items1_sat):
            for sat in items2_sat:
                if sat not in items1_sat:
                    self.count_sat[sat] = 0
                    i = 0 
                    temp_items2 = [item for item in items2 if dplatform[item.properties['platform']] == sat]
                    while items2_prop[sat] is None and i < len(temp_items2):
                        item = temp_items2[i]
                        try :
                            crs, transfo, arr, _ = item.assets.crop_as_array(config[ dplatform[ item.properties['platform'] ] ]['qa_bands'][0], bbox= shape(geometry).bounds)
                            items2_prop[sat] = {'crs':crs, 'transfo':transfo, 'arr':arr}
                        except :
                            pass
                        i += 1


        items2 = [item for item in items2 if items2_prop[dplatform[item.properties['platform']]] is not None]
        items2_sat = [sat for sat in items2_sat if items2_prop[sat] is not None]

        if len(items2) > 0:
            daoi = {}
            for sat in items2_sat:
                daoi[sat] = rasterize([transform_geom(self.crs, items2_prop[sat]['crs'] , old_geometry.buffer(100).convex_hull)], out_shape = items2_prop[sat]['arr'].shape[1:], transform=items2_prop[sat]['transfo'], fill=np.nan, all_touched=False)
            cc_nodata = Parallel(n_jobs=self.n_jobs, prefer=self.prefer,  verbose=self.verbose)(delayed(get_cc)(item, geometry, daoi, config, shadow=self.shadow) for item in items_prob)
  
            indexes_cc_ok = (np.array(cc_nodata)[:,0] < 25) #cc < 10%
            indexes_nodata_ok = (np.array(cc_nodata)[:,1] < self.nodata) #nodata < 10%
            cloud_cover = np.array(cc_nodata)[indexes_cc_ok, 0]
            nodata_cover = np.array(cc_nodata)[indexes_nodata_ok, 1]
            items_ok = [items_prob[i] for i in range(len(items_prob)) if (indexes_cc_ok[i] and indexes_nodata_ok[i])]

            #update CC:
            for i,item in enumerate(items_ok):
                item.properties['eo:cloud_cover'] = cloud_cover[i]
                item.properties['nodata'] = nodata_cover[i]

            if self.verbose > 0:
                print(f'Items accepted during research 2 with cc=25 (on aoi) : {len(items_ok)}')

            #assembling
            items1.extend(items_ok)
            items1 = sorted(items1, key=lambda x:x.properties['datetime'])
            _ = Parallel(n_jobs=-1, prefer='threads', verbose=self.verbose)(delayed(get_res)(item, geometry) for item in items1)
            if self.verbose > 0:
                print('Items found for combined research: ', len(items1))

            #filtering
            x = [item.properties['datetime'] for item in items1]
            cc = [item.properties['eo:cloud_cover'] for item in items1]
            nd = [item.properties.get('nodata', np.nan) for item in items1]
            res = [item.properties['resolution'] for item in items1]

            y = [self.sats.index(dplatform[ item.properties['platform'] ]) for item in items1]
            df = pd.DataFrame(data=np.array([x,y,cc,nd, res]).T, columns=['date', 'sat', 'cc', 'nodata', 'resolution'])

            if len(x) > 0:
                #select items
                df = select_dates(df, target_weight=30, lower_bound=20, alpha=self.alpha, verbose=self.verbose, cc_ub=10, sr_ub=30, nd_ub=self.nodata)
                if df is not None:
                    items_kept = [items1[i] for i in df.index.to_numpy()[df.selected == 1]]
                else :
                    #one item over 2 is kept
                    items_kept = [items1[i] for i in range(len(items1)) if i%2 == 0]
                    print('One item over 2 is kept')
                
                #count per satellite
                for item in items_kept:
                    self.count_sat[ dplatform[ item.properties['platform'] ] ] = self.count_sat[ dplatform[ item.properties['platform'] ] ] + 1

                return items_kept, geometry, geometry_buffer, old_geometry
            else :
                print('Critical Issue with the tile')
                return None, geometry, geometry_buffer, old_geometry


    def download(self, folder:str, geometry:dict, start_date, end_date, indices=['rgb']):
        """Downlaod items

        Args:
            folder (str): folder to download 
            geometry (dict): __geo_interface__ dict 
            start_date (_type_): _description_
            end_date (_type_): _description_
            indices (list, optional): _description_. Defaults to ['rgb'].
        """

        log_path = os.path.join(folder,'log-frames.txt')
        os.makedirs(folder, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write('-- Chronos download log --\n')

        #research
        items, geometry, geometry_buffer, old_geometry = self.research(geometry, start_date, end_date)
        if items is None:
            with open(os.path.join(log_path), 'w') as f:
                f.write('Critical issue with the tile \n')
            return None

        #target
        target = {'sat':None, 'band':None, 'res':100, 'crs':None, 'transform':None, 'shape':None}
        for sat in self.sats: #reverse to not have L5 as target 
            res = min(config[sat]['resolution'].values())
            if res < target['res'] or ( res == target['res'] and self.count_sat[sat] > self.count_sat.get(target['sat'], 0) ):
                target['sat'] = sat
                target['res'] = res
                target['band'] = min(config[sat]['resolution'], key=config[sat]['resolution'].get)
                

        #get target to define the spatial resolution of the time series
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
            if self.show_aoi:
                aoi = rasterize([transform_geom(self.crs, crs_target, old_geometry.buffer(100).convex_hull.boundary)], \
                                out_shape = arr_target.shape[1:], transform=transfo_target, fill=np.nan, all_touched=False)

            #download 
            if self.verbose > 0:
                print('Starting download at: ', folder)

            return Parallel(n_jobs=self.n_jobs, prefer=self.prefer,  verbose=self.verbose)\
                (delayed(get_indices)(folder, item, config[dplatform[item.properties['platform']]],\
                                    shape(geometry_buffer).envelope, log_path=log_path, indices=indices, target=target, aoi=aoi, \
                                        force_reproject=self.force_reproject, pansharpening=self.pansharpening, \
                                            shadow=self.shadow,
                                                sharpening=self.sharpening, dtype=self.dtype, normalization=self.normalization) for item in items)
            
        else :
            with open(os.path.join(log_path), 'w') as f:
                f.write('target not available \n')
            return None
        
        
        

    def get_registration(self, folder, keep_only_common=False, remove_old=False):

        indices = os.listdir(folder)
        indices = [x for x in indices if os.path.isdir(os.path.join(folder, x))]
        targets = [x for x in indices if x not in ['temp', 'qa', 'cloud_mask']]

        if len(targets) > 1:
            if 'rgb' in targets:
                target = 'rgb'
            else :
                target = targets[0]

            targets = [x for x in targets if x != target]
            
            extra_inputs = [] 
            extra_outputs = []
            for indice in targets:
                extra_inputs.append([os.path.join(folder, indice, x) for x in os.listdir(os.path.join(folder, indice)) if x.endswith('.tif')])
                extra_outputs.append([os.path.join(folder, indice+'_r', x) for x in os.listdir(os.path.join(folder, indice)) if x.endswith('.tif')])
        else :
            extra_inputs = None
            extra_outputs = None 

        if self.verbose > 0:    
            print('Starting registration...')

        
        _registration_(os.path.join(folder, target), outdir=os.path.join(folder, target+'_r'),\
                      extra_inputs=extra_inputs, extra_outputs=extra_outputs)

        #get files to delete by looking at the file present in the folder target but not in the folder target_r
        for indice in targets:
            files = os.listdir(os.path.join(folder, indice))
            files_r = os.listdir(os.path.join(folder, indice+'_r'))
            deleted_files = set(files) - set(files_r)
            if self.verbose > 1:
                print('files to remove: ', deleted_files) 

        #remove not registered files and rename files
        if remove_old:
            for indice in targets+[target]:
                shutil.rmtree(os.path.join(folder, indice))
                os.rename(os.path.join(folder, indice+'_r'), os.path.join(folder, indice))

        #remove files not registered
        if keep_only_common:
            for indice in targets+[target]:
                for file in deleted_files:
                    os.remove(os.path.join(folder, target, file))
        
    def dl_pipeline(self, geometry, start_date, end_date, folder, indices=['rgb'], remove_old=False, keep_only_common=False, registration=True):
        """Download pipeline

        Download + registration 

        Args:
            geometry (dict): _description_
            start_date (datetime.datetime): starting date of the time series
            end_date (datetime.datetime): ending date of the time series
            folder (str): folder to download
            indices (list, optional): Indices you want to download. Defaults to ['rgb']. 
            remove_old (bool, optional): Remove old files. Defaults to False.
            keep_only_common (bool, optional): Keep only common files. Defaults to False.
        """

        #download
        self.download(folder, geometry, start_date, end_date, indices=indices)

        #registration
        if registration:
            self.get_registration(folder, remove_old=remove_old, keep_only_common=keep_only_common)
