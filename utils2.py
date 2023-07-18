import string
from unidecode import unidecode
from shapely.geometry import shape

def str_wo_space(x):
    return  unidecode(x.translate(str.maketrans('', '', string.punctuation)).replace(' ', '-'))

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")

def localisation(geometry):
    centroid = shape(geometry).centroid 
    Latitude = str(centroid.y)
    Longitude = str(centroid.x)
    location = geolocator.reverse(Latitude+","+Longitude)
    dict_loc = location.raw['address']
    keys = [x for x in ['country_code', 'state', 'county', 'city'] if x in dict_loc.keys()]
    name = '-'.join([ str_wo_space(dict_loc[key]) for key in keys])
    return name, Latitude, Longitude

from OSMPythonTools.nominatim import Nominatim
from shapely import wkt
from rasterio.warp import transform_geom
from shapely.geometry import shape

def get_polygon(name, epsg):
    finder = Nominatim()
    spatial_entity = finder.query(name, wkt=True)
    pol = wkt.loads(spatial_entity.wkt())
    pol_ = shape(transform_geom('EPSG:4326', epsg, pol))
    return pol_, spatial_entity