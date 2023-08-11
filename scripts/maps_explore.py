#%%
import prettymaps

#%%
plot = prettymaps.plot("3171 Keswick Rd, Baltimore, Maryland", preset="heerhugowaard")

#%%
from pkg.data import DATA_PATH

activities_loc = DATA_PATH / "strava" / "BDP_2023-03-19" / "activities"

#%%
import glob

files = glob.glob(str(activities_loc / "*.gpx"))

#%%

import os

os.listdir(activities_loc)

#%%

from gpxplotter import read_gpx_file, create_folium_map, add_segment_to_map

the_map = create_folium_map()
for track in read_gpx_file(files[-5]):
    for i, segment in enumerate(track["segments"]):
        add_segment_to_map(the_map, segment)

# To display the map in a Jupyter notebook:
the_map
