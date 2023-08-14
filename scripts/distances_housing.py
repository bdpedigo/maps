# %%
import googlemaps
import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

# %%
creds_loc = "api-creds.txt"
with open(creds_loc) as f:
    creds = f.readline()
creds
# %%
gmaps = googlemaps.Client(key=creds)


# %%


trip_times = [5, 10, 15, 20, 25]  # in minutes
# trip_times = [1,2,3,4,5]
# trip_times = np.linspace(0.5, 5, 20)
travel_speed = 4.5  # walking speed in km/hour

G, coords = ox.graph_from_address(
    "Mary Gates Hall",
    network_type="drive",
    return_coords=True,
    dist=1500,
)
# G = ox.graph_from_bbox(47.686840, 47.601737, -122.274726, -122.436564)
gdf_nodes = ox.graph_to_gdfs(G, edges=False)
min_dist = np.inf
for i, (lat, long) in enumerate(gdf_nodes[["y", "x"]].values):
    dist = ox.distance.great_circle_vec(lat, long, coords[0], coords[1])
    if dist < min_dist:
        min_dist = dist
        center_node = gdf_nodes.index[i]

G = ox.project_graph(G)
gdf_nodes = ox.graph_to_gdfs(G, edges=False)

# %%
ox.plot_graph(G, node_size=1, edge_linewidth=0)


# %%
ox.plot_graph(G, node_size=0, edge_linewidth=0.2, edge_color="#999999")


# %%

mpm_multiplier = 26.8224

# add an edge attribute for time in minutes required to traverse each edge
meters_per_minute = travel_speed * 1000 / 60  # km per hour to m per minute
for _, _, _, data in G.edges(data=True, keys=True):
    if "maxspeed" not in data:
        meters_per_minute = 25 * mpm_multiplier
    elif isinstance(data["maxspeed"], list):
        meters_per_minute = int(data["maxspeed"][0].split(" ")[0]) * mpm_multiplier
    else:
        meters_per_minute = int(data["maxspeed"].split(" ")[0]) * mpm_multiplier
    data["time"] = data["length"] / meters_per_minute

# get one color for each isochrone
iso_colors = ox.plot.get_colors(
    n=len(trip_times), cmap="Reds", start=0, return_hex=True
)

# color the nodes according to isochrone then plot the street network
node_colors = {}
for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
    subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance="time")
    for node in subgraph.nodes():
        node_colors[node] = color
nc = [node_colors[node] if node in node_colors else "none" for node in G.nodes()]
ns = [15 if node in node_colors else 0 for node in G.nodes()]

# %%
fig, ax = ox.plot_graph(
    G,
    node_color=nc,
    node_size=ns,
    node_alpha=0.8,
    edge_linewidth=0.2,
    edge_color="#999999",
)

# make the isochrone polygons
isochrone_polys = []
for trip_time in sorted(trip_times, reverse=True):
    subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance="time")
    node_points = [
        Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
    ]
    bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
    isochrone_polys.append(bounding_poly)
gdf = gpd.GeoDataFrame(geometry=isochrone_polys)

# %%
# plot the network then add isochrones as colored polygon patches
fig, ax = ox.plot_graph(
    G, show=False, close=False, edge_color="#999999", edge_alpha=0.2, node_size=0
)
gdf.plot(ax=ax, color=iso_colors, ec="none", alpha=0.6, zorder=-1)

center_node_data = gdf_nodes.loc[center_node]
ax.plot(
    center_node_data["x"],
    center_node_data["y"],
    color="darkred",
    marker="*",
    markersize=7,
    zorder=100,
)

# %%

north = 47.686840
south = 47.601737
east = -122.274726
west = -122.436564

G = ox.graph_from_bbox(north, south, east, west)

# %%

n_steps = 50
nodes = ox.graph_to_gdfs(G, edges=False)
fig, ax = ox.plot_graph(
    G, node_size=0, edge_linewidth=0.2, edge_color="#999999", show=False
)
long_grid, lat_grid = np.meshgrid(
    np.linspace(east, west, n_steps), np.linspace(south, north, n_steps)
)

ax.axis("on")
ax.get_xaxis().set_visible(True)
ax.set_xticks(np.linspace(east, west, 10))
labels = ax.set_xticklabels(np.round(np.linspace(east, west, 10), 4))
ax.margins(0.1)
ax.tick_params(which="both", direction="out", length=5, color="#999999")
[s.set_visible(True) for s in ax.spines.values()]

for label in labels:
    label.set_rotation(45)
    label.set_horizontalalignment("center")
    label.set_color(color="#999999")

from tqdm.autonotebook import tqdm


def calculate_min_distance(lat_in, long_in, gdf_nodes, threshold=None):
    min_dist = np.inf
    for i, (lat_query, long_query) in enumerate(gdf_nodes[["y", "x"]].values):
        dist = ox.distance.great_circle_vec(lat_query, long_query, lat_in, long_in)
        if dist < min_dist:
            min_dist = dist
            center_node = gdf_nodes.index[i]
        if threshold is not None:
            if dist < threshold:
                return True, center_node
    if threshold is not None:
        return False, None
    return min_dist, center_node


good_queries = []
for lat, long in tqdm(
    zip(lat_grid.ravel(), long_grid.ravel()), total=len(long_grid.ravel())
):
    is_close, close_node = calculate_min_distance(lat, long, nodes, threshold=100)
    if is_close:
        ax.plot(long, lat, "o", color="darkred", markersize=3, alpha=1, zorder=10)
        good_queries.append({"lat": lat, "long": long, "close_node": close_node})

# ax.scatter(
#     xs.ravel(), ys.ravel(), s=3, color="darkred", marker="o", alpha=1, zorder=10
# )
plt.show()

# %%
import pandas as pd

good_queries = pd.DataFrame(good_queries)[["lat", "long"]]
query_list = list(zip(good_queries["lat"], good_queries["long"]))

# %%
querytime = datetime.today()


# %%
chunk_size = 10


query_list[:chunk_size]


def _query_distances_chunk(query_list):
    chunk_size = len(query_list)
    distance_matrix_out = gmaps.distance_matrix(query_list, coords, mode="transit")

    origin_addresses = []
    durations = []
    distances = []
    index = []
    for i in range(chunk_size):
        if distance_matrix_out["rows"][i]["elements"][0]["status"] != "OK":
            continue
        origin_addresses.append(distance_matrix_out["origin_addresses"][i])
        durations.append(
            distance_matrix_out["rows"][i]["elements"][0]["duration"]["text"]
        )
        distances.append(
            distance_matrix_out["rows"][i]["elements"][0]["distance"]["text"]
        )
        index.append(query_list[i])

    origin_addresses = pd.Series(origin_addresses, name="origin_address")
    durations = pd.Series(durations, name="duration")
    distances = pd.Series(distances, name="distance")
    out_df = pd.DataFrame([origin_addresses, durations, distances]).T
    out_df.index = index
    out_df.index.name = "query"
    return out_df


def query_distances(query_list, chunk_size=10):
    out_df = pd.DataFrame()
    for i in tqdm(range(0, len(query_list), chunk_size)):
        out_df = pd.concat(
            [out_df, _query_distances_chunk(query_list[i : i + chunk_size])]
        )
    out_df.reset_index(inplace=True)
    out_df["lat"] = out_df["query"].apply(lambda x: x[0])
    out_df["long"] = out_df["query"].apply(lambda x: x[1])
    out_df["duration_min"] = out_df["duration"].apply(lambda x: int(x.split(" ")[0]))
    return out_df


distance_outs = query_distances(query_list)
distance_outs

# %%
# gmaps.distance_matrix(query_list[190:200], coords, mode="transit")
# %%
distance_outs.to_csv("distance_outs.csv")

# %%
distance_outs = pd.read_csv("distance_outs.csv", index_col=0)


def _get_minutes(x):
    split_x = x.split(" ")
    if len(split_x) == 2:
        return int(split_x[0])
    elif len(split_x) >= 3:
        return 60 * int(split_x[0]) + int(split_x[2])
    else:
        raise ValueError(f"Unexpected duration string {x}")


distance_outs["duration_min"] = distance_outs["duration"].apply(_get_minutes)

assert (
    len(
        distance_outs[
            distance_outs["duration"].str.contains("hour")
            & (distance_outs["duration_min"] < 60)
        ]
    )
    == 0
)

distance_outs


# %%
distance_outs = gpd.GeoDataFrame(
    distance_outs,
    geometry=gpd.points_from_xy(distance_outs["long"], distance_outs["lat"]),
    crs=nodes.crs,
)

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cmap = mpl.cm.Reds
norm = mpl.colors.Normalize(vmin=0, vmax=90)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

fig.colorbar(
    mappable,
    cax=ax,
    orientation="horizontal",
    label="Some Units",
)

colors = mappable.to_rgba(distance_outs["duration_min"])

# %%
fig, ax = ox.plot_graph(
    G, node_size=0, edge_linewidth=0.2, edge_color="#999999", show=False
)

ax.scatter(
    distance_outs["long"],
    distance_outs["lat"],
    s=3,
    color=colors,
    marker="o",
    alpha=1,
    zorder=10,
)

# %%
trip_times = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # in minutes

# make the isochrone polygons
isochrone_polys = []
for trip_time in sorted(trip_times, reverse=True):
    # subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance="time")
    # node_points = [
    #     Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)
    # ]

    select_nodes = distance_outs[distance_outs["duration_min"] <= trip_time]
    node_points = [
        Point((x[0], x[1]))
        for x in select_nodes[["long", "lat"]].itertuples(index=False)
    ]
    bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
    isochrone_polys.append(bounding_poly)
    select_nodes.plot()

# %%
contour_gdf = gpd.GeoDataFrame(geometry=isochrone_polys)
contour_gdf.crs = nodes.crs

# plot the network then add isochrones as colored polygon patches

from osmnx import project_gdf, project_graph


# get one color for each isochrone
iso_colors = ox.plot.get_colors(
    n=len(trip_times), cmap="plasma", start=0, return_hex=True
)

project = False
if project:
    gdf_nodes = project_gdf(nodes)
    to_crs = gdf_nodes.crs
    contour_gdf = project_gdf(contour_gdf, to_crs=to_crs)
    G_projected = project_graph(G, to_crs=to_crs)
else:
    G_projected = G

gdf_nodes = ox.graph_to_gdfs(G_projected, edges=False)

fig, ax = ox.plot_graph(
    G_projected,
    show=False,
    close=False,
    edge_color="#999999",
    edge_alpha=0.2,
    node_size=0,
)

gdf.plot(ax=ax, color=iso_colors, ec="none", alpha=0.6, zorder=-1)

center_node_data = gdf_nodes.loc[center_node]
ax.plot(
    center_node_data["x"],
    center_node_data["y"],
    color="darkred",
    marker="*",
    markersize=10,
    zorder=100,
)

plt.show()

# %%
select_nodes.plot()

# %%
heatmap = np.empty((n_steps, n_steps))
heatmap[:] = np.nan
heatmap

heatmap_df = pd.DataFrame(
    heatmap,
    columns=np.linspace(east, west, n_steps),
    index=np.linspace(south, north, n_steps),
)

# %%

# %%
distance_outs["lat_bin"] = pd.Categorical(distance_outs["lat"]).as_ordered()
distance_outs["long_bin"] = pd.Categorical(distance_outs["long"]).as_ordered()
distance_heatmap = distance_outs.pivot_table("duration_min", "lat_bin", "long_bin")


def grey_nan_dilation(df):
    values = df.values
    new_values = values.copy()
    for i in range(0, values.shape[0]):
        for j in range(1, values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                one_neighbors = []
                if i > 0:
                    one_neighbors.append(values[i - 1, j])
                if i < values.shape[0] - 1:
                    one_neighbors.append(values[i + 1, j])
                if j > 0:
                    one_neighbors.append(values[i, j - 1])
                if j < values.shape[1] - 1:
                    one_neighbors.append(values[i, j + 1])

                if not np.isnan(one_neighbors).all():
                    new_values[i, j] = np.nanmean(one_neighbors)

    df_infilled = pd.DataFrame(new_values, index=df.index, columns=df.columns)
    return df_infilled


distance_heatmap_infilled = grey_nan_dilation(distance_heatmap)
distance_heatmap_infilled2 = grey_nan_dilation(distance_heatmap_infilled)
distance_heatmap_infilled3 = grey_nan_dilation(distance_heatmap_infilled2)

# %%
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns

df = distance_heatmap_infilled3
cols = df.columns.as_ordered().values
rows = df.index.as_ordered().values
values = df.values
interpolator = RegularGridInterpolator(
    (rows, cols), values, bounds_error=True, method="linear"
)

x_steps = 500
y_steps = 500
new_row_range = np.geomspace(rows.min(), rows.max(), y_steps)
new_col_range = np.geomspace(cols.min(), cols.max(), x_steps)


rows, cols = np.meshgrid(new_row_range, new_col_range, indexing="ij")
zs = interpolator((rows, cols))

interp_df = pd.DataFrame(data=zs, index=new_row_range, columns=new_col_range)

# %%
fig, axs = plt.subplots(1, 5, figsize=(50, 10))


heatmap_kws = dict(xticklabels=False, yticklabels=False, square=True, cbar=False)
sns.heatmap(distance_heatmap, ax=axs[0], **heatmap_kws)
axs[0].invert_yaxis()

sns.heatmap(distance_heatmap_infilled, ax=axs[1], **heatmap_kws)
axs[1].invert_yaxis()

sns.heatmap(distance_heatmap_infilled2, ax=axs[2], **heatmap_kws)
axs[2].invert_yaxis()

sns.heatmap(distance_heatmap_infilled3, ax=axs[3], **heatmap_kws)
axs[3].invert_yaxis()

sns.heatmap(interp_df, ax=axs[4], **heatmap_kws)
axs[4].invert_yaxis()

for ax in axs:
    ax.set_xlabel("")
    ax.set_ylabel("")

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# sns.heatmap(interp_df, ax=ax, cmap="inferno", **heatmap_kws)

df = interp_df
y = df.index.values
x = df.columns.values
x_step = x[1] - x[0]
grid_x = np.linspace(x.min() - x_step / 2, x.max() + x_step / 2, len(x) + 1)
grid_y = np.linspace(y.min() - x_step / 2, y.max() + x_step / 2, len(y) + 1)
X, Y = np.meshgrid(grid_x, grid_y)
Z = df.values
ax.pcolormesh(
    X, Y, Z, vmin=np.nanmin(Z), vmax=np.nanmax(Z), cmap="inferno", shading="auto"
)


fig, ax = ox.plot_graph(
    G_projected,
    show=False,
    close=False,
    edge_color="#999999",
    edge_alpha=0.2,
    node_size=0,
    ax=ax,
)

plt.show()

# levels = np.arange(0, 60, 10)
# cs = plt.contour(zs, levels=levels, colors="#999999")
# ax.clabel(
#     cs, cs.levels, colors="#999999", inline_spacing=0, fontsize=10, rightside_up=True
# )
# ax.invert_yaxis()

# %%

census_loc = "maps/data/2020_Census_Tracts_-_Seattle.geojson"
tract_gdf = gpd.read_file(census_loc)

# %%

seattle_borders = tract_gdf.unary_union

# %%
Point(-122.334791, 47.638582).within(seattle_borders)
# %%

interp_gdf = (
    interp_df.reset_index()
    .melt(id_vars="index", var_name="long", value_name="duration_min")
    .rename(columns={"index": "lat"})
)
interp_gdf["geometry"] = gpd.points_from_xy(interp_gdf["long"], interp_gdf["lat"])
interp_gdf = gpd.GeoDataFrame(interp_gdf, geometry="geometry", crs=nodes.crs)

import time

currtime = time.time()
chunk_size = 1000
n_chunks = int(np.ceil(len(interp_gdf) / chunk_size))
masks = []
for chunk_i in tqdm(range(n_chunks)):
    mask_i = interp_gdf.iloc[chunk_i * chunk_size : (chunk_i + 1) * chunk_size].within(
        seattle_borders
    )
    masks.append(mask_i)

mask = pd.concat(masks)
print(f"{time.time() - currtime:.3f} seconds elapsed.")
# %%
interp_gdf[mask].plot(column="duration_min")


# %%
interp_gdf_masked = interp_gdf[mask]

# redo this pivot correctly with a reindex after
interp_gdf_masked_square = interp_gdf_masked.pivot_table(
    index="lat", columns="long", values="duration_min"
)

# %%
fig, axs = plt.subplots(
    2, 1, figsize=(10, 10), gridspec_kw=dict(height_ratios=[20, 1], hspace=0)
)
# sns.heatmap(interp_df, ax=ax, cmap="inferno", **heatmap_kws)

ax = axs[0]
df = interp_gdf_masked_square
y = df.index.values
x = df.columns.values
x_step = x[1] - x[0]
y_step = y[1] - y[0]
grid_x = np.linspace(x.min() - x_step / 2, x.max() + x_step / 2, len(x) + 1)
grid_y = np.linspace(y.min() - y_step / 2, y.max() + y_step / 2, len(y) + 1)
X, Y = np.meshgrid(grid_x, grid_y)
Z = df.values

cmap = mpl.cm.Spectral_r
vmin = 0
vmax = 60
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
ax.pcolormesh(
    X,
    Y,
    Z,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    shading="auto",
)

# levels = np.arange(0, 60, 10)
cs = plt.contour(Z, levels=[25], colors="black", zorder=10)
ax.clabel(
    cs, cs.levels, colors="black", inline_spacing=0, fontsize=10, rightside_up=True
)

fig, ax = ox.plot_graph(
    G_projected,
    show=False,
    close=False,
    edge_color="#999999",
    edge_alpha=0.1,
    edge_linewidth=0.5,
    node_size=0,
    ax=ax,
)

# center_node_data = gdf_nodes.loc[center_node]
# ax.plot(
#     center_node_data["x"],
#     center_node_data["y"],
#     color="black",
#     marker="*",
#     markersize=15,
#     zorder=100,
# )

ax = axs[1]
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation="horizontal",
    label="Transit time (min)",
    fraction=1,
    shrink=0.7,
    pad=0,
)

plt.savefig("maps/results/figs/heatmap.png", dpi=300, bbox_inches="tight")
# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
cs = plt.contour(X, Y, Z, levels=[25], colors="black", zorder=10)
ax.clabel(
    cs, cs.levels, colors="black", inline_spacing=0, fontsize=10, rightside_up=True
)
