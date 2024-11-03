import os

import geopandas as gpd


def read_gdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    gdf = gpd.read_feather(path)
    return gdf[gdf.geometry.is_valid]


def gdf_box_contains(container, contained):
    if container.crs != contained.crs:
        raise ValueError("Coordinate systems should match")

    # Get the total bounds for both GeoDataFrames
    minx_c, miny_c, maxx_c, maxy_c = container.total_bounds
    minx_t, miny_t, maxx_t, maxy_t = contained.total_bounds

    # Check if the bounding box of 'contained' is completely within the bounding box of 'container'
    if not (minx_t >= minx_c and maxx_t <= maxx_c and miny_t >= miny_c and maxy_t <= maxy_c):
        return False


def gdf_box_overlaps(container, contained):
    if container.crs != contained.crs:
        raise ValueError("Coordinate systems should match")

    # Get the total bounds for both GeoDataFrames
    minx_c, miny_c, maxx_c, maxy_c = container.total_bounds
    minx_t, miny_t, maxx_t, maxy_t = contained.total_bounds

    # Check if the bounding box of 'contained' overlaps with the bounding box of 'container'
    if (minx_t <= maxx_c and maxx_t >= minx_c and
        miny_t <= maxy_c and maxy_t >= miny_c):
        return True
    return False
