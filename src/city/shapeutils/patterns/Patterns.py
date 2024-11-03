import gc
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import ops
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm


class Skeleton:
    def __init__(self, polygon, crs=26910, resolution=1):
        self.polygon = polygon
        self.crs = crs
        self.res = resolution
        self.gdf = gpd.GeoDataFrame(columns=['pol_id', 'geometry'], geometry='geometry', crs=self.crs)
        return

    def generate(self, export=False):
        gdf = self.polygon.copy()
        resolution = self.res

        pts_dfs = []
        print("Generating skeletons")
        for i, (geom, t) in enumerate(zip(gdf['geometry'], tqdm(range(len(gdf['geometry']))))):
            # print(f"{i + 1}/{len(gdf)}")

            # Create points along the boundary of the geometry
            pts = [geom.boundary.interpolate(resolution * j).coords[0] for j in range(int(geom.length / resolution))]
            pts_df = gpd.GeoDataFrame({'geometry': [Point(pt) for pt in pts]}, geometry='geometry', crs=self.crs)
            pts_dfs.append(pts_df)

            # Create voronoi diagram
            vor = Skeleton(pts)

            # Extract skeleton lines
            lines = gpd.GeoDataFrame(
                {'geometry': [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]},
                geometry='geometry', crs=self.crs)

            # Filter skeleton lines within the footprint
            lines = lines[lines.within(geom)]

            # Buffer vertices to overlay with lines
            buffers = [Point(end_pts).buffer(0.05) for line in lines['geometry'] for end_pts in
                       line.coords]  # [Point(vor.vertices[point][0]).buffer(0.2) for point in valid_vertices]

            # Get unique vertices and overlay skeleton
            r_pts = gpd.GeoDataFrame({'geometry': buffers}, geometry='geometry',
                                     crs=self.crs).drop_duplicates().reset_index()  # .loc[nodes.tolist(), :]

            if len(lines) > 0:

                # Get vertices that intersects with three or more lines
                overlay = gpd.overlay(lines, r_pts, how='intersection')
                nodes = np.unique([k for k in overlay['index'] if len(overlay[overlay['index'] == k]) > 2])

                # If skeleton has only one line
                if len(nodes) == 0:
                    lines = lines
                    joined_lines = [lines['geometry'].unary_union.simplify(5)]

                else:
                    # Subtract lines from polygons (nodes)
                    lines = gpd.overlay(lines, r_pts.loc[r_pts['index'].isin(nodes)], how='difference')

                    # Join segments and simplify
                    joined_lines = [ln for ln in ops.linemerge(lines['geometry'].unary_union).simplify(5)]

                # Append to GeoDataFrame
                for line in joined_lines:
                    l = len(self.gdf)

                    if line.__class__.__name__ == 'MultiLineString':
                        line = ops.linemerge(line).simplify(5)

                    self.gdf.at[l, 'pol_id'] = i
                    self.gdf.at[l, 'geometry'] = line

            else:
                print("Skipped")

            gc.collect()

        if export:
            gpd.GeoDataFrame(pd.concat(pts_dfs), crs=self.crs).to_file('points.geojson', driver='GeoJSON')
            self.gdf.to_file('skeletons.geojson', driver='GeoJSON')
        return self.gdf.set_geometry('geometry')


class Isovist:
    def __init__(self, origins, barriers, radius=500, crs=26910):
        self.origins = origins.reset_index()
        self.barriers = barriers
        self.barriers_index = barriers.sindex
        self.radius = radius
        self.crs = crs
        return

    def create(self, tolerance=100):

        # Buffer origin point according to radius
        buffer = self.origins.copy()
        buffer['geometry'] = buffer.buffer(self.radius).simplify(tolerance=tolerance)
        buffer['centroid'] = buffer.centroid

        # Create view lines crossing over barriers
        lines = gpd.GeoDataFrame(
            {
                'id': [i for b, i in zip(buffer['geometry'], buffer.index) for pt_cds in b.boundary.coords],
                'geometry': [LineString([c, Point(pt_cds)]) for b, c in zip(buffer['geometry'], buffer['centroid']) for pt_cds in b.boundary.coords]
            }, geometry='geometry'
        )
        print(lines.sindex)

        # Subtract barriers from view lines
        start = time.time()
        print('\Subtracting barriers')
        lines_diff = gpd.overlay(lines, self.barriers, how='difference')
        print(f'~ {int((time.time() - start)/60)} minutes') # 51 min

        # start = time.time()
        # # Extract lines that intersects with origin
        # origins_uu = gpd.GeoDataFrame({'geometry': [self.origins.buffer(15).unary_union]}, geometry='geometry')
        # print(origins_uu.sindex)
        # print('\View lines that intersects origin (orverlay)')
        # view_lines = gpd.overlay(lines_diff, origins_uu, how='intersection')
        # print(f'~ {int((time.time() - start)/60)} minutes') # 206 min

        start = time.time()
        print('\View lines that intersects origin (recursion)')
        # Extract lines that intersects with origin
        origins_uu = gpd.GeoDataFrame({'geometry': [self.origins.unary_union]}, geometry='geometry')
        origins_uu.crs = self.crs
        print(origins_uu.sindex)

        lines_orig = gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry')
        lines_orig.crs = self.crs
        for i, geom in zip(lines_diff['id'], lines_diff['geometry']):
            j = len(lines_orig)
            if (geom.__class__.__name__ == 'LineString') and (geom.distance(origins_uu['geometry'][0]) < 0.001):
                lines_orig.at[j, 'id'] = i
                lines_orig.at[j, 'geometry'] = geom
            elif geom.__class__.__name__ == 'MultiLineString':
                for line in geom:
                    if line.distance(origins_uu['geometry'][0]) < 0.001:
                        lines_orig.at[len(lines_orig), 'id'] = i
                        lines_orig.at[j, 'geometry'] = line
        print(f'~ {int((time.time() - start)/60)} minutes')

        # Create isovist polygon
        gdf = self.origins.copy()
        for i in lines.id.unique():
            polygon = []
            for line in lines_orig.loc[lines_orig['id'] == i]['geometry']:
                if line is not None:
                    polygon.append(Point(line.coords[1]))
            gdf.at[i, 'geometry'] = Polygon(polygon)

        return gdf['geometry']
