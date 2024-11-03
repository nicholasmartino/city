import os

import matplotlib

import sys
import math
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import time

sys.path.insert(0, "/Users/nicholasmartino/Google Drive/Python/morphology")
from shapeutils.ShapeTools import Shape, SpatialAnalyst, divide_line_by_count, get_point
from Fabric import Blocks, Parcels, Development
from Network import Streets
from shapely.affinity import scale
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import nearest_points, triangulate
from sqlalchemy import create_engine


class Zone:
    def __init__(self, name='', envelopes=None, fsr=None, land_use=None, frontage=None, neighborhood=None, parcels=None,
                 buildings=None, blocks=None, streets=None, lanes=None, directory='../data', crs=26910, export=False,
                 r_seed=0, verbose=False, max_coverage=None):

        self.name = name
        self.land_use = land_use
        self.envelopes = envelopes
        self.fsr = fsr
        self.max_coverage = max_coverage
        self.frontage = frontage

        self.frontages = []
        self.rears = []

        self.neigh = neighborhood
        # self.buildings = buildings
        # self.parcels = parcels
        # self.streets = streets
        self.lanes = lanes
        self.crs = crs
        self.export = export

        self.dir = f'{directory}/zones/{name}'
        try:
            if not os.path.exists(self.dir): os.mkdir(self.dir)
        except:
            pass
        self.segments = None

        # Set random seed
        self.r_seed = r_seed
        np.random.seed(self.r_seed)

        if blocks is not None:
            self.blocks = blocks
            self.boundaries = self.generate_block_boundaries()

        if self.envelopes is not None:
            self.ground_front_yard = self.envelopes['standard'].boxes[0].front
            if self.ground_front_yard == 0: self.ground_front_yard = 1.5

            self.ground_rear_yard = self.envelopes['standard'].boxes[0].rear
            if self.ground_rear_yard == 0: self.rear_yard = 1.5

        self.verbose = verbose
        return

    def buffer_overlay(self, left, right, buffer, how="difference"):
        """
        Returns a GeoPandas overlay of elements from the left DataFrame on buffered elements from the right DataFrames.

        :param left: GeoDataFrame;
        :param right: GeoDataFrame;
        :param buffer: float;
        :param how: Type of overlay operation according to the GeoPandas documentation;
        :return:
        """
        factor = 1.05
        right['geometry'] = [scale(geom, factor, factor, factor, origin='centroid') for geom in right['geometry']]
        right['buffer'] = right.buffer(buffer, cap_style=2)
        return gpd.overlay(left, right.set_geometry('buffer'), how=how)

    def export_parcels(self):
        return self.neigh.parcels.gdf.to_file('del_parcels.geojson', driver='GeoJSON')

    def export_blocks(self):
        return self.blocks.gdf.to_file('del_blocks.geojson', driver='GeoJSON')

    def export_boundaries(self):
        return self.blocks.gdf.to_file('del_boundaries.geojson', driver='GeoJSON')

    def explode_parcels(self):
        print(f"{self.name} - Exploding parcels")

        # gdf = self.neigh.parcels.gdf[self.neigh.parcels.gdf['zone'] == self.name]
        gdf = self.neigh.parcels.gdf.copy()
        gdf = gdf[~gdf['geometry'].isna()]
        gdf["geometry"] = gdf.geometry.simplify(tolerance=5, preserve_topology=True)

        # Explode parcel geometry into segments
        segments = gpd.GeoDataFrame({'id': [], 'geometry': []}, geometry='geometry', crs=self.crs)
        for i, ln in zip(gdf.id, gdf.boundary):

            # Check if geometry is multipart
            if 'Multi' in ln.geom_type:
                geoms = ln
            else:
                geoms = [ln]

            # Iterate over coordinates to City segments
            for geom in geoms:
                for seg in list(map(LineString, zip(geom.coords[:-1], geom.coords[1:]))):
                    j = len(segments)
                    segments.at[j, 'pid'] = int(i)
                    segments.at[j, 'geometry'] = seg
        segments['sid'] = segments.index
        segments['length'] = segments.length
        return segments

    def is_block_parcel(self):
        return int(sum(self.neigh.blocks.gdf.area)) == int(sum(self.neigh.parcels.gdf.area))

    def get_segments_distances(self):
        """

        :return: A dictionary where keys are parcel ids and values are Shapely LineStrings representing the furtherst parcel segment from the closest lane
        """
        # Explode parcels into segments if is not exploded yet
        if self.segments is None:
            segments = self.explode_parcels()
        else:
            segments = self.segments

        segments['length'] = segments.length
        segments['d2lane'] = [segment.distance(self.lanes.unary_union) for segment in segments.centroid]
        segments['d2streets'] = [segment.distance(self.streets.gdf.unary_union) for segment in segments.centroid]
        return segments

    def get_parcel_frontages(self):
        """
        Extract frontage line segments from parcels based on boundaries parameter.

        :return: GeoDataFrame
        """

        self.boundaries = self.generate_block_boundaries()
        parcels = self.neigh.parcels.gdf.copy()
        parcels = parcels[parcels.area > 0]

        if self.is_block_parcel():
            parcels['geometry'] = parcels.buffer(-self.ground_front_yard)
            frontages = parcels.copy()
            frontages['geometry'] = parcels.boundary
            frontages = frontages[frontages.length > 0]
            frontages = Shape(frontages).explode()

        else:
            assert self.lanes is not None, AssertionError("Zone object has no 'lanes' parameter")

            if type(self.lanes) == tuple:
                self.lanes = self.lanes[0]

            parcels = parcels.copy().reset_index(drop=True)
            frontages = Shape(parcels).extract_open_boundaries(inner_rings=False)
            frontages['Type'] = list(parcels.loc[frontages['parent_id'], 'Type'])

            front_ctr = frontages.copy()
            front_ctr.sindex
            front_ctr['geometry'] = frontages.centroid.buffer(2)

            bounds = gpd.GeoDataFrame({'geometry': self.boundaries}, crs=parcels.crs)
            bounds.sindex
            frontages = frontages[frontages['sid'].isin(gpd.overlay(front_ctr, bounds)['sid'])]

            # st = time.time()
            # frontages = frontages[[
            #     (geom.intersects(self.boundaries)) and
            #     (not geom.intersects(self.lanes.buffer(self.ground_rear_yard + 0.1).dropna().unary_union))
            #     for geom in frontages['geometry'].centroid
            # ]]
            # print(f"Frontages extracted ({int(time.time() - st)}s)")

        return frontages

    def get_building_frontages(self):
        self.boundaries = self.generate_block_boundaries()
        buildings = self.neigh.buildings.gdf.copy()
        frontages = Shape(buildings).extract_open_boundaries(inner_rings=False)

        return frontages[[geom.centroid.intersects(self.boundaries) for geom in frontages['geometry']]]

    def subtract_setbacks(self):
        """
        Return parcels with setbacks subtracted.

        :return: GeoDataFrame
        """

        parcels = self.neigh.parcels.gdf.copy()

        if self.is_block_parcel():
            parcels['geometry'] = parcels.buffer(-self.ground_front_yard)
            return parcels

        else:
            assert 'Type' in parcels.columns, KeyError("'Type' column not found in parcels GeoDataFrame")

            parcels = self.neigh.parcels.gdf.copy()
            frontages = self.get_parcel_frontages()
            rear = self.lanes.copy()

            parcels.sindex
            frontages.sindex
            rear.sindex

            non_court_flt = ~parcels['Type'].isin(['Courtyard', 'Podium', 'Cascading'])
            frontages['geometry'] = frontages.buffer(self.ground_front_yard, cap_style=3)
            rear.crs = parcels.crs
            rear['geometry'] = rear.buffer(self.ground_rear_yard, cap_style=3)

            parcels = gpd.overlay(parcels, frontages.loc[:, ['geometry']], how="difference")
            parcels.loc[non_court_flt, ['geometry']] = gpd.overlay(parcels.loc[non_court_flt, ['geometry']],
                                                                   rear.loc[:, ['geometry']], how="difference")
            parcels.crs = self.crs

            if len(parcels) > 0:
                return parcels
            else:
                print("Zone warning: Setbacks subtraction returned empty geometries")
                return self.neigh.parcels.gdf

    def chamfer_blocks(self, radius=5):
        blocks = self.blocks.gdf.copy()
        parcels = self.neigh.parcels.gdf.copy()

        # Buffer block geometries to chamfer edges
        blocks['geometry'] = [geom.buffer(radius, cap_style=3, join_style=3).buffer(-radius) for geom in
                              blocks['geometry']]

        # Overlay parcels to block geometries
        parcels = gpd.overlay(parcels, blocks.loc[:, ['geometry']]).drop_duplicates('pid', keep='first')

        self.blocks.gdf = blocks
        self.neigh.parcels.gdf = parcels

        if self.export:
            parcels.to_file('del_parcels.geojson', driver='GeoJSON')
            blocks.to_file('del_blocks.geojson', driver='GeoJSON')
        return blocks

    def generate_block_boundaries(self):
        try:
            self.blocks
        except:
            self.neigh.generate_blocks()
        blocks = self.neigh.blocks.gdf.copy()
        blocks = Shape(blocks, reset_index=True).divorce()
        uu = blocks.exterior.buffer(3).unary_union
        return uu

    def build_envelopes(self, export=False):
        print("Building envelopes")

        if 'zone' in self.neigh.parcels.gdf.columns:
            gdf = self.neigh.parcels.gdf[self.neigh.parcels.gdf['zone'] == self.name]
        else:
            gdf = self.neigh.parcels.gdf.copy()
        pcl = self.neigh.parcels.gdf.copy()
        env = self.envelopes

        # segments = self.explode_parcels()
        # if export:
        #     segments.to_file(f'{self.dir}/{self.name}_segments.geojson', driver='GeoJSON')
        #
        # # Identify frontage and rear parcel lines based on street position
        # frontages = self.get_closest(segments, self.streets.gdf.unary_union)
        # rear = self.get_closest(segments, self.lanes.unary_union)
        #
        # # Drop segments identified both as front and rear parcel lines
        # ambiguous = list(set(frontages['sid']).intersection(set(rear['sid'])))
        # only_front = frontages[~frontages['sid'].isin(ambiguous)]
        # only_rear = rear[~rear['sid'].isin(ambiguous)]
        #
        # # Get segments where distance to front reference is smaller than distance to rear reference and filter segments
        # mask = frontages[frontages['sid'].isin(ambiguous)]['distance'] < rear[rear['sid'].isin(ambiguous)]['distance']
        # frontages = pd.concat([only_front, frontages[frontages['sid'].isin(mask[mask].index)]])
        # rear = pd.concat([only_rear, rear[rear['sid'].isin(mask[~mask].index)]])

        # Identify frontage and rear parcel lines based on distance to lane

        segments = self.get_segments_distances()
        frontages_ids = [list(segments[segments['pid'] == parcel_id].sort_values('d2lane', ascending=False)['sid'])[0]
                         for parcel_id in segments['pid'].unique()]
        rear_ids = [list(segments[segments['pid'] == parcel_id].sort_values('d2lane', ascending=True)['sid'])[0] for
                    parcel_id in segments['pid'].unique()]
        frontages = segments.loc[frontages_ids, :]
        rear = segments.loc[rear_ids, :]

        # Create frontages column on parcels
        self.neigh.parcels.gdf.loc[frontages['pid'], 'Frontage'] = frontages.length

        if export:
            self.lanes.to_file(f'{self.dir}/{self.name}_lanes.geojson', driver='GeoJSON')
            frontages.to_file(f'{self.dir}/{self.name}_front.geojson', driver='GeoJSON')
            rear.to_file(f'{self.dir}/{self.name}_rear.geojson', driver='GeoJSON')

        boxes = gpd.GeoDataFrame()
        for case, envelope in env.items():
            if case == 'standard':
                for b, box in enumerate(envelope.boxes):

                    if box.front > 0:
                        # Subtract zone parcels from buffered frontage line segments
                        frontages_uu = gpd.GeoDataFrame({'geometry': [frontages.unary_union]})
                        overlay = self.buffer_overlay(left=gdf, right=frontages_uu, buffer=box.front)
                    else:
                        overlay = gdf

                    if box.rear > 0:
                        # If the base line for rear yard starts from lane centerline
                        lanes_uu = gpd.GeoDataFrame({'geometry': [self.lanes.unary_union]})
                        if box.rear_anchor == 'centerline':
                            overlay = self.buffer_overlay(left=overlay, right=lanes_uu, buffer=box.rear)

                        # If the base line for rear yard starts from parcel boundary
                        else:
                            overlay = self.buffer_overlay(left=overlay, right=rear.loc[:, ['geometry']],
                                                          buffer=box.rear)

                    # Subtract zone parcels from buffered side line segments by buffering neighbouring parcels !!!
                    for i in overlay.index:
                        # Filter adjacent parcels if block information exists inf parcel gdf
                        if 'block_id' in pcl.columns:
                            in_block = pcl[pcl['block_id'] == overlay.at[i, 'block_id']]
                        else:
                            in_block = pcl

                        if len(in_block) > 1:
                            overlay.loc[[i], :] = self.buffer_overlay(left=overlay.loc[[i], :], right=in_block.drop(i),
                                                                      buffer=box.side)

                        # If parcel occupies the whole block, offset the setback by using a negative buffer
                        elif len(in_block) == 1:
                            if 'buffer' in in_block.columns:
                                if not list(in_block['buffer'].is_empty)[0]:
                                    overlay.loc[[i], :] = self.buffer_overlay(left=overlay.loc[[i], :], right=in_block,
                                                                              buffer=box.side * -1, how="intersection")

                    # # Transform into polygons and append to boxes GeoDataFrame
                    # j = len(boxes)
                    # boxes.at[j, 'parcel_id'] = overlay.at[i, 'id']
                    # boxes.at[j, 'box'] = b
                    # boxes.at[j, 'height'] = box.height
                    # try: boxes.at[j, 'geometry'] = overlay.loc[[i], :].unary_union
                    # except: pass

                    # Make a minimum offset on higher boxes
                    if b > 0:
                        overlay['geometry'] = overlay.buffer(-b * 0.01)

                    overlay['height'] = [box.height for o in range(len(overlay))]
                    boxes = pd.concat([boxes, overlay])

            else:
                # Identify parcels that match the case...
                print("")

        if export:
            boxes.to_file(f'{self.dir}/{self.name}_envelope.geojson', driver='GeoJSON')
            boxes.to_feather(f"{self.dir}/{self.name}_envelopes.feather")
        return boxes

    def export_envelopes_pdk(self):
        envelopes = self.build_envelopes(),
        layer = pdk.Layer(
            type="GeoJsonLayer",
            data=envelopes[0],
            extruded=True,
            getElevation="height",
            getFillColor=[255, 210, 210, 180],
            get_line_color=[255, 255, 255],
            auto_highlight=False,
            pickable=False,
        )
        with open(f"{self.dir}/pydeck.json", "w") as file:
            file.write(layer.to_json())
        return layer


class DevelopmentZone(Zone):
    def __init__(self, zone, types, unit_mix, unit_count, bldg_form, development=None, units=None, whole_block=False,
                 tower_area=None, tower=None, podium=None, blocks=None, r_seed=0):
        Zone.__init__(self, zone.city, zone.envelopes, zone.fsr, frontage=zone.frontage, neighborhood=zone.neigh,
                      r_seed=zone.r_seed, directory=zone.dir[:-len(zone.city) - 6], crs=zone.crs, export=zone.export,
                      max_coverage=zone.max_coverage)
        self.development = development
        self.zone = zone
        self.types = types
        self.unit_mix = unit_mix
        self.unit_count = unit_count
        self.unit_gdf = gpd.GeoDataFrame(self.unit_mix).sort_values('share')
        self.bldg_form = bldg_form
        self.units = units
        self.whole_block = whole_block
        self.types = types
        self.buildings = gpd.GeoDataFrame()
        self.tower_area = tower_area
        self.tower = tower
        self.podium = podium
        self.r_seed = r_seed
        self.blocks = blocks

    def mix_development_types(self):
        """
        Mix development types on different blocks
        :param types:
        :return:
        """
        parcels = self.neigh.parcels.gdf.copy()
        parcels = parcels[parcels['Type'].isin(self.types)]

        if len(parcels['Type'].unique()) > 1:
            size = int(len(parcels) / len(parcels['Type'].unique()))
            parcels['Type'] = np.nan

            ### Select parcels for duplexes and join with closest parcel
            if 'Duplex, Triplex & Fourplex' in self.types:
                dup = np.random.choice(parcels.index, [int(size / 2)])
                parcels.loc[dup, 'Type'] = 'Duplex, Triplex & Fourplex'

                # For each parcel get closest of duplexes
                parcels['centroids'] = parcels.centroid
                parcels['closest'] = [nearest_points(
                    parcels.loc[[j for j in parcels['id'] if j != i]].centroid.unary_union, parcels.loc[i, 'centroids']
                )[0] for i in parcels['id']]

                # Draw a line between closest parcels
                parcels['geometry'] = [LineString([centroid, closest]) for centroid, closest in
                                       zip(parcels['centroids'], parcels['closest'])]
                parcels.loc[:, ['geometry']].to_file('del_duplexes.geojson', driver='GeoJSON')

                # Filter lines intersecting with duplexes
                duplexes = parcels[parcels['Type'] == 'Duplex, Triplex & Fourplex']

                # Filter parcels intersecting with filtered lines
                parcels = self.neigh.parcels.gdf.copy()
                parcels = parcels[parcels['Type'].isin(self.types)]
                join = gpd.sjoin(parcels, duplexes.loc[:, ['geometry']])
                duplexes = gpd.GeoDataFrame({
                    'id': [i for i in join['index_right'].unique()],
                    'geometry': [join.loc[join['index_right'] == i, ['geometry']].buffer(1).unary_union.buffer(-1) for i
                                 in join['index_right'].unique()]})

                columns = [col for col in parcels.columns if col not in ['Type', 'geometry']]
                duplexes.index = parcels.loc[parcels['id'].isin(list(join['index_right'].unique())), columns].index
                duplexes.loc[:, columns] = parcels.loc[parcels['id'].isin(list(join['index_right'].unique())), columns]
                duplexes['Type'] = 'Duplex, Triplex & Fourplex'

                # Remove duplexes and join with parcels
                parcels = pd.concat([parcels.drop(join['id']), duplexes])
                parcels.loc[[i for i in parcels.index if i not in dup], 'Type'] = np.nan

            ### Randomly pick mini mid-rise lots
            if 'Mini Mid-Rise' in self.types:
                mmr = np.random.choice(parcels[parcels['Type'].isna()].index, [size])
                parcels.loc[mmr, 'Type'] = 'Mini Mid-Rise'

            ### Randomly pick live/work lots
            if 'Live/Work' in self.types:
                lwk = np.random.choice(parcels[parcels['Type'].isna()].index, [size])
                parcels.loc[lwk, 'Type'] = 'Live/Work'

            if 'Single-Unit Detached' in self.types:
                if parcels['Type'].isna().sum() > 0:
                    parcels.loc[parcels[parcels['Type'].isna()].index, 'Type'] = 'Single-Unit Detached'
        return parcels

    def assign_unit_types(self, r_seed=0):
        """
        Assign one unit type for each parcel based on unit mix
        :param r_seed: random seed
        :return: parcels GeoDataFrame with assigned unit types
        """
        np.random.seed(r_seed)

        parcels = self.neigh.parcels.gdf.copy()
        parcels = parcels[~parcels['Type'].isna()]

        # Set up unit mix properties
        unit_gdf = gpd.GeoDataFrame(self.unit_mix).sort_values('share')
        unit_gdf['total_area'] = 0
        unit_gdf['current_share'] = 0
        unit_gdf['share_diff'] = unit_gdf['share'] - unit_gdf['current_share']

        # Iterate over blocks to distribute unit types per parcel
        for bid in parcels['bid'].unique():
            block_parcels = parcels[parcels['bid'] == bid].copy()

            # Select unit with highest share difference
            unit_gdf = unit_gdf.sort_values(['share_diff', 'current_share'])

            # Iterate over block parcels
            for pid in block_parcels['id']:
                highest_diff_type = list(unit_gdf['type'])[0]

                # Assign type with highest difference to current parcel
                parcels.loc[parcels['id'] == pid, 'unit_type'] = highest_diff_type
                block_parcels.loc[block_parcels['id'] == pid, 'unit_type'] = highest_diff_type

                # Update area shares and differences
                unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'total_area'] = \
                    unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'total_area'].sum() + \
                    block_parcels.loc[block_parcels['id'] == pid, 'area'].sum()

                unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'current_share'] = (
                        unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'total_area'] /
                        sum(unit_gdf['total_area'])).values[0]
                unit_gdf['share_diff'] = unit_gdf['share'] - unit_gdf['current_share']
                unit_gdf = unit_gdf.sort_values('share_diff', ascending=False)

        # Create unit area
        parcels['unit_area'] = [unit_gdf[unit_gdf['type'] == value]['area'].values[0] for value in parcels['unit_type']]
        parcels = parcels.apply(pd.to_numeric, errors='ignore').set_geometry('geometry')
        return parcels

    def place_types(self, block, list_of_types, r_seed=0):
        np.random.seed(r_seed)
        uniques = []
        while uniques != list_of_types:
            block.gdf['Type'] = np.random.choice(list_of_types, size=len(block.gdf))
            uniques = list(block.gdf['Type'].unique())
            if (uniques == list_of_types) & (sum(block.gdf['Type'].isna()) == 0):
                break
        return block

    def update_block_fsr(self):
        units = self.units.copy()

        blocks = self.neigh.generate_blocks()

        units['area'] = units.area
        blocks['area'] = blocks.area

        blocks = SpatialAnalyst(blocks, units.loc[:, ['area', 'geometry']]).spatial_join(operations=['sum'])

        blocks['new_fsr'] = blocks['area_sum'] / blocks['area']
        return blocks

    def get_building_depths(self):
        """
        Calculate building depths to maximize fsr considering parcel width, building height and setbacks 

        :return:
        """

        bldg_form = self.bldg_form.copy()
        parcels = self.neigh.parcels.gdf.copy()

        return

    def get_min_unit_area(self):
        return min([d['area'] for d in self.unit_mix])

    def get_max_unit_area(self):
        return max([d['area'] for d in self.unit_mix])

    def get_parcel_depth(self):
        gdf = self.neigh.parcels.gdf.copy()
        gdf['depth'] = [ctr.distance(bound) * 2 for ctr, bound in zip(gdf.centroid, gdf.boundary)]
        return gdf

    def height_from_fsr(self, bld_type):
        """
        Estimates building height to maximize FSR given its current footprint.
        :param bld_type:
        :return:
        """

        parcels = self.zone.neigh.parcels.gdf.copy()
        parcels['area'] = parcels.area

        if bld_type in ['Courtyard', 'Detached']:
            assert 'pid' in self.units.columns, AssertionError("'pid' column not found in units layer")
            parcels['area_units'] = [self.units[self.units['pid'] == i].area.sum() for i in parcels.index]
            remaining_fsr = parcels['fsr'] - (parcels['area_units'] / parcels['area'])
            return pd.Series([0 if i == np.inf else i
                              for i in (remaining_fsr * parcels['area']) / parcels['area_units']])

        elif bld_type in ['Podium', 'Cascading']:
            assert 'pid' in self.podium.columns, AssertionError("'pid' column not found in podium layer")
            assert 'pid' in self.tower.columns, AssertionError("'pid' column not found in tower layer")

            parcels['area_podium'] = [self.podium[self.podium['pid'] == i].area.sum() for i in parcels.index]
            parcels['area_tower'] = [self.tower[self.tower['pid'] == i].area.sum() for i in parcels.index]

            if bld_type == 'Podium':
                remaining_fsr = parcels['fsr'] - (parcels['area_podium'] / parcels['area'])
                n_storeys = ((remaining_fsr * parcels['area']) / parcels['area_tower']).replace(np.inf, 0)
                self.tower['storeys'] = [int(n_storeys.loc[self.tower.loc[i, 'pid']]) for i in self.tower.index]
                return self.tower['storeys']

            elif bld_type == 'Cascading':
                remaining_fsr = self.development.fsr - (parcels['area_podium'] / parcels['area'])
                return (((remaining_fsr * 1.618) * parcels['area']) / parcels['area_tower']).replace(np.inf, 0)

    def generate_hall(self, tp=None, approach='centroid', area=50):
        """
        Generate circulation halls according to a certain approach.
        :param tp: Value within Type column of parcels parameter
        :param approach: 'centroid'
        :param area: Approximate hall area
        :return:
        """
        gdf = self.neigh.parcels.gdf.copy()
        if tp is not None: gdf = gdf[gdf['Type'] == tp].copy()

        if approach == 'centroid':
            # Snap parcel centroids to court
            court_facade = Shape(gdf).extract_inner_rings().unary_union
            doors = gpd.GeoDataFrame({'geometry': [nearest_points(geom, court_facade)[1] for geom in gdf.centroid]},
                                     crs=gdf.crs)

            # Extract minimum rotated rectangle
            recs = gpd.GeoDataFrame({'geometry': [geom.minimum_rotated_rectangle for geom in gdf['geometry']]},
                                    crs=gdf.crs)
            # recs['side'] = [math.sqrt(i) for i in recs.area]
            # recs['scale'] = 7 / recs['side']
            recs['scale'] = area / recs.area

            # Scale down minimum rectangle to create entrance hall
            halls = recs.copy()
            halls['geometry'] = [scale(geom, xfact=factor, yfact=factor, origin=origin) for geom, factor, origin in
                                 zip(halls['geometry'], halls['scale'], doors['geometry'])]
            halls['area'] = halls.area
            # halls = halls[halls.reset_index().area < (0.4 * gdf.reset_index().area)]
            return halls

    def generate_units(self, fixed_height=True, min_parcel_area=200):
        """
        Generate buildings on parcels.
        :param min_parcel_size: Minimum parcel size in square meters
        :return: Buildings GeoDataFrame with height column
        """

        CEILING = 3
        MIN_UNIT_DEPTH = 5
        PODIUM_STOREYS = 3
        N_TOWERS = 1
        TOWER_AREA = 400
        CORRIDOR_WIDTH = 1.5

        assert len(self.neigh.parcels.gdf) > 0, AssertionError("Empty parcels")
        assert 'Type' in self.neigh.parcels.gdf.columns
        assert self.envelopes is not None, AssertionError("Envelopes object is null")
        assert len(self.neigh.parcels.gdf['pid'].unique()) == len(self.neigh.parcels.gdf), \
            AssertionError("pid not unique in parcels layer")

        start_time = time.time()
        parcels = self.neigh.parcels.gdf.copy()
        parcels = parcels[~parcels['Type'].isna()]
        if 'land_use' not in parcels.columns:
            parcels['land_use'] = np.nan
        all_units = gpd.GeoDataFrame(columns=parcels.columns, geometry='geometry', crs=parcels.crs)

        # Get envelope information from zone
        max_height = self.envelopes['standard'].boxes[0].height

        # Comply with minimum front setback requirements
        if self.whole_block:
            self.neigh.blocks.gdf = self.neigh.parcels.gdf
            self.blocks = Blocks(self.neigh.blocks.gdf)
        else:
            self.neigh.blocks.gdf = self.neigh.generate_blocks()  # !!!

        # Subtract setbacks and filter parcel areas
        parcels_gdf = self.subtract_setbacks()  # !!! very time consuming task
        parcels_gdf = parcels_gdf[parcels_gdf.area > min_parcel_area]
        self.neigh.parcels.gdf = parcels_gdf

        if sum(self.neigh.parcels.gdf.area) > 10:
            if len(self.neigh.parcels.gdf) > 0:
                court_types = ['Courtyard', 'Podium', 'Cascading']
                parcels = self.neigh.parcels.gdf.copy()

                print(f"Preprocessing for unit generation finished ({int(time.time() - start_time)}s)")
                for tp in parcels['Type'].unique():
                    start_time = time.time()

                    # Extract urban form information according to type
                    form = self.bldg_form[self.types.index(tp)]
                    bld_depth = form['BldgD']
                    n_storeys = form['Storeys']

                    # Reclassify type if building depth is too small
                    if (bld_depth < MIN_UNIT_DEPTH) and (tp in ['Courtyard', 'Podium', 'Cascading']):
                        parcels.loc[parcels['Type'] == tp, 'Type'] = 'Detached'
                        tp = 'Detached'
                    self.neigh.parcels.gdf = parcels.copy()

                    if tp not in court_types:
                        frontages = self.get_parcel_frontages()  # !!! time consuming task
                    else:
                        frontages = gpd.GeoDataFrame()
                    if (len(frontages) > 0) or (tp in court_types):
                        parcels = self.get_parcel_depth()
                        self.neigh.parcels.gdf = parcels

                        # Generate townhouse units based on largest frontage line and number of storeys
                        if tp in ['Townhouse', 'Live/Work']:
                            twh = parcels[parcels['Type'] == tp].copy()
                            if len(twh) > 0:

                                # Get parcel depth based on distance from centroid to boundary
                                twh.loc[twh['depth'] > MIN_UNIT_DEPTH, 'depth'] = twh['depth'] - MIN_UNIT_DEPTH
                                twh.loc[twh['depth'] < MIN_UNIT_DEPTH, 'depth'] = MIN_UNIT_DEPTH
                                twh.loc[twh['depth'] > 15, 'depth'] = 15

                                # Estimate unit frontage based on defined depth
                                twh['unit_frontage'] = twh['unit_area'] / twh['depth'] / n_storeys
                                twh.loc[twh['unit_frontage'] < 3, 'unit_frontage'] = 3

                                # Extract parcel frontages
                                frontages = frontages.sort_values(by=['pid', 'length'], ascending=False)

                                # Get largest frontage line
                                twh['frontage'] = [list(frontages[frontages['pid'] == pid]['geometry'])[0] if len(
                                    frontages[frontages['pid'] == pid]) > 0 else None for pid in twh['pid']]
                                twh['front_length'] = [geom.length if geom is not None else 0 for geom in
                                                       twh['frontage']]

                                # Estimate number of units
                                twh['n_units'] = (twh['front_length'] / twh['unit_frontage']).astype(int)
                                twh = twh[twh['n_units'] > 0]

                                if len(twh) > 0:
                                    twh = twh.reset_index(drop=True)
                                    twh['front_div'] = divide_line_by_count(twh['frontage'], twh['n_units'])

                                    parts = gpd.GeoDataFrame({
                                        'geometry': [div for parts in twh['front_div'] for div in parts],
                                        'root': [t for t, parts in enumerate(twh['front_div']) for i, div in
                                                 enumerate(parts)],
                                        'path1': [i for t, parts in enumerate(twh['front_div']) for i, div in
                                                  enumerate(parts)]
                                    })

                                    twh['offset'] = [MultiLineString(list(Shape(parts[parts['root'] == t]).offset_in(
                                        twh.loc[t, 'depth'], twh.loc[t, 'geometry']).geometry)) for t in twh.index]

                                    all_twh_units = gpd.GeoDataFrame({
                                        'geometry': [MultiLineString([inline, outline]).minimum_rotated_rectangle
                                                     for original, offset in zip(twh['front_div'], twh['offset'])
                                                     for inline, outline in zip(original, offset)],
                                        'root': [i for i, original, offset in
                                                 zip(twh.index, twh['front_div'], twh['offset'])
                                                 for inline, outline in zip(original, offset)],
                                        'id': [i for i, original, offset in
                                                 zip(twh['id'], twh['front_div'], twh['offset'])
                                                 for inline, outline in zip(original, offset)]
                                    })
                                    all_twh_units['land_use'] = 'Residential'
                                    all_twh_units = Shape(all_twh_units).stack(spacing=CEILING, n_stacks=2)
                                    all_units = pd.concat([all_twh_units, all_units])

                                    assert 'id' in all_units.columns, KeyError("'id' column not found")

                                    """
                                    for i in list(twh.index):
                                        # Divide frontage line according to number of units
                                        st = time.time()
                                        parts = gpd.GeoDataFrame(
                                            {'geometry': divide_line_by_count(twh.loc[i, 'frontage'], twh.loc[i, 'n_units'])},
                                            crs=twh.crs)
                                        parts['id'] = parts.index

                                        # Get start point of each part to define townhouse access point
                                        # parts['access'] = [Point(geom.xy[0][0], geom.xy[1][0]).buffer(1) for geom in parts['geometry']]
                                        # accesses = parts.copy().set_geometry('access').drop('geometry', axis=1)

                                        # Offset unit frontages and generate unit shape
                                        parts_offset = Shape(parts).offset_in(twh.loc[i, 'depth'], twh.loc[i, 'geometry'])

                                        # Create minimum rotated rectangle to generate unit and assign unit height
                                        for j in list(set(parts['id']).intersection(parts_offset['id'])):
                                            l = len(all_units)
                                            all_units.loc[l, :] = twh.loc[i, :]
                                            original = parts.loc[parts['id'] == j, 'geometry'].values[0]
                                            offset = parts_offset.loc[parts_offset['id'] == j, 'geometry'].values[0]
                                            geom = MultiLineString([original, offset]).minimum_rotated_rectangle
                                            all_units.loc[l, 'geometry'] = geom
                                            all_units.loc[l, 'geometry'] = Polygon([tuple(
                                                list(coords) + [3]) for coords in all_units.loc[l, 'geometry'].boundary.coords])
                                            if tp == 'Townhouse':
                                                access = get_point(parts.loc[j, 'geometry'], 'start')
                                                all_units.loc[l + 1, 'geometry'] = geom.difference(
                                                    access.intersection(geom).minimum_rotated_rectangle.buffer(0.1, join_style=2))
                                            else:
                                                all_units.loc[l + 1, 'geometry'] = geom
                                                all_units.loc[l + 1, 'land_use'] = 'Commercial'
                                            all_units.loc[[l, l + 1], 'height'] = CEILING
                                    """

                                else:
                                    print("No frontage found for parcel, no townhouse unit generated")

                                all_units['land_use'] = all_units['land_use'].fillna('Residential')

                                if 'frontage' in all_units.columns:
                                    all_units = all_units.drop('frontage', axis=1)
                                all_units = all_units.apply(pd.to_numeric, errors='ignore')
                                all_units = all_units.set_geometry('geometry', crs=parcels.crs)

                            else:
                                print("No Townhouse value in Type array")
                                # # Create commercial bottom for live/work units
                                # if tp == 'Live/Work':
                                #
                                #   all_units.loc[all_units['Type'] == tp, 'geometry'] = all_units.buffer(
                                #       -0.05, cap_style=2, join_style=2)
                                #
                                #     for lw_unit in list(all_units[all_units['Type'] == 'Live/Work'].index):
                                #         all_units = pd.concat([all_units, all_units.loc[[lw_unit], :]], ignore_index=True)
                                #         l = len(all_units)
                                #         all_units.loc[l-1, 'height'] = 3
                                #         all_units.loc[l-1, 'land_use'] = 'Commercial'
                                #         all_units.loc[l-1, 'geometry'] = all_units.loc[lw_unit, 'geometry'].buffer(0.05, cap_style=2, join_style=2)

                        elif tp in ['Detached']:
                            det_pcl = parcels[parcels['Type'] == tp].copy()
                            if len(det_pcl) > 0:
                                if self.zone.max_coverage is None:
                                    self.zone.max_coverage = 0.6
                                det_pcl['geometry'] = [scale(geom, self.zone.max_coverage, self.zone.max_coverage)
                                                       for geom in det_pcl['geometry']]
                                det_frt = frontages[frontages['Type'] == tp].copy()

                                assert 'pid' in det_pcl.columns, AssertionError("'pid' column not found")

                                # Generate entrance hall in largest frontage parcel
                                corridors = gpd.GeoDataFrame(crs=det_pcl.crs)
                                for pid in det_pcl['pid'].unique():
                                    if 'pid' in det_frt['pid']:
                                        geom = det_pcl.loc[det_pcl['pid'] == pid, 'geometry']
                                        access_pt = list(det_frt[det_frt['pid'] == pid].sort_values(
                                            'length', ascending=False).centroid)[0]
                                        intersection = geom.intersection(access_pt.buffer(3))
                                        circulation = [g.minimum_rotated_rectangle.buffer(2, join_style=2) for g in
                                                       intersection]
                                        det_pcl.loc[det_pcl['pid'] == pid, 'geometry'] = geom.subtract(
                                            MultiPolygon(circulation))
                                        corridors.loc[len(corridors), 'geometry'] = MultiPolygon(circulation)

                                # Subdivide remaining into spaces according to average unit area
                                det_pcl = Shape(det_pcl).subdivide_obb(max_area=max(det_pcl['unit_area']))
                                det_pcl = pd.concat([det_pcl, corridors]).reset_index(drop=True)
                                det_pcl = det_pcl.dropna(axis=1)
                                det_pcl = gpd.overlay(det_pcl, parcels[parcels['Type'] == tp].copy().loc[:,
                                                               ['geometry'] + [col for col in parcels.columns if col not in det_pcl.columns]])

                                self.units = det_pcl
                                if not fixed_height:
                                    n_storeys = self.height_from_fsr(bld_type=tp).apply(np.floor)
                                else:
                                    n_storeys = max_height / CEILING

                                det_pcl['land_use'] = 'Residential'
                                for i in n_storeys.index:
                                    stacked = Shape(det_pcl.loc[det_pcl['pid'] == i, :]).stack(spacing=CEILING,
                                                                                               n_stacks=n_storeys[i])
                                    all_units = pd.concat([all_units, stacked]).reset_index(drop=True)

                                    assert 'id' in all_units.columns, KeyError("'id' column not found")

                        elif tp in ['Courtyard', 'Podium', 'Cascading']:
                            gdf = parcels[parcels['Type'] == tp].copy()
                            if len(gdf) > 0:

                                self.neigh.join_block_id()
                                gdf = self.neigh.parcels.gdf.copy()
                                gdf = gdf[gdf['Type'] == tp]

                                # Rewrite this with min unit area as an array
                                if not fixed_height:
                                    bld_depth = -gdf['court_depth'] + CORRIDOR_WIDTH

                                unit_width = self.get_min_unit_area() / bld_depth

                                merged_units = Shape(gdf).subdivide_offset(
                                    width=unit_width, depth=bld_depth, outwards=False)
                                merged_units = Shape(merged_units).divorce()
                                merged_units = Shape(merged_units, reset_index=True).get_rectangularity()

                                # Assemble small units (merge small to closest unit)
                                # Classify units according to its size
                                units = merged_units.copy()
                                units['accuracy'] = units['area'] / units['unit_area']
                                units.loc[units['accuracy'] < 0.5, 'small'] = 1
                                units = units[units['small'] != 1]

                                # Merge and subdivide non-rectangular units
                                merged_units = units.copy()
                                non_rec = merged_units[merged_units['rectangularity'] < 0.7]
                                if len(non_rec) > 1:
                                    assembled = Shape(non_rec).assemble_to_area(self.get_max_unit_area())
                                else:
                                    assembled = non_rec
                                merged_units = gpd.GeoDataFrame(
                                    pd.concat([merged_units[merged_units['rectangularity'] >= 0.7], assembled]),
                                    crs=gdf.crs)

                                # Create corridors
                                gdf_buff = gdf.copy()
                                unit_depth = (bld_depth - CORRIDOR_WIDTH)
                                gdf_buff['geometry'] = gdf_buff.buffer(-unit_depth)
                                cor = gpd.overlay(merged_units[merged_units['area'] > 1],
                                                  gdf_buff.loc[:, ['geometry']], how="intersection")
                                merged_units = gpd.overlay(merged_units[merged_units['area'] > 1],
                                                           gdf_buff.loc[:, ['geometry']], how="difference")
                                cor['Type'] = 'Corridor'
                                merged_units = Shape(pd.concat([merged_units, cor]),
                                                     reset_index=True).divorce().reset_index(drop=True)

                                """
                                hall_method = False
                                if hall_method:
                                    halls = self.generate_hall(tp=tp)

                                    # Remove entrance hall from building footprint
                                    gdf = gpd.overlay(gdf, halls, how="difference")
                                    gdf['area'] = gdf.area

                                    # Calculate number of units
                                    gdf['n_units'] = (gdf['area'] / gdf['unit_area']).apply(np.floor)
                                    gdf = gdf[gdf['n_units'] > 0]

                                    # Divide remaining footprints according to number of units
                                    gdf['div_size'] = gdf['area'] / gdf['n_units'] / bld_depth
                                    units = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=gdf.crs)
                                    for i in gdf.index:
                                        unit_area = gdf.loc[i, 'unit_area']
                                        if gdf.loc[i, 'area'] > unit_area:
                                            grid = Shape(gdf.loc[i, 'geometry']).subdivide_parallels(
                                                gdf.loc[i, 'div_size'])
                                            for col in gdf.columns:
                                                if col != 'geometry':
                                                    grid.loc[:, col] = gdf.loc[i, col]
                                            units = pd.concat([units, grid])

                                    units['pid'] = units['id']
                                    units = units.reset_index(drop=True)
                                    units['id'] = units.index
                                    units['area'] = units.area

                                    # Classify units according to its size
                                    units['accuracy'] = units['area'] / units['unit_area']
                                    small = units[units['accuracy'] < 0.5].copy()

                                    # Test if a small unit is corner and break it
                                    small = Shape(small).is_convex()
                                    small_uu = small.unary_union
                                    small_cnv = small.loc[
                                        small['convex'], [col for col in small if col not in ['convex']]]
                                    for i in small.loc[~small['convex']].index:
                                        for geom in triangulate(small.loc[i, 'geometry']):
                                            if geom.centroid.buffer(0.01).intersects(small_uu):
                                                j = max(small_cnv.index) + 1
                                                small_cnv.loc[j, :] = small.copy().loc[i, :]
                                                small_cnv.loc[j, 'geometry'] = geom

                                    not_small = units[~units['id'].isin(small['id'])]
                                    closest = Shape(small, reset_index=False).get_closest(not_small, reset_index=False)

                                    geoms = gpd.GeoDataFrame({'geometry': []}, crs=closest.crs)
                                    for ref_id in closest['id'].unique():
                                        base_ids = list(closest[closest['id'] == ref_id]['base_id'])
                                        geoms = pd.concat([geoms, (
                                            Shape(pd.concat(
                                                [closest[closest['id'] == ref_id], small[small['id'].isin(base_ids)]]),
                                                reset_index=True).dissolve())])

                                    # Join merged and unmerged units together
                                    not_joined = not_small[~not_small['id'].isin(list(closest['id'].unique()))]
                                    merged_units = pd.concat([not_joined, geoms]).reset_index(drop=True)
                                else:
                                    self.neigh.join_block_id()
                                    gdf = self.neigh.parcels.gdf.copy()
                                    gdf = gdf[gdf['Type'] == tp]

                                    # # Get unit width and divide units
                                    # unit_depth = (bld_depth - 1.5)
                                    # gdf = Shape(gdf).divorce().reset_index(drop=True)
                                    # gdf['geometry'] = [Polygon(geom.exterior) for geom in gdf['geometry']]
                                    # merged_units = Shape(gdf).subdivide_offset(width=int(unit_width), depth=bld_depth,
                                    #                                            outwards=False)

                                    merged_units = Shape(merged_units).divorce()
                                    merged_units = Shape(merged_units, reset_index=True).get_rectangularity()

                                    # Assemble small units (merge small to closest unit)
                                    # Classify units according to its size
                                    units = merged_units.copy()
                                    units['accuracy'] = units['area'] / units['unit_area']
                                    units.loc[units['accuracy'] < 0.5, 'small'] = 1
                                    units = units[units['small'] != 1]

                                    # Merge and subdivide non-rectangular units
                                    merged_units = units.copy()
                                    non_rec = merged_units[merged_units['rectangularity'] < 0.7]
                                    if len(non_rec) > 1:
                                        assembled = Shape(non_rec).assemble_to_area(self.get_max_unit_area())
                                    else:
                                        assembled = non_rec
                                    merged_units = gpd.GeoDataFrame(
                                        pd.concat([merged_units[merged_units['rectangularity'] >= 0.7], assembled]),
                                        crs=gdf.crs)

                                    # Create corridors
                                    gdf_buff = gdf.copy()
                                    gdf_buff['geometry'] = gdf_buff.buffer(-unit_depth)
                                    cor = gpd.overlay(merged_units[merged_units['area'] > 1],
                                                      gdf_buff.loc[:, ['geometry']], how="intersection")
                                    merged_units = gpd.overlay(merged_units[merged_units['area'] > 1],
                                                               gdf_buff.loc[:, ['geometry']], how="difference")
                                    cor['Type'] = 'Corridor'
                                    merged_units = Shape(pd.concat([merged_units, cor]),
                                                         reset_index=True).divorce().reset_index(drop=True)
                                """

                                # Join block id
                                self.units = self.neigh.join_block_id(merged_units)

                                if tp == 'Courtyard':
                                    # Stack according to number of floors
                                    if not fixed_height:
                                        n_storeys = self.height_from_fsr(bld_type=tp).astype(int)
                                    else:
                                        n_storeys = int(n_storeys)
                                    merged_units = Shape(merged_units).stack(spacing=3, n_stacks=n_storeys)
                                    merged_units['land_use'] = 'Residential'
                                    all_units = pd.concat([all_units, merged_units]).reset_index(drop=True)
                                    self.units = self.neigh.join_block_id(all_units)
                                    assert 'id' in all_units.columns, KeyError("'id' column not found")

                                elif tp in ['Podium', 'Cascading']:

                                    blocks = self.update_block_fsr()
                                    if self.development is None:
                                        self.development = Development(parcels)
                                        self.development.fsr = self.fsr['standard']
                                    dev_fsr = self.development.fsr
                                    blocks['tower_area'] = TOWER_AREA
                                    # ((dev_fsr - blocks['new_fsr']) * blocks.area) / n_storeys

                                    # if len(merged_units[merged_units['Type'] == 'Corridor']) > 0:
                                    #     halls = self.neigh.join_block_id(
                                    #         merged_units[merged_units['Type'] == 'Corridor'])
                                    units = self.neigh.join_block_id(self.units.copy())
                                    t_units = units[units['Type'] != 'Corridor']

                                    tower_units = gpd.GeoDataFrame({'geometry': [], 'area': []}, crs=units.crs)

                                    blocks['tower_root'] = [
                                        np.random.choice(list(t_units[t_units['bid'] == i].index), N_TOWERS)
                                        if i in t_units['bid'].unique() else None for i in blocks.index
                                    ]

                                    towers = gpd.GeoDataFrame({
                                        'bid': [t_units.loc[i, 'bid'] for sl in blocks['tower_root']
                                                if sl is not None for i in sl],
                                        'pid': [t_units.loc[i, 'pid'] for sl in blocks['tower_root']
                                                if sl is not None for i in sl],
                                        'geometry': [t_units.loc[i, 'geometry'] for sl in blocks['tower_root']
                                                     if sl is not None for i in sl]
                                    })
                                    towers['id'] = towers.index
                                    towers['area'] = towers.area

                                    tower_units = towers.copy()
                                    for j in towers.index:
                                        units_in_block = t_units['bid'] == towers.loc[j, 'bid']
                                        block_units = t_units[units_in_block]
                                        bid = block_units['bid'].unique()[0]
                                        actual_tower_area = tower_units[tower_units['bid'] == bid].area.sum()
                                        block_units[f'd2t{j}'] = [geom.distance(tower_units.unary_union.centroid)
                                                                  for geom in block_units['geometry']]
                                        while actual_tower_area < TOWER_AREA:
                                            adjacent = block_units[[geom.intersects(
                                                tower_units.buffer(-0.1).buffer(1, cap_style=2).unary_union)
                                                for geom in block_units.geometry]]
                                            adjacent = adjacent[~adjacent['id'].isin(tower_units['id'])]
                                            if len(adjacent) > 0:
                                                closest = Shape(tower_units).get_closest(adjacent)
                                                for k in closest.index:
                                                    if k in list(block_units['id']):
                                                        block_units = block_units[
                                                            ~block_units['id'].isin(closest['id'])]
                                                n = max(tower_units.index) + 1
                                                tower_units.loc[n, :] = closest.loc[
                                                                        list(closest.index)[0], :]
                                                tower_units.loc[n, 'bid'] = bid
                                                actual_tower_area = tower_units[tower_units['bid'] == bid].area.sum()
                                                if (actual_tower_area >= TOWER_AREA) or (
                                                        len(block_units) == 0):
                                                    break
                                            else:
                                                break

                                        """
                                        for i in blocks['id'].unique():

                                            # Randomly choose hall for tower circulation and find adjacent units
                                            block_halls = halls[halls['bid'] == i].copy()

                                            if len(block_halls) > 0:
                                                np.random.seed(self.r_seed)
                                                tower_halls = block_halls.loc[
                                                              np.random.choice(list(block_halls.index), N_TOWERS), :]

                                                units_copy = units.copy().reset_index(drop=True)
                                                block_units = units_copy[units_copy['bid'].isin(tower_halls['bid']) &
                                                                         (units_copy['Type'] != 'Corridor')].\
                                                    reset_index(drop=True)
                                                block_units['id'] = block_units.index
                                                if len(block_units) > 1:
                                                    for j in tower_halls.index:
                                                        tower_units = tower_halls.copy()
                                                        actual_tower_area = tower_units.area.sum()
                                                        block_units[f'd2t{j}'] = [
                                                            geom.distance(tower_halls.unary_union.centroid) for geom in
                                                            block_units['geometry']]
                                                        while actual_tower_area < TOWER_AREA:
                                                            adjacent = block_units[[geom.intersects(
                                                                tower_units.buffer(-0.1).buffer(1,
                                                                                                cap_style=2).unary_union)
                                                                for geom in block_units.geometry]]
                                                            adjacent = adjacent[~adjacent['id'].isin(tower_units['id'])]
                                                            if len(adjacent) > 0:
                                                                closest = Shape(tower_units).get_closest(adjacent)
                                                                for k in closest.index:
                                                                    if k in list(block_units['id']):
                                                                        block_units = block_units[
                                                                            ~block_units['id'].isin(closest['id'])]
                                                                n = max(tower_units.index) + 1
                                                                tower_units.loc[n, :] = closest.loc[
                                                                                        list(closest.index)[0], :]
                                                                tower_units.loc[n, 'bid'] = i
                                                                actual_tower_area = tower_units.area.sum()
                                                                if (actual_tower_area >= TOWER_AREA) or (
                                                                        len(block_units) == 0):
                                                                    break
                                                            else:
                                                                break
                                                else:
                                                    print("Single-unit development, skipping tower growth")

                                            else:
                                                print("No 'hall' unit found in block, building not generated")
                                        """

                                    # # Setback tower units
                                    # setback = units.unary_union.buffer(-1.5)
                                    # tower_units['geometry'] = [
                                    # geom.intersection(setback) for geom in tower_units['geometry']]

                                    podium_units = Shape(units).stack(spacing=CEILING, n_stacks=PODIUM_STOREYS)
                                    for col in ['id', 'pid', 'bid']:
                                        podium_units[col] = list(units.loc[podium_units['parent_id'], col])

                                    self.podium = podium_units
                                    self.tower = tower_units
                                    if (not fixed_height) and (len(self.tower) > 0):
                                        n_storeys = (self.height_from_fsr(bld_type=tp) / CEILING).astype(int)
                                    else:
                                        n_storeys = int(n_storeys)
                                    stacked_tower = Shape(tower_units).stack(spacing=CEILING, n_stacks=n_storeys)
                                    for col in ['id', 'pid', 'bid']:
                                        stacked_tower[col] = list(tower_units.loc[stacked_tower['parent_id'], col])
                                    stacked_tower['land_use'] = 'Residential'
                                    podium_units['land_use'] = 'Residential'
                                    tower_units = stacked_tower.copy()

                                    if tp == 'Cascading':
                                        for i in units[units['Type'] == tp][
                                            'bid'].unique():  # Iterate over cascading tower blocks

                                            b_units = units[units['bid'] == i].copy()

                                            b_tower_units = tower_units[tower_units['bid'] == i].copy()
                                            b_pod_units = podium_units[podium_units['bid'] == i]

                                            if (len(b_tower_units) > 0) and (len(b_pod_units) > 0):
                                                # Get distance from tower
                                                tower = Shape(b_tower_units).dissolve()
                                                tower_uu = tower.unary_union
                                                b_units['d2tower'] = [geom.distance(tower_uu) for geom in
                                                                      b_units.geometry]
                                                b_units['in_tower'] = [
                                                    geom.centroid.buffer(1).intersects(tower_uu) for
                                                    geom in b_units.geometry]
                                                mtn_units = b_units[
                                                    (~b_units['in_tower']) & (
                                                            b_units['Type'] != 'Corridor')].copy()

                                                max_tower_height = max(b_tower_units['height']) * n_storeys
                                                max_pod_height = max(b_pod_units['height']) * PODIUM_STOREYS

                                                if type(max_tower_height) in [float, int]:
                                                    max_tower_height = [max_tower_height]

                                                    if sum(max_tower_height) > 0:
                                                        max_tower_height = max(max_tower_height)  #

                                                    heights = list(
                                                        reversed(range(int((max_tower_height - max_pod_height)
                                                                           / CEILING))))

                                                    for j in heights:
                                                        to_stack = list(mtn_units.sort_values('d2tower').index)[:2]
                                                        peak_units = Shape(mtn_units.loc[to_stack, :]).stack(
                                                            spacing=CEILING,
                                                            n_stacks=j)
                                                        peak_units['height'] = peak_units['height'] + max_pod_height
                                                        # Subtract setback from peak units
                                                        boxes_df = pd.DataFrame({
                                                            'height': [b.height for b in
                                                                       self.envelopes['standard'].boxes],
                                                            'setback': [max([b.rear, b.front]) for b in
                                                                        self.envelopes['standard'].boxes]})
                                                        for height in peak_units['height'].unique():
                                                            boxes_df = boxes_df.sort_values('height')
                                                            setback = list(boxes_df.loc[boxes_df[
                                                                                            'height'] <= height, 'setback'])[0]
                                                            try:
                                                                peak_units.loc[peak_units[
                                                                                   'height'] == height, 'geometry'] = \
                                                                    [geom.intersection(
                                                                        gdf.unary_union.buffer(-setback))
                                                                        for geom in peak_units.loc[peak_units[
                                                                                                       'height'] == height, 'geometry']]  ### !!! Time consuming
                                                            except:
                                                                print( "Failed to comply with setbacks in Cascading tower generation")

                                                        podium_units = pd.concat([podium_units, peak_units])
                                                        mtn_units = mtn_units.drop(to_stack)
                                                    podium_units['land_use'] = 'Residential'
                                                    mtn_units['land_use'] = 'Residential'
                                                else:
                                                    print("Cascading not generated")
                                            else:
                                                print(f"No tower and podium units, "
                                                      f"Cascading not generated for block {i}")
                                    else:
                                        print("No tower units found, cascading type could not be generated")

                                    all_units = pd.concat([all_units, tower_units, podium_units]).reset_index(drop=True)
                                    self.units = self.neigh.join_block_id(all_units)

                                    assert 'id' in all_units.columns, KeyError("'id' column not found")

                                else:
                                    print("No corridor unit found, tower not generated")

                    else:
                        print(f"Parcel frontages not found for parcel(s) of type {tp}, units not generated")

                    print(f"{tp} generation finished ({int(time.time() - start_time)}s)")

            else:
                print("Development Zone warning: No parcels found to generate units on")
        else:
            print("Total parcel area smaller than 10 m2, units not generated")

        all_units['footprint_area'] = all_units.area
        all_units['pid'] = all_units['id']
        all_units['id'] = all_units.index
        all_units.crs = self.neigh.parcels.gdf.crs

        return all_units

    def densify(self, bldg_types, site_id=0, b_size=None, street_width=13, subdivide=True, design=True,
                fixed_height=True):
        """
        Densify neighbourhood according to parameters

        :param bldg_types: 
        :param site_id:
        :param b_size:
        :param street_width:
        :param subdivide:
        :param design:
        :param fixed_height: If false maximized height to optimal FSR
        :return:
        """
        site_gdf_raw = self.neigh.parcels.gdf.loc[self.neigh.parcels.gdf['id'].isin(site_id), :]

        assert self.unit_count is not None, AssertionError("Unit count is None, run Estimate tab")
        assert type(self.unit_mix) == list, TypeError("Unit mix is not a list")
        assert len(site_gdf_raw) > 0, AssertionError("Empty parcels layer, site id not found in 'id' column")

        start_time = time.time()
        subdivide_toggle = subdivide
        design_toggle = design
        neigh = self.neigh
        types_named = {t: i for i, t in enumerate(self.types)}
        site_gdf = site_gdf_raw.copy().to_crs(26910)
        print(f"Block and boundaries generated ({int(time.time() - start_time)}s)")

        start_time = time.time()
        # Subdivide into smaller elements
        if subdivide_toggle:
            lanes = []

            if sum(site_gdf.area) > b_size:
                blocks = Blocks(site_gdf, zone=self)

                # Subdivide large block into smaller parts
                max_area = b_size * 10000
                if list(site_gdf['Area'].values)[0] > max_area:
                    blocks.gdf = blocks.recursive_obb(max_area=max_area)

                    # Offset
                    blocks.gdf['geometry'] = [geom.buffer(-float(street_width / 2)) for geom in blocks.gdf['geometry']]
                    blocks.gdf = blocks.gdf.reset_index(drop=True)

                # Randomly choose building types
                blocks = self.place_types(blocks, bldg_types)

                # Subdivide blocks into parcels
                new_parcels = Parcels(gpd.GeoDataFrame({'geometry': []}))
                for tp in bldg_types:
                    bldg_depth = int(self.bldg_form[types_named[tp]]['BldgD'])
                    min_plot_width = int(self.bldg_form[types_named[tp]]['PrclW'])
                    if tp in ['Bungalow Court', 'Courtyard', 'Mid-Rise', 'Podium', 'Cascading']:
                        blocks_tp = blocks.gdf[blocks.gdf['Type'] == tp].copy()
                        shp = Shape(blocks_tp.to_crs(26910), reset_index=True)
                        shp.gdf.index = blocks_tp.index
                        new_parcels.gdf = pd.concat([new_parcels.gdf, shp.gdf.copy()])
                    else:
                        shp = Shape(blocks.gdf[blocks.gdf['Type'] == tp].to_crs(26910), reset_index=True)
                        shp.gdf = shp.subdivide_spine(width=float(min_plot_width))
                        # Assign lines in the middle of the block as lanes
                        lanes.append(shp.meridian)
                        shp.gdf.index = blocks.gdf[blocks.gdf['Type'] == tp].index

                        # Divide lane ways into parcels
                        if tp in ['Detached']:
                            for lane, i in zip(shp.meridian[0], shp.gdf.index):
                                divs = gpd.GeoDataFrame(
                                    {'geometry': divide_line_by_count(lane, int(lane.length / min_plot_width))},
                                    crs=shp.gdf.crs)
                                divs['geometry'] = divs.buffer(b_size * 10000, cap_style=2)
                                shp.gdf.at[i, 'geometry'] = MultiPolygon(list(
                                    Shape(gpd.overlay(divs, shp.gdf.loc[[i], ['geometry']])).divorce()['geometry']))

                        new_parcels.gdf = pd.concat([new_parcels.gdf, shp.gdf.copy()])

                new_parcels.gdf = Shape(new_parcels.gdf, reset_index=True).divorce()

                # Reassign zone streets and lanes based on new subdivided block
                if len(lanes) > 0:
                    self.lanes = gpd.GeoDataFrame({'geometry': pd.concat([l for sl in lanes for l in sl])})
                self.streets = Streets(pd.concat([blocks.streets.gdf,
                                                  gpd.GeoDataFrame({'geometry': [
                                                      site_gdf.buffer(8).boundary.simplify(2).values[0]]})]))

                # Exclude parcel that was subdivided and create new parcels object inside zone
                neigh.parcels.gdf = neigh.parcels.gdf[~neigh.parcels.gdf['id'].isin(list(site_gdf_raw['id']))]
                neigh.parcels = Parcels(pd.concat([neigh.parcels.gdf, new_parcels.gdf]))

                # Re-generate urban blocks and Chamfer blocks+parcels corners
                # existing_blocks = self.blocks.gdf
                self.blocks = Blocks(self.neigh.generate_blocks())  # !!! time consuming task
                self.blocks.gdf = self.chamfer_blocks()

                # Merge subdivided blocks to existing blocks and create new blocks object inside zone
                # self.blocks = Blocks(pd.concat([self.blocks.gdf, existing_blocks]))
                self.boundaries = self.generate_block_boundaries()  # !!! time consuming task

                # # Create parcels layer to append to Pydeck
                # site_gdf = gpd.overlay(
                #     site_gdf_raw.copy().to_crs(neigh.parcels.gdf.crs),
                #     neigh.parcels.gdf.dropna(subset=['geometry']), keep_geom_type=True).fillna(0).to_crs(4326)

                export = False
                if export:
                    self.blocks.gdf.to_file('del_blocks.geojson', driver='GeoJSON')
                    neigh.parcels.gdf.to_file('del_parcels.geojson', driver='GeoJSON')
                    gpd.GeoDataFrame({'geometry': [self.boundaries]}).to_file('del_boundaries.geojson',
                                                                              driver='GeoJSON')

        else:
            if self.whole_block:
                self.neigh.blocks = Blocks(self.neigh.parcels.gdf)
                self.blocks = Blocks(self.neigh.blocks.gdf)
            # Create block objects if subdivide mode is not activated
            self.neigh.blocks = Blocks(site_gdf, zone=self)
            # self.blocks.gdf = self.place_types(self.blocks.gdf, bldg_types)

        # Design building types
        if design_toggle:

            # Assign types to parcels according to previous subdivision and generate buildings according to each type
            mixed_block = ['Single-Unit Detached', 'Duplexes', 'Live/Work', 'Mini Mid-Rise']

            # Get number unit count for each type defined on unit_mix based on its share
            for unit in self.unit_mix:
                unit['count'] = int(round(self.unit_count * float(unit['share']), 0))

            # Create Development Zone
            dev = DevelopmentZone(self, types=self.types, unit_count=self.unit_count, unit_mix=self.unit_mix,
                                  bldg_form=self.bldg_form, development=self.development)
            dev.lanes = self.lanes

            if not subdivide_toggle:
                # Assign building type
                dev.neigh.parcels.gdf.loc[
                    dev.neigh.parcels.gdf['id'].isin(list(site_gdf_raw['id'])), 'Type'] = bldg_types

            # Filter only selected parcels
            dev.neigh.parcels.gdf = neigh.parcels.gdf[~neigh.parcels.gdf['Type'].isna()].copy()

            # Assign development types for mixed blocks
            dev.types = list(set(mixed_block).intersection(set(bldg_types)))
            if len(dev.neigh.parcels.gdf['Type'].unique()) > 1:
                mask = dev.neigh.parcels.gdf['Type'].isin(mixed_block)
                dev.neigh.parcels.gdf.loc[mask, 'Type'] = dev.mix_development_types().loc[mask, 'Type']

            # Assign unit types based on development types and unit mix
            self.neigh.parcels.gdf = self.neigh.join_block_id()
            dev.types = self.types
            dev.neigh.parcels.gdf.loc[~dev.neigh.parcels.gdf['Type'].isna(), ['unit_type', 'unit_area']] = \
                dev.assign_unit_types().loc[:, ['unit_type', 'unit_area']]

            # Divide townhouse parcels that are too large types
            parcels = dev.neigh.parcels.gdf.copy()
            parcels['depth'] = [geom.length if geom is not None else 0 for geom in
                                Shape(parcels).min_rot_rec()['smallest_segment']]
            flt = (parcels['depth'] > 30) & (parcels['Type'] == 'Townhouse')
            parcels.loc[flt, 'geometry'] = Shape(parcels[flt]).subdivide_spine(width=100)['geometry']
            parcels = Shape(parcels).divorce().reset_index(drop=True)
            dev.neigh.parcels.gdf = parcels

            # Divide Detached blocks into smaller parcels
            if len(parcels[parcels['Type'] == 'Detached']) > 0:
                det_pcl = Shape(parcels[(parcels['Type'] == 'Detached') & (parcels.area > 10)]).simplify(5)
                det_pcl = Shape(det_pcl).subdivide_spine(width=20)
                det_pcl = Shape(det_pcl).divorce().reset_index(drop=True)
                det_pcl['pid'] = det_pcl.index
                dev.neigh.parcels.gdf = pd.concat([parcels[parcels['Type'] != 'Detached'], det_pcl])

            print(f"Preprocessing for densification finished ({int(time.time() - start_time)}s)")

            # Generate development units according to assigned types
            dev.neigh.parcels.gdf = dev.neigh.parcels.gdf.reset_index(drop=True)
            dev.neigh.parcels.gdf['pid'] = dev.neigh.parcels.gdf.index
            buildings = dev.generate_units(fixed_height)  # !!! time consuming task

            # Filter buildings by area
            buildings = buildings[buildings.area > 1]
            buildings = buildings.dropna(subset=['geometry']).fillna(0)
            self.buildings = pd.concat([self.buildings, buildings])

            # Comply with setbacks
            if not fixed_height: self.buildings = self.comply_setbacks()

            return self.buildings

    def comply_setbacks(self):
        assert self.buildings is not None, AssertionError(f"{self.__class__.__name__}Error: Buildings parameter not found")
        assert self.blocks is not None, AssertionError(f"{self.__class__.__name__}Error: Blocks parameter not found")
        assert 'height' in self.buildings.columns, KeyError("'height' column not found in 'buildings' GeoDataFrame")
        assert len(self.zone.envelopes) > 0, AssertionError(
            f"{self.__class__.__name__}Error: Standard envelope not found")

        bld = self.buildings.copy()
        blocks = self.blocks.gdf.copy()
        boxes = self.zone.envelopes[list(self.zone.envelopes.keys())[0]].boxes
        bld.sindex

        assert len(boxes) > 0, AssertionError(
            f"{self.__class__.__name__}Error: Empty boxes envelopes within {self.zone.__class__.__name__}")

        # Select the box of each unit based on maximum height
        bld['box'] = [min([i if h < box.height else np.nan for i, box in enumerate(boxes)]) for h in bld['bottom_hgt']]

        # Comply with setbacks of such box
        for i, box in zip(bld[~bld['box'].isna()]['box'].unique(), boxes):
            set_blocks = blocks.copy()
            set_blocks.sindex
            set_blocks['geometry'] = set_blocks.buffer(-box.front)
            bld = pd.concat([bld[(bld['box'] != i) & (~bld['box'].isna())],
                            gpd.overlay(bld[bld['box'] == i], set_blocks.loc[:, ['geometry']])])
        return bld

    def test_court_capacity(self, depth):
        """
        Gets whether a parcel have the capacity for a courtyard building type given a certain building depth

        :param depth:
        :return:
        """
        gdf = self.neigh.parcels.gdf.copy()
        gdf['tcc_id'] = gdf.reset_index(drop=True).index
        bound_gdf = gdf.copy()
        bound_gdf['geometry'] = bound_gdf.boundary.buffer(depth)
        diff_gdf = gpd.overlay(gdf, bound_gdf.loc[:, ['geometry']], how='difference')
        flt1 = gdf['tcc_id'].isin(diff_gdf['tcc_id'])
        gdf.loc[flt1, 'court_capable'] = True
        gdf.loc[~flt1, 'court_capable'] = False
        gdf['court_capable'] = gdf['court_capable'].astype(bool)
        return gdf

    def test_densify(self):
        self.densify()
        return

    def plot_parcels(self):
        return self.neigh.parcels.gdf.plot()


class Envelope:
    def __init__(self, boxes):
        self.boxes = boxes
        return


class Box:
    def __init__(self, height=None, front=None, side=None, rear=None, rear_anchor='boundary', pitch_height=None):
        self.height = height
        self.front = front
        self.side = side
        self.rear = rear
        self.pitch_height = pitch_height
        self.rear_anchor = rear_anchor
        return

    def offset(self, polygon):
        return


class Trapeze:
    def __init__(self, anchor, angle, base_height, max_height=None):
        self.anchor = anchor
        self.angle = angle
        self.base_height = base_height
        self.max_height = max_height
        return


def connect_sql():
    db_server_info = {
        "host": 'localhost',
        "port": 5432,
        "dbname": 'urban_dash',
        "user": 'nichmar',
        "password": 'postgres'
    }

    conn_str = f"postgresql+psycopg2://{db_server_info['user']}:{db_server_info['password']}@{db_server_info['host']}:{db_server_info['port']}/{db_server_info['dbname']}"
    alchemyEngine = create_engine(conn_str, pool_recycle=3600)
    postgreSQLConnection = alchemyEngine.connect()
