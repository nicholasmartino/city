import pickle
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
from Network import Streets
# from Patterns import Skeleton
from ShapeTools import Shape, Analyst, divide_line_by_count
from shapely import ops
from shapely.geometry import Point, Polygon, MultiLineString
from sklearn.preprocessing import MinMaxScaler

pd.options.display.max_columns = 15
pd.options.display.max_rows = 6
pd.options.display.max_colwidth = 40
pd.options.display.width = 300


class Blocks:
    def __init__(self, gdf, parcels=None, streets=None, zone=None, crs=26910, bldg_spacing=10, verbose=False):

        if 'level_0' in gdf.columns: gdf = gdf.drop('level_0', axis=1).reset_index()
        gdf.crs = crs

        # Trim dissolved blocks with streets
        if streets is not None:
            width = 2
            streets.gdf['geometry'] = streets.gdf.buffer(width)
            print(streets.gdf.sindex)
            gdf = gpd.overlay(gdf, streets.gdf, how='difference').reset_index()

        # Create block id and join to buildings and parcels
        gdf['id'] = gdf.index
        if parcels is not None:
            parcels.gdf['block_id'] = gpd.sjoin(parcels.gdf, gdf.loc[:, ['geometry']], how="left").groupby(by='id')[
                'index_right'].first()
            parcels.buildings.gdf['block_id'] = \
            gpd.sjoin(parcels.buildings.gdf, gdf.loc[:, ['geometry']], how="left").groupby(by='id')[
                'index_right'].first()

            for i in gdf['id']:
                gdf.loc[gdf['id'] == i, 'n_parcels'] = len(parcels.gdf[parcels.gdf['block_id'] == i])
                gdf.loc[gdf['id'] == i, 'n_buildings'] = len(
                    parcels.buildings.gdf[parcels.buildings.gdf['block_id'] == i])

        self.gdf = gdf
        self.parcels = parcels
        self.crs = crs
        self.spacing = 10
        self.streets = streets
        if zone is not None: self.zone = zone
        if verbose: print("Blocks object created")
        self.verbose = verbose
        return

    def dimension(self):
        gdf = self.gdf.copy()

        gdf['area'] = [geom.area for geom in gdf['geometry']]
        gdf['perimeter'] = [geom.length for geom in gdf['geometry']]

        return gdf

    def shape(self):
        gdf = self.gdf.copy()

        # Get polygon-specific measures
        print("Processing polygon indicators")
        gdf = Shape(gdf).process()
        return gdf

    def recursive_obb(self, max_area=10000):
        gdf = self.gdf.copy()
        gdf['area'] = gdf.area

        # Get polygon-specific measures
        print("Recursive oriented bounding box procedural subdivision")
        sample = Shape(gdf[gdf['area'] > max_area])
        gdf_div = sample.subdivide_obb(max_area=max_area)

        # Get streets from subdivision if streets is None
        if self.streets is None: self.streets = Streets(
            gdf=gpd.GeoDataFrame({'geometry': sample.splitters}, crs=self.gdf.crs))
        return pd.concat([gdf[gdf['area'] <= max_area], gdf_div])

    def generate_buildings(self, type, unit_mix, width, depth):
        start_time = time.time()
        gdf = self.gdf.copy()

        # Build envelopes from zone
        zone = self.zone
        gdf['zone'] = zone.name
        max_height = zone.envelopes['standard'].boxes[0].height
        setback = zone.envelopes['standard'].boxes[0].front
        if setback == 0: setback = 1

        # Filter type and explode multigeometries (if they are multigeometries)
        gdf = gdf[gdf['Type'] == type]
        gdf['id'] = gdf.index
        if 'MultiPolygon' in [geom.geom_type for geom in gdf['geometry']]:
            parcels = Parcels(gpd.GeoDataFrame({
                'geometry': [j for sl in gdf['geometry'] for j in sl],
                'bid': [i for i, sl in zip(gdf['id'], gdf['geometry']) for j in sl]
            }), preprocess=False)
        else:
            parcels = Parcels(gpd.GeoDataFrame({
                'geometry': [j for j in gdf['geometry']],
                'bid': [0 for j in gdf['geometry']]
            }), preprocess=False)

        # Create parcel objects
        zone.parcels = parcels

        # Find frontage segment
        block_frontages = zone.find_block_frontages()

        segments = zone.get_segments_distances().sort_values('d2streets').reset_index(drop=True)
        frontages = []
        rears = []
        for pid in zone.parcels.gdf['id'].unique():
            # # Extract parcel segments
            parcel_segments = segments.loc[segments['pid'] == pid, :]['geometry']
            rear_geom = list(parcel_segments)[len(parcel_segments) - 1]

            frontage_geom = MultiLineString(list(block_frontages[block_frontages['pid'] == pid]['geometry']))
            frontages.append(frontage_geom)
            rears.append(rear_geom)

            # Assign length to parcels GeoDataFrame
            zone.parcels.gdf.loc[zone.parcels.gdf['id'] == pid, 'frontage_length'] = [frontage_geom.length]
            zone.parcels.gdf.loc[zone.parcels.gdf['id'] == pid, 'rear_length'] = [rear_geom.length]
            zone.parcels.gdf.loc[zone.parcels.gdf['id'] == pid, 'depth'] = [
                frontage_geom.centroid.distance(rear_geom.centroid)]

        # Assign frontage and rear geometries
        zone.parcels.gdf['frontage'] = frontages
        zone.parcels.gdf['rear'] = rears

        # Get maximum building depth based on zoning setback requirements
        ground_envelope = zone.envelopes['standard'].boxes[0]
        zone.parcels.gdf['max_b_depth'] = zone.parcels.gdf['depth'] - ground_envelope.front - ground_envelope.rear
        zone.parcels.gdf.loc[zone.parcels.gdf['max_b_depth'] > 10, 'max_b_depth'] = 10

        if type == 'Townhouse':

            # Get number of floors based on ceiling height (3m)
            floors = int(max_height / 3)

            # Calculate frontage length and number of units based on maximum depth and unit area
            for unit in unit_mix:
                # Equalize shares 0-1
                unit['share'] = float(unit['share']) / sum([float(u['share']) for u in unit_mix])

                # Calculate number of units of each type given the unit areas, type frontage and maximum depth
                buildable_area = zone.parcels.gdf[f"frontage_length"] * zone.parcels.gdf['max_b_depth']
                zone.parcels.gdf[f"{unit['type']}_count"] = (buildable_area / float(unit['area'])).astype(int)

                # Recalculate total area
                zone.parcels.gdf[f"{unit['type']}_total_area"] = zone.parcels.gdf[f"{unit['type']}_count"] * float(
                    unit['area'])

            # Get total number of units for each parcel
            unit_types = [t['type'] for t in unit_mix]

            # Set up unit mix properties
            unit_gdf = gpd.GeoDataFrame(unit_mix).sort_values('share')
            unit_gdf['total_area'] = 0
            unit_gdf['current_share'] = 0
            unit_gdf['share_diff'] = unit_gdf['share'] - unit_gdf['current_share']

            # Iterate over blocks to distribute unit types per parcel
            for bid in zone.parcels.gdf['bid'].unique():
                block_parcels = zone.parcels.gdf[zone.parcels.gdf['bid'] == bid]

                # Select unit with highest share difference
                unit_gdf = unit_gdf.sort_values(['share_diff', 'current_share'])

                # Iterate over block parcels
                for pid in block_parcels['id']:
                    highest_diff_type = list(unit_gdf['type'])[0]

                    # Assign type with highest difference to current parcel
                    zone.parcels.gdf.loc[zone.parcels.gdf['id'] == pid, 'type'] = highest_diff_type
                    block_parcels.loc[block_parcels['id'] == pid, 'type'] = highest_diff_type

                    # Update area shares and differences
                    unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'total_area'] = unit_gdf.loc[unit_gdf[
                                                                                                         'type'] == highest_diff_type, 'total_area'].sum() + \
                                                                                        block_parcels.loc[block_parcels[
                                                                                                              'id'] == pid, 'area'].sum()
                    unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'current_share'] = (
                            unit_gdf.loc[unit_gdf['type'] == highest_diff_type, 'total_area'] / sum(
                        unit_gdf['total_area'])).values[0]
                    unit_gdf['share_diff'] = unit_gdf['share'] - unit_gdf['current_share']
                    unit_gdf = unit_gdf.sort_values('share_diff', ascending=False)

            # Iterate over parcels to City units
            units = []
            all_units = gpd.GeoDataFrame()
            all_frontages = gpd.GeoDataFrame()

            # Offset frontages, divide lines into equal parts and connect frontage and offseted line

            units = Shape(zone.parcels.gdf).offset_divide(distance='max_b_depth', n_divisions='n_units')

            for i in zone.parcels.gdf['id'].unique():
                parcel = zone.parcels.gdf.iloc[i]

                # Divide parcel frontage into equal segments according to the number of units of its assigned type
                line = zone.parcels.gdf.loc[i, 'frontage']
                n_units = parcel[f"{parcel['type']}_count"]

                if n_units > 0:
                    parts = gpd.GeoDataFrame({'geometry': divide_line_by_count(line, n_units)})
                    parts['geometry'] = parts.buffer(parcel['max_b_depth'], cap_style=2)

                    # Buffer parcel block and offset to calculate setback
                    setback_gdf = gpd.GeoDataFrame(
                        {'geometry': [geom.buffer(-setback) for geom in
                                      zone.parcels.gdf.loc[zone.parcels.gdf['bid'] == parcel['bid'], 'geometry'].buffer(
                                          0.01)]})

                    # Buffer frontage line to maximum building depth
                    unit = gpd.overlay(parts, setback_gdf, how='intersection')
                    all_units = pd.concat([all_units, unit])

        """
                # Iterate over unit types
                for j, unit in enumerate(unit_mix):
                    if parcel['max_b_depth'] > 0:

                        approx_type_frontage = approx_type_area / parcel['max_b_depth']
                        print(f'{j} Type frontage - {approx_type_frontage}m')

                        # Assign parts until unit frontage larger than approximate type frontage
                        frontage_length = 0
                        if len(parts) > 0:
                            unit_frontage = [list(parts['geometry'])[0]]
                            parts = parts.drop(0).reset_index(drop=True)
                            while frontage_length < approx_type_frontage:
                                if len(parts) == 0: break
                                unit_frontage = MultiLineString(list(unit_frontage) + [list(parts['geometry'])[0]])
                                parts = parts.drop(0).reset_index(drop=True)
                                frontage_length = unit_frontage.length
                                if frontage_length >= approx_type_frontage:
                                    break



                            # Split
                            frontages = gpd.GeoDataFrame({'geometry': split(unit_frontage, type_count)})

                            # # Assign parts according to unit type shares (non-distributed facade)
                            # if j < (len(unit_mix) - 1):
                            #     n_parts = int(unit['share'] * len(parts))
                            #     unit['frontage'] = ops.linemerge(list(parts.loc[range(n_parts)]['geometry']))
                            #     parts = parts.drop(range(n_parts)).reset_index(drop=True)
                            # else:
                            #     unit['frontage'] = ops.linemerge(list(parts['geometry']))

                            # # Divide facade according to the number of units of such type
                            # mix_gdf = gpd.GeoDataFrame(unit_mix)
                            # type_count = zone.parcels.gdf.loc[i, f"{unit['type']}_count"]
                            # if type_count == 0: type_count == 1
                            # if type_count > 1:
                            #     frontages = gpd.GeoDataFrame({'geometry': split(unit['frontage'], type_count)})
                            # all_frontages = pd.concat([all_frontages, frontages])

                            units = frontages.copy()
                            units['geometry'] = units.buffer(building_depth, cap_style=2)
                            units = gpd.overlay(units, setback_gdf, how='intersection')
                            all_units = pd.concat([all_units, units])
                            # # Move frontage geometry to generate building footprint
                            # xoff = parcel['rear'].centroid.x - parcel['frontage'].centroid.x
                            # yoff = parcel['rear'].centroid.y - parcel['frontage'].centroid.y
                            # ratio = parcel['max_b_depth'] / parcel['depth']
                            #
                            # # If there's only one unit on parcel, move it, else divide and move it
                            # if type_count == 1:
                            #     units.append(affinity.translate(unit['frontage'], xoff, yoff))
                            # else:
                            #     for frontage in split(unit['frontage'], type_count):
                            #         units.append(MultiLineString([frontage, affinity.translate(frontage, xoff * ratio, yoff * ratio)]).minimum_rotated_rectangle)


        units = gpd.GeoDataFrame({'geometry': units})
        frontages['height'] = max_height / floors
        units.crs = zone.parcels.gdf.crs
        units['geometry'] = units.buffer(building_depth)
        units['geometry'] = units.difference(zone.parcels.gdf.unary_union.buffer(0.01).buffer(-zone.envelopes['standard'].boxes[0].front))

        # Set side yard to zero and rear yard to 5 to City Townhouses
        for box in zone.envelopes['standard'].boxes:
            box.front = 2
            box.rear = 5
            box.side = 0

        # Build 3D envelopes based on zoning prescriptions
        raw_parcels = zone.parcels.gdf
        envelopes = zone.build_envelopes()
        envelopes['Type'] = type
        """

        all_units.crs = zone.parcels.gdf.crs
        all_units['height'] = max_height / floors
        print(f"{type} generation time --- {(time.time() - start_time)}s ---")

        # gpd.GeoDataFrame({'geometry': frontages}).to_file('del_frontages.geojson', driver='GeoJSON')

        return all_units


class Parcels:
    def __init__(self, gdf, buildings=None, zone=None, crs=26910, preprocess=False, verbose=False, reset_index=True):
        """

        :param gdf: Parcels GeoDataFrame
        :param buildings: Buildings GeoDataFrame
        :param zone: Zone object
        :param crs: Coordinate reference system
        """
        if preprocess:
            # Clean self-intersecting polygons
            print("Cleaning self-intersecting geometries")
            gdf['geometry'] = [geom if geom.is_valid else geom.buffer(0) for geom in gdf['geometry']]
            gdf = gdf.set_geometry('geometry')

        if reset_index:
            gdf = gdf.reset_index(drop=True)
            gdf['id'] = gdf.index
        gdf['pid'] = gdf.index
        if gdf.crs is None:
            gdf.crs = crs
        gdf['area'] = gdf.area

        # Join parcel id to buildings
        if buildings is not None:
            buildings.gdf['polygon'] = buildings.gdf['geometry']
            buildings.gdf['geometry'] = buildings.gdf.centroid
            start_time = time.time()
            print(f"Joining information from buildings {buildings.gdf.sindex} to parcels {gdf.sindex}")
            buildings.gdf['parcel_id'] = gpd.sjoin(buildings.gdf, gdf.loc[:, ['geometry']], how="left", op="within",
                                                   lsuffix='', rsuffix='').groupby(by='id')['index_right'].first()
            buildings.gdf['geometry'] = buildings.gdf['polygon']
            buildings.gdf = buildings.gdf.drop(['polygon'], axis=1)
            print(f"Data from buildings joined to parcels in {(time.time() - start_time)/60} minutes")

        if zone is not None:
            self.zone = zone

        self.gdf = gdf
        self.buildings = buildings
        self.centroids = gdf.centroid
        self.boundaries = gdf.boundary

        if verbose:
            print("Parcels object created")
        self.verbose = verbose
        return

    def shape(self):
        gdf = self.gdf.copy()

        # Get polygon-specific measures
        print("Processing polygon indicators")
        gdf = Shape(gdf).process()
        return gdf

    def occupation(self):
        gdf = self.gdf.copy()
        buildings = self.buildings.gdf.copy()

        for i in gdf['id']:
            parcel_buildings = buildings[buildings['parcel_id'] == i]

            if len(parcel_buildings) > 0:
                gdf.at[i, 'coverage'] = (parcel_buildings['area'].sum()/gdf[gdf['id'] == i].area).values

        return gdf

    def all(self):
        self.gdf = self.shape()
        self.gdf = self.occupation()
        return self.gdf

    def recursive_obb(self, max_area=10000):
        gdf = self.gdf.copy()

        # Get polygon-specific measures
        print("Recursive oriented bounding box procedural subdivision")
        gdf_div = Shape(gdf[gdf['area'] > max_area]).subdivide_obb(max_area=max_area)
        return pd.concat([gdf[gdf['area'] <= max_area], gdf_div])


class Buildings:
    def __init__(self, gdf, group_by=None, gb_func=None, crs=26910, to_crs=None):
        gdf = gdf.reset_index(drop=True)

        # Set/Adjust coordinate reference system
        if gdf.crs is None: gdf.crs = crs
        if to_crs is not None: gdf = gdf.to_crs(to_crs)

        if group_by is not None:
            if gb_func is not None:
                gdf = gdf.groupby(group_by, as_index=False).agg(gb_func)

            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=to_crs)
            gdf = gdf.dissolve(by=group_by)

        # For City of Vancouver open data:
        if 'topelev_m' in gdf.columns: gdf['height'] = gdf['topelev_m'] - gdf['baseelev_m']

        # Get polygon-specific measures
        gdf['id'] = gdf.index

        self.gdf = gdf
        self.crs = crs
        return

    def convex_hull(self):
        gdf = self.gdf.copy()
        gdf['convex_hull'] = [geom.convex_hull for geom in gdf['geometry']]
        gdf['conv_area'] = [geom.area for geom in gdf['convex_hull']]
        gdf['conv_perim'] = [geom.length for geom in gdf['convex_hull']]

        return gdf.drop('convex_hull', axis=1)

    def bounding_box(self):
        gdf = self.gdf.copy()

        # Area and perimeter
        gdf['bound_box'] = [geom.minimum_rotated_rectangle for geom in gdf['geometry']]
        gdf['box_area'] = [geom.area for geom in gdf['bound_box']]
        gdf['box_perim'] = [geom.length for geom in gdf['bound_box']]

        if 'ftprt_perim' in gdf.columns: gdf['ftprt_compactness'] = gdf['ftprt_perim']/(4 * gdf['box_area'])

        # Dimensions
        side_a = [Point(geom.bounds[0], geom.bounds[1]).distance(Point(geom.bounds[2], geom.bounds[1])) for geom in gdf['bound_box']]
        side_b = [Point(geom.bounds[0], geom.bounds[1]).distance(Point(geom.bounds[0], geom.bounds[3])) for geom in gdf['bound_box']]
        gdf['box_width'] = [min(i, j) for i, j in zip(side_a, side_b)]
        gdf['box_length'] = [max(i, j) for i, j in zip(side_a, side_b)]
        gdf['box_w2l_ratio'] = gdf['box_width']/gdf['box_length']

        # Ratios
        gdf['r_boxarea'] = gdf['area']/gdf['box_area']
        gdf['r_boxperim'] = gdf['perimeter']/gdf['box_perim']

        return gdf.drop('bound_box', axis=1)

    def triangulate(self):
        gdf = self.gdf.copy()

        gdf['triangulation'] = [ops.triangulate(geom) for geom in gdf['geometry']]

        gdf['tri_n_triangles'] = [len(list_geom) for list_geom in gdf['triangulation']]
        gdf['tri_area_sum'] = [sum([geom.area for geom in list_geom]) for list_geom in gdf['triangulation']]
        gdf['tri_perim_sum'] = [sum([geom.length for geom in list_geom]) for list_geom in gdf['triangulation']]
        gdf['tri_range'] = [max([geom.length for geom in list_geom]) - min([geom.length for geom in list_geom]) for list_geom in gdf['triangulation']]

        return gdf.drop('triangulation', axis=1)

    def centroid(self):
        gdf = self.gdf.copy()

        gdf['centroid'] = [geom.centroid for geom in gdf['geometry']]

        variance = []
        mean_dists = []
        for geom, centroid in zip(gdf['geometry'], gdf['centroid']):
            v_distances = [Point(coord).distance(centroid) for coord in geom.exterior.coords]
            mean = sum(v_distances)/len(v_distances)
            mean_dists.append(mean)
            dist_mean_sqd = [pow((v_dst - mean), 2) for v_dst in v_distances]
            variance.append(sum(dist_mean_sqd) / (len(v_distances) -1))

        gdf['cnt2vrt_mean'] = mean_dists
        gdf['cnt2vrt_var'] = variance

        return gdf.drop('centroid', axis=1)

    def encl_circle(self):
        gdf = self.gdf.copy()
        return gdf

    def skeleton(self):
        skeletons = Skeleton(gdf=self.gdf, crs=self.crs)
        return skeletons

    def all(self):
        print("> Calculating all indicators for Buildings")
        self.gdf = self.convex_hull()
        self.gdf = self.bounding_box()
        self.gdf = self.triangulate()
        self.gdf = self.centroid()
        self.gdf = self.encl_circle()
        return gpd.GeoDataFrame(self.gdf, crs=self.crs)

    def get_gfa(self, ceiling=4):
        assert 'height' in self.gdf.columns, 'Assign height column to buildings GeoDataFrame'
        self.gdf['gfa'] = (self.gdf['height']/ceiling) * self.gdf.area
        return self.gdf


class Development:
    def __init__(self, parcel, units=None, fsr=None, net_gross=0.9, commission=0.03, hard_costs=None, soft_costs=0.15,
                 financing_rate=0.05, profit_target=0.15, occupancy_rate=0.98, operating_costs=6000, rent=None,
                 sell=60, cap_rate=0.05, max_height=None, site_coverage=None, current_units=None, crs=26910,
                 dem_cost=0):
        """

        :param parcel: GeoDataFrame with one row and Polygon geometry column
        :param units: DataFrame with four columns (Type, Ratio, Area, Price)
        :param fsr: Floor Space Ratio
        :param net_gross: Net/gross area ratio
        :param commission:
        :param hard_costs:
        :param soft_costs:
        :param dccs:
        :param financing_rate:
        :param profit_target:
        :param occupancy_rate:
        :param operating_costs: Operating costs per unit (utilities, management, insurance, repair, taxes)
        """

        self.site_area = parcel.to_crs(crs).area

        self.site_cover = site_coverage
        self.soft_costs = soft_costs
        self.hard_costs = hard_costs
        self.net_gross = net_gross

        if type(fsr) == str:
            self.fsr = parcel[fsr]
        else:
            self.fsr = fsr

        if type(dem_cost) == str:
            self.dem_cost = parcel[dem_cost]
        else:
            self.dem_cost = dem_cost

        if type(max_height) == str:
            self.max_height = parcel[max_height]
        else:
            self.max_height = max_height

        self.occupancy = occupancy_rate
        self.op_costs = operating_costs
        self.cap_rate = cap_rate

        if type(rent) == str:
            self.rent = parcel[rent]
        else:
            self.rent = rent

        self.sell = sell

        self.commission = commission
        self.financing_rate = financing_rate
        self.profit_target = profit_target

        if units is not None:
            self.units = units

        if current_units is not None:
            if type(current_units) == str:
                self.current_unit_count = parcel[current_units]
            else:
                self.current_unit_count = current_units
        return

    def dcc(self):
        # https://bylaws.vancouver.ca/bulletin/bulletin-development-cost-levies.pdf
        df = pd.DataFrame(self.practical_fsr(), columns=['p_fsr'])
        df.loc[df['p_fsr'] < 1.2, 'dcc'] = 70.6
        df.loc[(df['p_fsr'] >= 1.2) & (df['p_fsr'] < 1.5), 'dcc'] = 152.52
        df.loc[df['p_fsr'] >= 1.5, 'dcc'] = 305.37
        df['dcc'] = df['dcc'].fillna(150)
        return df['dcc']

    def revenue(self):
        return sum(self.units['share'] * self.units['price']) * self.marketable_area()

    def unit_count(self):
        return self.marketable_area() / sum(self.units['share'] * self.units['area'])

    def total_unit_count(self):
        return int(sum(self.unit_count()))

    def max_gfa(self):
        return self.site_area * self.practical_fsr()

    def marketable_area(self):
        return self.max_gfa() * self.net_gross

    def hc_tot(self):
        return (self.hard_costs * self.max_gfa()) + self.dem_cost

    def buildable_area(self):
        assert self.site_cover is not None, AssertionError("Maximum site coverage not defined")
        assert self.max_height is not None, AssertionError("Maximum height not defined")
        return (self.site_cover * self.site_area * self.max_height)/3

    def buildable_fsr(self):
        return self.buildable_area()/self.site_area

    # def get_ebit(self):
    #     return round((self.revenue() * self.commission) + self.hc_tot() + (self.hc_tot() * self.soft_costs) + (
    #                 self.dccs * self.total_unit_count()), 2)

    def get_total_costs(self):
        return self.hc_tot() + (self.soft_costs * self.hc_tot()) + (self.dcc() * self.max_gfa())
        # round(self.get_ebit() + (self.financing_rate * ((self.hc_tot() + self.soft_costs + self.dcc() / 2)), 2)

    def get_net_income(self):
        return round(((self.current_unit_count * self.occupancy) * self.rent * 12) - (self.op_costs * self.current_unit_count), 2)

    def value_rent(self):
        return round(self.get_net_income() / self.cap_rate, 2)

    def value_redevelop(self):
        return round(self.site_area * self.practical_fsr() * self.sell, 2)

    def value_land_residual(self):
        net_revenue = self.revenue() - (self.commission * self.revenue())
        subtotal = self.get_total_costs()
        interim_fin = self.financing_rate * subtotal
        total = subtotal + interim_fin
        profit = self.profit_target * self.revenue()
        return round(net_revenue - total - profit, 2)

    def get_n_stories_from_max_fsr(self):
        return

    def practical_fsr(self):
        assert self.site_area is not None, AssertionError("Site area not defined")
        assert self.fsr is not None, AssertionError("FSR not defined")
        return pd.concat([self.buildable_fsr(), pd.Series(self.fsr, index=self.buildable_fsr().index)], axis=1).min(axis=1)

    def print_params(self):
        print(self.__dict__)
        return


class Neighbourhood:
    def __init__(self, name='', parcels=None, buildings=None, streets=None, trees=None, blocks=None, boundary=None):
        """
        A set of buildings placed on parcels interconnected by streets.
        :param blocks:
        :param parcels: Parcels object
        :param buildings: Buildings object
        :param streets: Streets object
        :param trees: GeoDataFrame
        """
        self.name = name
        self.streets = streets
        self.parcels = parcels
        self.buildings = buildings
        self.centers = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=4326)

        self.open_space = None
        self.trees = trees
        self.blocks = blocks
        self.boundary = boundary
        return

    def get_gdf(self, layer):
        if layer == 'streets':
            assert (self.streets.gdf is not None)
            gdf = self.streets.gdf.copy().reset_index(drop=True)
        elif layer == 'open_space':
            assert (self.open_space is not None)
            gdf = self.open_space.copy().reset_index(drop=True)
        return gdf

    def get_current_fsr(self):
        """
        Get current floor space ratio by parcel based on buildings with height information.

        :return:
        """

        parcels = self.parcels.gdf.copy()
        buildings = self.buildings.gdf.copy()

        assert 'height' in self.buildings.gdf.columns, AssertionError("Height information not found in buildings layer")
        assert 'pid' in self.buildings.gdf.columns, AssertionError("'pid' column not found in buildings layer")

        parcels['gfa'] = [sum(buildings[buildings['pid'] == i].area) for i in parcels.index]
        parcels['fsr'] = parcels['gfa']/parcels.area
        return parcels

    def update_lng_lat_ids(self):
        self.centers['id'] = [f'{round(geom.x, 5)}_{round(geom.y, 5)}' for geom in self.centers['geometry'].centroid]
        return

    def add_center(self, coordinates):
        """
        Add a neighbourhood center to the centers GeoDataFrame according to the defined coordinates
        :param coordinates: List or tuple of longitude and latitude coordinates
        :return:
        """
        self.centers.at[len(self.centers), 'geometry'] = Point(coordinates)
        return

    def remove_center(self, index):
        """
        Remove a neighbourhood center based on a longitude/latitude index
        :param index:
        :return:
        """
        self.update_lng_lat_ids()
        self.centers = self.centers[self.centers['id'] != index]
        self.centers['geometry'] = self.centers.centroid
        return

    def buffer_centers(self, radius=10):
        """
        Buffer neighbourhood centers to transform them into polygons
        :param radius: Buffer radius in meters
        :return:
        """
        gdf = self.centers.to_crs(3857).copy()
        gdf['geometry'] = gdf.buffer(radius)
        return gdf.to_crs(4326)

    def export_centers_feather(self, merge_existing=True):
        """
        Write feather file from neighbourhood centers
        :param merge_existing: If true it will try to read an existing centers.feather file
        :return:
        """
        self.update_lng_lat_ids()

        if merge_existing:
            try:
                gdf = pd.concat([gpd.read_feather('centers.feather'), self.buffer_centers()])
            except:
                gdf = self.buffer_centers()
        else:
            gdf = self.buffer_centers()

        gdf = gdf.drop_duplicates(subset='id').reset_index(drop=True)
        gdf.to_feather('centers.feather')
        return

    def export_centers_pdk(self):
        return pdk.Layer(
            id='centers',
            type="GeoJsonLayer",
            data=self.buffer_centers(),
            extruded=False,
            getFillColor=[0, 0, 255, 100],
            get_line_color=[255, 255, 255, 255],
            auto_highlight=True,
            pickable=True)

    def assign_gradient_fsr(self, fsr_range, max_dist=500):
        """
        Assign maximum floor space ratio to parcels based on the proximity to neighbourhood centers
        :param fsr_range:
        :return:
        """

        fsr_range = [min(fsr_range), max(fsr_range)]
        gdf = self.parcels.gdf.to_crs(26910).copy()

        if len(self.centers) == 0:
            try: self.centers = gpd.read_feather('centers.feather')
            except: pass

        radii = list(range(1, int(max_dist/100)))
        for i, r in zip(radii, radii.__reversed__()):
            centers = self.centers.copy().to_crs(26910)
            centers['geometry'] = centers.buffer(r * 100)
            centers = Shape(centers).dissolve()
            gdf.loc[gdf['id'].isin(gpd.overlay(gdf, centers.loc[:, ['geometry']])['id']), 'rad'] = i
        gdf['rad'] = gdf['rad'].fillna(0)

        # Calculate distances to centers
        for i, center in enumerate(self.centers.to_crs(26910)['geometry']):
            gdf[f'd2_{i}'] = [1/geom.distance(center) if geom.distance(center) > 0 else 1 for geom in gdf['geometry']]
            gdf[f'inv_power_{i}'] = (1/(gdf[f'd2_{i}'] ** 2))
            gdf[f'NegExp03_{i}'] = [np.exp(((-0.3) * dist)) for dist in gdf[f'd2_{i}']]
            gdf[f'log_norm_{i}'] = [1/np.log(dist) for dist in gdf[f'd2_{i}']]

        gdf['dist_inv'] = list(gdf.loc[:, [col for col in gdf.columns if 'd2_' in col]].min(axis=1))
        gdf['dist_power'] = list(gdf.loc[:, [col for col in gdf.columns if 'inv_power' in col]].min(axis=1))

        gdf = gdf.sort_values(by='dist_power', ascending=True)
        gdf['dist_power_rev'] = list(gdf.sort_values(by='dist_power', ascending=False)['dist_power'])

        gdf = gdf.sort_index()

        gdf['dist_negexp'] = list(gdf.loc[:, [col for col in gdf.columns if 'NegExp03_' in col]].sum(axis=1))
        gdf['dist_log'] = list(gdf.loc[:, [col for col in gdf.columns if 'log_norm_' in col]].sum(axis=1))

        # Normalize data according to fsr range
        scaler = MinMaxScaler(fsr_range)
        scaler01 = MinMaxScaler()

        fsr_cols = ['dist_inv', 'dist_power', 'dist_power_rev', 'dist_negexp', 'dist_log', 'rad']

        for col in fsr_cols:
            for l in range(1, 6):
                if l > 1:
                    low_qtl = gdf[col].quantile(1 / l)
                    up_qtl = gdf[col].quantile(1 / (l - 1))
                    gdf.loc[
                        (gdf[col] < up_qtl) & (gdf[col] > low_qtl), f'{col}_qtl'] = l
                else:
                    gdf.loc[:, f'{col}_qtl'] = 1

        for col in fsr_cols + [f'{col}_qtl' for col in fsr_cols]:

            try:
                gdf.loc[:, [col]] = gdf.loc[:, [col]].replace(np.inf, 0)
                self.parcels.gdf.loc[:, [f'fsr_{col}']] = scaler.fit_transform(gdf.loc[:, [col]])
                self.parcels.gdf.loc[:, [f'fsrScaled_{col}']] = scaler01.fit_transform(gdf.loc[:, [col]]).round(4)
            except:
                pass
        return

    def check_id(self, layer='parcels'):
        if layer == 'blocks':
            gdf = self.blocks.gdf.copy()
        else:
            gdf = self.parcels.gdf.copy()

        if 'id' not in gdf.columns:
            gdf['id'] = gdf.reset_index(drop=True).index
        elif gdf['id'].isna().sum() > 0:
            gdf.loc[gdf['id'].isna(), 'id'] = [max(gdf['id']) + i for i in range(1, len(gdf[gdf['id'].isna()]))]
        return gdf

    def join_block_id(self, gdf=None):
        """
        Spatially join block id to parcels.
        :return:
        """

        if gdf is None:
            parcels_gdf = self.check_id('parcels')
            blocks_gdf = self.check_id('blocks')
            parcels_gdf.sindex
            blocks_gdf.sindex
            parcels_gdf = gpd.overlay(parcels_gdf, blocks_gdf.loc[:, ['id', 'geometry']]).rename(
                {'id_1': 'id', 'id_2': 'bid'}, axis=1)
            parcels_gdf = Neighbourhood(parcels=Parcels(parcels_gdf)).check_id('parcels')
            return parcels_gdf.loc[:, ~parcels_gdf.columns.duplicated()]

        else:
            assert len(gdf) > 0, AssertionError("Empty gdf, block id can not be joined")

            if 'bid' in gdf.columns:
                gdf = gdf.drop('bid', axis=1)
            blk_gdf = self.blocks.gdf.copy()
            blk_gdf.sindex
            blk_gdf['bid'] = blk_gdf['id']
            try:
                return gpd.overlay(gdf, blk_gdf.loc[:, ['bid', 'geometry']])
            except:
                pass

    def join_parcel_id(self, gdf):
        if 'pid' in gdf.columns:
            gdf = gdf.drop('pid', axis=1)
        pcl_gdf = self.parcels.gdf.copy()
        pcl_gdf['pid'] = pcl_gdf.index
        return gpd.overlay(gdf, pcl_gdf.loc[:, ['pid', 'geometry']])

    def join_open_space_id(self):
        gdf = self.parcels.gdf.copy()
        gdf['os_id'] = Shape(gdf).get_closest(self.open_space, reset_index=True)['ref_id']
        return gdf

    def generate_blocks(self, join_id=True):
        """
        Generates blocks geometries from parcels

        :return: blocks GeoDataFrame
        """

        buffer = self.parcels.gdf.buffer(3).unary_union
        if 'Multi' in buffer.__class__.__name__:
            geometry = [block.buffer(-3) for block in buffer]
        else:
            geometry = [buffer.buffer(-3)]
        blocks_gdf = gpd.GeoDataFrame({'geometry': geometry})
        blocks_gdf = Shape(blocks_gdf).divorce()
        blocks_gdf = blocks_gdf[blocks_gdf.area > 1]
        blocks_gdf['id'] = blocks_gdf.index
        blocks_gdf['geometry'] = [Polygon(geom) for geom in blocks_gdf.exterior]
        blocks_gdf.crs = self.parcels.gdf.crs
        blocks_gdf = blocks_gdf[~blocks_gdf.geometry.is_empty]

        try:
            assert len(blocks_gdf) > 0, AssertionError("Block generation returned no blocks")
        except:
            pass
        self.blocks = Blocks(blocks_gdf)

        if join_id:
            id_parcels = self.join_block_id()
            assert len(id_parcels) > 0, AssertionError("Block generation resulted in empty parcel on id join")

        return blocks_gdf

    def extract_open_space(self, parcel_buffer=1):
        """
        Subdivide empty space
        :return:
        """

        streets = self.streets.gdf.copy()
        streets = streets.unary_union.buffer(1).buffer(-0.99)

        parcels = self.parcels.gdf.copy()
        parcels = parcels.unary_union.buffer(parcel_buffer).buffer(-parcel_buffer)

        mmr = streets.minimum_rotated_rectangle.buffer(5)
        space = mmr.difference(parcels)
        space = gpd.GeoDataFrame({'geometry': [space]}, crs=self.streets.gdf.crs)
        return Shape(space).divorce()

    def estimate_canopy_cover(self, layer='streets'):
        gdf = self.get_gdf(layer)
        gdf['id'] = gdf.index

        trees_gdf = self.trees.copy()

        overlay = gpd.overlay(gdf, trees_gdf)
        overlay['area'] = overlay.area

        gdf['canopy_area'] = overlay.groupby('id').sum()['area']
        gdf['canopy_cover'] = gdf['canopy_area']/gdf.area
        gdf.loc[gdf['canopy_cover'] >= 0.5, 'Green'] = 1
        gdf.loc[gdf['canopy_cover'] < 0.5, 'Green'] = 0

        gdf = gdf.fillna(0)

        return gdf

    def count_trees(self, layer='streets'):
        """
        Count number of trees and canopy area whose centroids are within a defined layer
        :param layer: String, either 'streets' or 'open_space'
        :return:
        """
        gdf = self.get_gdf(layer)
        gdf['id'] = gdf.index

        trees_gdf = self.trees.copy()
        trees_gdf['count'] = 1
        trees_gdf['can_area'] = trees_gdf.area
        trees_gdf['geometry'] = trees_gdf.centroid.buffer(1)

        overlay = gpd.overlay(gdf, trees_gdf)
        grouped = overlay.groupby('id').sum()
        gdf['n_trees'] = grouped['count']
        gdf['can_area'] = grouped['can_area']
        gdf['can_cover'] = gdf['can_area']/gdf.area
        gdf = gdf.fillna(0)
        return gdf

    def convex_hull(self):
        return gpd.GeoDataFrame({'geometry': [
            pd.concat([self.streets.gdf, self.parcels.gdf]).unary_union.convex_hull]},
            crs=self.parcels.gdf.crs)

    def get_fsr(self):
        pcl = self.parcels.gdf.copy()
        bld = self.buildings.get_gfa().copy()

        pcl['fsr'] = Analyst(pcl).spatial_join(bld.loc[:, ['gfa', 'geometry']])['gfa_mean']/pcl.area
        return pcl

    def export(self, driver, crs, suffix=''):

        file_types = {
            'GPKG': '.gpkg',
            'ESRI Shapefile': '.shp',
            'GeoJSON': '.geojson'
        }

        def clean_gdf(gdf):
            geom = gdf['geometry']
            gdf.columns = [c.lower() for c in gdf.columns]
            gdf = gdf.drop('geometry', axis=1).T.drop_duplicates().T
            gdf['geometry'] = geom
            gdf = gdf.replace(True, 1)
            gdf = gdf.replace(False, 0)
            return gpd.GeoDataFrame(gdf)

        bld_gdf = clean_gdf(self.buildings.gdf.copy())
        pcl_gdf = clean_gdf(self.parcels.gdf.copy())
        # str_gdf = clean_gdf(self.streets.gdf.copy())

        bld_gdf.to_crs(crs).to_file(f'data/{self.name}{file_types[driver]}', layer=f'buildings{suffix}', driver=driver)
        pcl_gdf.to_crs(crs).to_file(f'data/{self.name}{file_types[driver]}', layer=f'parcels{suffix}', driver=driver)
        # str_gdf.to_crs(crs).to_file(f'{self.name}{file_types[driver]}', layer=f'streets{suffix}', driver=driver)
        return

    def dump_pkl(self):
        return pickle.dump(self, open(f'../data/{self.name}-neigh.pkl', 'wb'))

    def load_pkl(self):
        return pickle.load(open(f'../data/{self.name}-neigh.pkl', 'rb'))
