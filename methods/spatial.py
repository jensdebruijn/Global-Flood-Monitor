import math
import csv
from geopy.distance import great_circle
from psycopg2.extensions import AsIs


def buffer(pg, wkt, buffer_m, SRID=4326):
    wkt = f'SRID={SRID};' + wkt
    pg.cur.execute("""
        SELECT ST_AsText(ST_Buffer(ST_GeogFromText(%s), %s));
    """, (wkt, buffer_m))
    return pg.cur.fetchone()[0]

def union(pg, wkts):
    if len(wkts) == 1:
        return wkts[0]
    geometries = [
        f"ST_GeomFromText('{wkt}')"
        for wkt in wkts
    ]
    array = "ARRAY[" + ",".join(geometries) + "]"
    pg.cur.execute("""
        SELECT ST_AsText(ST_Union(%s))
    """, (AsIs(array), ))
    return pg.cur.fetchone()[0]

def add_SRID_to_wkt(wkt, SRID=4326):
    return f'SRID={SRID};' + wkt

def get_geojson_from_osm_element(elem):
    properties = elem.get("tags")
    properties['id'] = elem['id']
    geojson = {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "crs": {
                "type": "name",
                "properties": {
                    "name": "EPSG:4326"
                }
            }
        }
    }

    if elem["type"] == "node":
        geojson['geometry']['type'] = 'Point'
        geojson['geometry']['coordinates'] = [elem["lon"], elem["lat"]]
        return geojson

    elif elem["type"] == "way":
        geojson['geometry']['type'] = 'LineString'
        geojson['geometry']['coordinates'] = [
                [coords["lon"], coords["lat"]]
                for coords in elem["geometry"] 
            ]
        return geojson

    elif elem["type"] == "relation":
        # initialize result lists
        polygons = []
        poly = []
        points = []
        # conditions
        prev = "inner"
        not_first = False

        elem['members'] = [member for member in elem['members']
                            if member['type'] == 'way']
        for pos in range(len(elem["members"])):
            mem = elem['members'][pos]

            # check whether the coordinates of the next member need to be reversed
            # also sometimes the next member may not actually connect to the previous member, so if necessary, find a matching member
            if points != []:
                dist_start = (points[-1][0] - mem["geometry"][0]["lon"]
                                )**2 + (points[-1][1] - mem["geometry"][0]["lat"])**2
                dist_end = (points[-1][0] - mem["geometry"][-1]["lon"]
                            )**2 + (points[-1][1] - mem["geometry"][-1]["lat"])**2
                if dist_start == 0:
                    pass  # don't need to do anything
                elif dist_end == 0:
                    # flip the next member - it is entered in the wrong direction
                    mem["geometry"] = list(reversed(mem["geometry"]))
                else:
                    # try flipping the previous member
                    dist_flipped_start = (points[0][0] - mem["geometry"][0]["lon"])**2 + (
                        points[0][1] - mem["geometry"][0]["lat"])**2
                    dist_flipped_end = (points[0][0] - mem["geometry"][-1]["lon"])**2 + (
                        points[0][1] - mem["geometry"][-1]["lat"])**2
                    if dist_flipped_start == 0:
                        # just flip the start
                        points = list(reversed(points))
                    elif dist_flipped_end == 0:
                        # both need to be flipped
                        points = list(reversed(points))
                        mem["geometry"] = list(reversed(mem["geometry"]))
                    else:
                        # no matches -- look for a new match
                        point_found = False
                        for i in range(pos + 1, len(elem['members'])):
                            if not point_found:
                                new_pt = elem['members'][i]
                                dist_start = (new_pt['geometry'][0]['lon'] - points[-1][0])**2 + (
                                    new_pt['geometry'][0]['lat'] - points[-1][1])**2
                                dist_end = (new_pt['geometry'][-1]['lon'] - points[-1][0])**2 + (
                                    new_pt['geometry'][-1]['lat'] - points[-1][1])**2

                                if dist_start == 0 or dist_end == 0:
                                    point_found = True
                                    # swap the order of the members -- we have found the one we want
                                    elem['members'][pos], elem['members'][i] = elem['members'][i], elem['members'][pos]
                                    # save this new point as mem
                                    mem = elem['members'][pos]

                                    if dist_end == 0:
                                        mem['geometry'] = list(
                                            reversed(mem['geometry']))

                        if not point_found:
                            # don't work with this park
                            continue

            # address outer values
            if mem['role'] == 'outer':
                if prev == "inner":
                    # start new outer polygon
                    points = []

                if points == [] and not_first:
                    # append the previous poly to the polygon list
                    polygons.append(poly)
                    poly = []

                for coords in mem["geometry"]:
                    points.append([coords["lon"], coords["lat"]])

                if points[-1] == points[0]:
                    # finish the outer polygon if it has met the start
                    poly.append(points)
                    points = []
                # update condition
                prev = "outer"

            # address inner points
            if mem['role'] == "inner":
                for coords in mem["geometry"]:
                    points.append([coords["lon"], coords["lat"]])

                # check if the inner is complete
                if points[-1] == points[0]:
                    poly.append(points)
                    points = []
                # update condition
                prev = "inner"

            not_first = True

        # add in the final poly
        polygons.append(poly)

        if polygons != [[]]:
            # create MultiPolygon feature - separate multipolygon for each outer
            geojson['geometry']['type'] = 'MultiPolygon'
            geojson['geometry']['coordinates'] = polygons
            return geojson
        else:
            return None

    else:
        raise NotImplementedError

def distance_coordinates_(coord1, coord2):
    return great_circle(coord1, coord2).meters


def distance_coords(coord1, coord2):
    return great_circle(coord1[::-1], coord2[::-1]).meters


def bbox_from_path(path):
    vertices = path._vertices
    ulLon, lrLon = min(vertices[:, 0]), max(vertices[:, 0])
    lrLat, ulLat = min(vertices[:, 1]), max(vertices[:, 1])
    return ulLon, lrLat, lrLon, ulLat


def find_coordinate(fix_coord, meters, direction):
    change = .1
    lon, lat = fix_coord
    change_coord = fix_coord
    while True:
        if direction == 'lon':
            change_coord = (lon + change, lat)
        if direction == 'lat':
            change_coord = (lon, lat + change)
        if abs(distance_coordinates(fix_coord, change_coord) - meters) < 0.0000001:
            return change_coord, change
        distance = distance_coordinates(fix_coord, change_coord)
        ratio = distance / meters
        change = change / ratio


def filter_coordinates(unfiltered, region, min_lat, max_lat, radius):
    check_points = 20
    filtered = []
    if not isinstance(region, list):
        region = [region]
    close_coordinate, radius_degree = find_coordinate((min([min_lat, max_lat], key=abs), 0), 5000, 'lat')
    ccc = []
    for circle in unfiltered:
        circle_points = ([(math.cos(2*math.pi/check_points*x)*radius_degree + circle[1], math.sin(2*math.pi/check_points*x)*radius_degree + circle[0]) for x in range(0, check_points + 1)])
        if any(any(path.contains_point((circle)) for path in region) for circle in circle_points):
            filtered.append(circle)
            ccc.extend([(math.sin(2*math.pi/200*x)*radius_degree + circle[0], math.cos(2*math.pi/200*x)*radius_degree + circle[1]) for x in range(0, 200 + 1)])
    # plot_it(ccc)
    return filtered


def get_coordinates(region, radius):
    bbox = bbox_from_path(region)
    min_lon, min_lat, max_lon, max_lat = [float(coord) for coord in ['%.3f' % coord for coord in bbox]]

    x_dist = max(distance_coordinates((min_lat, min_lon), (min_lat, max_lon)),  distance_coordinates((max_lat, min_lon), (max_lat, max_lon)))
    y_dist = distance_coordinates((min_lat, max_lon), (max_lat, max_lon))

    x_gap = (math.sqrt(3) * radius) / 1.05
    y_gap = (1.5 * radius) / 1.05

    x_circles = math.ceil(x_dist / x_gap) + 1
    y_circles = math.ceil(y_dist / y_gap) + 1

    unfiltered = []
    lat_base = (min_lat, min_lon)
    for x_count in range(x_circles):
        unfiltered.append(lat_base)
        prev_lon = lat_base
        lat_base, xchange = find_coordinate(lat_base, x_gap, 'lat')
        for y_count in range(y_circles):
            prev_lon, ychange = find_coordinate(prev_lon, y_gap, 'lon')
            if y_count % 2 == 1:
                unfiltered.append(prev_lon)
            else:
                lat, lon = prev_lon
                unfiltered.append((lat, lon + .5 * xchange))
    return filter_coordinates(unfiltered=unfiltered, region=region, min_lat=min_lat, max_lat=max_lat, radius=radius)
