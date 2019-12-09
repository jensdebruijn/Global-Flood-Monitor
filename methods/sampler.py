#!/usr/bin/python3
import os
import numpy as np
from osgeo import gdal, osr

# By default, the GDAL and OGR Python bindings do not raise exceptions when errors occur.
# Instead they return an error value such as None and write an error message to sys.stdout.
# In Python, it is traditional to report errors by raising exceptions.
# You can enable this behavior in GDAL and OGR by calling the UseExceptions() function
gdal.UseExceptions()


def get_map_info(map):
    raster = gdal.Open(map)
    rb, gt, cols, rows = get_raster_info(raster)
    map_info = {
            'file': map,
            'name': os.path.basename(map),
            'extent': get_raster_bbox(raster),
            'NODATA': raster.GetRasterBand(1).GetNoDataValue(),
            'rb': rb,
            'gt': gt,
            'cols': cols,
            'rows': rows
        }
    return map_info


def map_dict(maps):
    maps = (get_map_info(map) for map in maps)
    return [map for map in maps if map is not None]


def MapToPixel(mx, my, gt):
    ''' Convert map to pixel coordinates
        @param  mx:    Input map x coordinate (double)
        @param  my:    Input map y coordinate (double)
        @param  gt:    Input geotransform (six doubles)
        @return: px,py Output coordinates (two ints)

        @change: changed int(p[x,y]+0.5) to int(p[x,y]) as per http://lists.osgeo.org/pipermail/gdal-dev/2010-June/024956.html
        @change: return floats
        @note:   0,0 is UL corner of UL pixel, 0.5,0.5 is centre of UL pixel
    '''
    if gt[2] + gt[4] == 0:  # Simple calc, no inversion required
        px = (mx - gt[0]) / gt[1]
        py = (my - gt[3]) / gt[5]
    else:
        px, py = ApplyGeoTransform(mx, my, InvGeoTransform(gt))
    return px, py


def PixelToMap(px, py, gt, center=False):
    ''' Convert pixel to map coordinates
        @param  px:    Input pixel x coordinate (double)
        @param  py:    Input pixel y coordinate (double)
        @param  gt:    Input geotransform (six doubles)
        @return: mx,my Output coordinates (two doubles)

        @note:   0,0 is UL corner of UL pixel, 0.5,0.5 is centre of UL pixel
    '''
    if center:
        px = px + .5
        py = py + .5
    mx, my = ApplyGeoTransform(px, py, gt)
    return mx, my


def InvGeoTransform(gt_in):
    # Compute determinate
    det = gt_in[1] * gt_in[5] - gt_in[2] * gt_in[4]

    if(abs(det) < 0.000000000000001):
        return

    inv_det = 1.0 / det

    # compute adjoint, and divide by determinate
    gt_out = [0, 0, 0, 0, 0, 0]
    gt_out[1] = gt_in[5] * inv_det
    gt_out[4] = -gt_in[4] * inv_det

    gt_out[2] = -gt_in[2] * inv_det
    gt_out[5] = gt_in[1] * inv_det

    gt_out[0] = (gt_in[2] * gt_in[3] - gt_in[0] * gt_in[5]) * inv_det
    gt_out[3] = (-gt_in[1] * gt_in[3] + gt_in[0] * gt_in[4]) * inv_det

    return gt_out


def ApplyGeoTransform(inx, iny, gt):
    ''' Apply a geotransform
        @param  inx:       Input x coordinate (double)
        @param  iny:       Input y coordinate (double)
        @param  gt:        Input geotransform (six doubles)

        @return: outx,outy Output coordinates (two doubles)
    '''
    outx = gt[0] + inx * gt[1] + iny * gt[2]
    outy = gt[3] + inx * gt[4] + iny * gt[5]
    return (outx, outy)


def GetExtent(raster_file):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ds = gdal.Open(raster_file)
    if ds is None:
        print("invalid file name or file ")
    else:
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        ext = []
        xarr = [0, cols]
        yarr = [0, rows]

        for px in xarr:
            for py in yarr:
                x = gt[0] + (px * gt[1]) + (py * gt[2])
                y = gt[3] + (px * gt[4]) + (py * gt[5])
                ext.append([x, y])
            yarr.reverse()

        return ext


def get_raster_bbox(src_ds):
    if isinstance(src_ds, str):
        src_ds = gdal.Open(src_ds)
    gt = src_ds.GetGeoTransform()
    ulx, xres, xskew, uly, yskew, yres  = src_ds.GetGeoTransform()
    lrx = ulx + (src_ds.RasterXSize * xres)
    lry = uly + (src_ds.RasterYSize * yres)
    return ulx, uly, lrx, lry


def GetExtent(gt, cols, rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0]+(px*gt[1])+(py*gt[2])
            y = gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext


def reproject_coord(coord, src_srs, tgt_srs):
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    x, y, z = transform.TransformPoint(coord[0], coord[1])
    return x, y

def ReprojectCoords(coords, src_srs, tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def get_raster_info(src_ds):
    rb = src_ds.GetRasterBand(1)
    gt = src_ds.GetGeoTransform()
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    return rb, gt, cols, rows


def sample_point_xy(px, py, rb, cols, rows, win_size=1, func=None):
    if (px >= 0) & (px <= cols) & (py >= 0) & (py <= rows):  # check if within map extent
        if win_size > 1:
            intval = rb.ReadAsArray(
                int(px - win_size / 2.), int(py - win_size / 2.), win_size, win_size)
        else:
            intval = rb.ReadAsArray(
                int(px), int(py), win_size, win_size)
        if func is not None:
            value = func(intval)
        else:
            value = intval
    else:
        value = np.nan
    return value


def sample_point_coord(lon, lat, rb, gt, cols, rows, win_size=1, func=None):
    px, py = MapToPixel(lon, lat, gt)
    return sample_point_xy(px, py, rb, cols, rows, win_size, func)
