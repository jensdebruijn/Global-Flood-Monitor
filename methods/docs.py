def create_url(location_ID):
    if location_ID.startswith('g-'):
        return f'http://www.geonames.org/{location_ID[2:]}'
    elif location_ID.startswith('osm-'):
        return f'http://www.openstreetmap.org/{location_ID[4:]}'
    elif location_ID.startswith('s-'):
        return 'http://www.hydrosheds.org/page/hydrobasins'
    else:
        raise NotImplementedError
