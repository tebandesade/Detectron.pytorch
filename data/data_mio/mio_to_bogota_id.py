


def bogota_to_mio(bogota_id):
    lookup = {
        0: 0,  # ... background
        1: 1,  #  articulated-truck - articulated-truck
        2: 2,  #  bicycle
        3: 3,  #  bus
        4: 4,  #  car
        5: 5,  #  motorcycle
        6: 4,  #  suv-car
        7: 4,  #  taxi-car
        8:  8,  #  person/pedestrian
        9: 9 ,  #  pickup-truck
        10: 10,  #  single unit truck -truck
        11: 11  #  work van - truck
    }
    return lookup[bogota_id]