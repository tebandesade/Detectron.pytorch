


def bogota_to_coco(bogota_id):
    lookup = {
        0: 0,  # ... background
        1: 8,  #  articulated-truck - truck
        2: 2,  #  bicycle
        3: 6,  #  bus
        4: 3,  #  car
        5: 4,  #  motorcycle
        6: 3,  #  suv-car
        7: 3,  #  taxi-car
        8:  1,  #  person/pedestrian
        9: 8 ,  #  pickup-truck -truck
        10: 8,  #  single unit truck -truck
        11: 8,  #  work van - truck
    }
    return lookup[bogota_id]