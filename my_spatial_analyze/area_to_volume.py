import math
import pandas as pd
import numpy as np


def area_to_avg_depth_hydrolakes(area, s100):
    def depth_01_1(area, s100):
        if area == 0:
            return 0
        else:
            log10_d = 0.3826 + 0.1512*math.log10(area) + 0.4820*math.log10(s100) + 0.0678
            return 10**log10_d

    def depth_1_10(area, s100):
        if area == 0:
            return 0
        else:
            log10_d = 0.1801 + 0.2985*math.log10(area) + 0.8473*math.log10(s100)+ 0.0689
            return 10**(log10_d)

    def depth_10_100(area, s100): 
        if area == 0:
            return 0
        else:
            log10_d = 0.0379 + 0.2445*math.log10(area) + 1.1517*math.log10(s100) + 0.0692
            return 10**(log10_d)

    def depth_100_500(area, s100):
        if area == 0:
            return 0
        else:
            log10_d = 0.0123 + 0.2664*math.log10(area) + 1.1474*math.log10(s100) + 0.1094
            return 10**(log10_d)
    if area < 1:
        dep_func = depth_01_1
    elif area < 10:
        dep_func = depth_1_10
    elif area < 100:
        dep_func = depth_10_100
    elif area < 500:
        dep_func = depth_100_500
    else:
        dep_func = depth_100_500
    return dep_func(area, s100)
    
def area_to_volume_hydrolakes(area, s100):
    return area_to_avg_depth_hydrolakes(area, s100)*area
    
def areas_to_avg_depths_hydrolakes(areas, s100):
    return np.array([area_to_avg_depth_hydrolakes(a, s100) for a in areas])

def depths_areas_to_volumes_hydrolakes(areas, depths):
    return np.array([a*d for a, d in zip(areas, depths)])