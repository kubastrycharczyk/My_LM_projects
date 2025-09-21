import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder




class concat:
    def __init__(self, file_path_main, file_path_add):
        self.df = pd.read_csv(file_path_main)
        self.df_add = pd.read_csv(file_path_add)

    def _eucl_dist(self,x,y, a,b):
        return math.sqrt((x-a)**2 + (y-b)**2)
        
    def _dist_calc(self, x, y):
        min_dist = float('inf')
        plate = ""
        for i, j, z in zip(self.df_add["lat"], self.df_add["lon"], self.df_add["plate"]):
            tmp = self._eucl_dist(x, y, i, j)
            if min_dist > tmp:
                min_dist = tmp
                plate = z
        return min_dist, plate
    
    def _calc_dist(self):
        t = []
        p = []
        for i, j in zip(self.df["Latitude"], self.df["Longitude"]):
            dist, plate = self._dist_calc(i, j)
            t.append(dist)
            p.append(plate)
        
        labenc = LabelEncoder()
        p = labenc.fit_transform(p)

        self.df["Plate_Dist"] = t
        self.df["Plate"] = p

    def get_done(self):
        self._calc_dist()
        print(self.df.head())
        return self.df


