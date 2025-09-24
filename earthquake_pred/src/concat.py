import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder




class concat:
    def __init__(self, file_path_main, file_path_add, dist_met='haversine'):
        self.df = pd.read_csv(file_path_main)
        self.df_add = pd.read_csv(file_path_add)
        self.dist_methods = {
            "euclidean": self._eucl_dist,
            "haversine": self._have_dist
        }
        self.dist_metric = dist_met
    
    def _rad(self, z):
        return math.pi * z / 180 

    def _have_dist(self, x,y, a,b):
        x, y, a, b = self._rad(x),self._rad(y), self._rad(a), self._rad(b)
        R = 6371
        v=math.sin((x-a)/2)**2 + math.cos(x) * math.cos(a) * math.sin((y-b)/2)**2
        return 2*R*math.atan2(math.sqrt(v), math.sqrt(1 - v))

    def _eucl_dist(self,x,y, a,b):
        return math.sqrt((x-a)**2 + (y-b)**2)
        
    def _dist_calc(self, x, y):
        if self.dist_metric not in self.dist_methods:
            raise ValueError("Inproper metric, choose euclidean or haversine")
        min_dist = float('inf')
        plate = ""
        for i, j, z in zip(self.df_add["lat"], self.df_add["lon"], self.df_add["plate"]):
            tmp = self.dist_methods[self.dist_metric](x, y, i, j)
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


