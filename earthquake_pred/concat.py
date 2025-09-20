import pandas as pd
import math




class concat:
    def __init__(self, file_path_main, file_path_add):
        self.df = pd.read_csv(file_path_main)
        self.df_add = pd.read_csv(file_path_add)

    def _eucl_dist(self,x,y, a,b):
        return math.sqrt((x-a)**2 + (y-b)**2)
        
    def _dist_calc(self, x, y):
        min = self._eucl_dist(x, y, 100000, 1000000)
        for i, j in zip(self.df_add["lat"], self.df_add["lon"]):
            tmp = self._eucl_dist(x, y, i, j)
            if min > tmp:
                min = tmp
        return min
    
    def _calc_dist(self):
        t = []
        for i, j in zip(self.df["Latitude"], self.df["Longitude"]):
            t.append(self._dist_calc(i, j))
        self.df["Plate_Dist"] = t

    def get_done(self):
        self._calc_dist()
        print(self.df.head())
        return self.df


new = concat("dataset.csv", "all.csv")
new.get_done()