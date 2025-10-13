import pandas as pd

class Points(pd.DataFrame):
    @property
    def _constructor(self):
        return Points

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        return cls(df)

    def points_by_cell(self, cell_id: str):
        return self[self['cell_id'] == cell_id]
    
    def points_by_time(self, time):
        return self[self['time'] == time]
    
    def get_unique(self, key):
        return self[key].unique()