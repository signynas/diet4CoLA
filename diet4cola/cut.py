import numpy as np

class CoLACut():
    def __init__(self, cell_center: tuple[int, int], offset_bound: int, max_length: int, seed: int = 42):
        np.random.seed(seed)
        offset_bound_half = offset_bound // 2
        self.cut_center     = (cell_center[0] + np.random.randint(offset_bound) - offset_bound_half, cell_center[1] + np.random.randint(offset_bound) - offset_bound_half)
        self.cut_length     = np.random.randint(max_length)
        self.cut_alpha      = np.deg2rad((np.random.rand() * 180) - 90)

        cut_axis_a  = self.cut_length // 2 
        a_y         = cut_axis_a * np.sin(self.cut_alpha)
        a_x         = a_y / np.tan(self.cut_alpha)

        origin_offset_y         = -a_y if self.cut_alpha < 0 else a_y
        destination_offset_y    = a_y if self.cut_alpha < 0 else -a_y

        self.cut_origin         = (self.cut_center[0] - a_x, self.cut_center[1] + origin_offset_y)
        self.cut_destination    = (self.cut_center[0] + a_x, self.cut_center[1] + destination_offset_y)