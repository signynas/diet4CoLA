import tqdm as tqdm
import czifile
import tifffile
import os

folder = 'data/czi'

for file in tqdm.tqdm(os.listdir(folder)):
    if file.endswith('.czi'):
        czi = czifile.CziFile(os.path.join(folder, file))
        img = czi.asarray()
        tifffile.imwrite(os.path.join('data/tiff', file.replace('.czi', '.tiff')), img)