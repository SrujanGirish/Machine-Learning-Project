from PIL import Image
from pathlib import Path
import numpy as np

from feature_extractor import FeatureExtractor

if __name__ == "__main__":
    fe = FeatureExtractor()
    
    for i in sorted(Path("./static/img/Image dataset").glob("*.jfif")):
        #print(i)
        #extract feature
        fea = fe.extract(img=Image.open(i))
        #print(type(fea),fea.shape)
        fp = Path("./static/feature")/(i.stem+".npy")
        print(fp)
        # save feature
        np.save(fp,fea)