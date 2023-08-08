from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    def __init__(self):
        bm = VGG16(weights="imagenet")
        self.model = Model(inputs=bm.input,outputs=bm.get_layer("fc1").output)

        pass
    def extract(self,img):
        img = img.resize((224,224)).convert("RGB")
        x = image.img_to_array(img)        
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        f = self.model.predict(x)[0]
        return f/np.linalg.norm(f)