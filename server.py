import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime 
from flask import Flask, request, render_template
from pathlib import Path
#from descriptions_monuments import d
from dictionary import dic
#import cv2 

app = Flask(__name__)

#read image extractor
fe = FeatureExtractor()
feature = []
img_path = []
for fp in Path("./static/feature").glob("*.npy"):
    feature.append(np.load(fp))
    img_path.append(Path("./static/img/Image dataset")/(fp.stem+".jfif"))
    
feature = np.array(feature)

@app.route("/",methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        file = request.files["query_img"]
        img = Image.open(file.stream)
        up_img_p = "static/uploded/"+datetime.now().isoformat().replace(":",".")+"_"+file.filename
        img.save(up_img_p)
        #searched image
        query = fe.extract(img)
        dists  = np.linalg.norm(feature - query, axis = 1)
        #print(dists)
        ids = np.argsort(dists)[:5]
        #print(ids)
        efficiency = round((1 - (np.min(dists) / np.max(dists))) * 100, 2)
        print(f"Efficiency: {efficiency}%")
        scores = [(dists[id],img_path[id],dic[id]) for id in ids]
        scores.append(('Efficiency', f'{efficiency}%'))
        print(scores)

        #return render_template("CBIR webpage.html", query_path=up_img_p, scores=scores, efficiency=efficiency)
        return render_template("index.html", query_path=up_img_p, scores=scores)
    else:
        return render_template("index.html")

if __name__=="__main__":
    app.run()
