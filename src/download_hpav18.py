import os
import requests
import pandas as pd
from multiprocessing import Pool

colors = ['red','green','blue','yellow']
DIR = "/kaggle2/hpaic/v18"
v18_url = 'http://v18.proteinatlas.org/images/'

imgList = pd.read_csv("../data/HPAv18RBGY_wodpl.csv")


def download(name):
    i = name
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path

        fname = os.path.join(DIR, f"{img_name}")
        if os.path.exists(fname):
            continue

        r = requests.get(img_url, allow_redirects=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
    return None

if __name__ == '__main__':
    p = Pool(8)
    p.map(download, imgList.Id.values)
