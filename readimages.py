import numpy as np
import os
from tqdm import tqdm
from PIL import Image

# The dataset used is from mnist
def readImages(imgfile, labelfile, DIR) :
    IMAGEMAGICNUMBER = 2051
    LABELMAGICNUMBER = 2049
    #DIR = "dataset"

    labels = []

    with open(labelfile, "rb") as f:
        # Magic number
        b = f.read(4)
        magicnum = int.from_bytes(b, byteorder = "big")
        if(magicnum != LABELMAGICNUMBER):
            print("ERROR")

        # Label quantity
        b = f.read(4)
        qnt = int.from_bytes(b, byteorder = "big")

        for i in range(0, qnt):
            b = f.read(1)
            b = int.from_bytes(b, byteorder = "big")
            labels.append(b)


    print(labels[0])
    with open(imgfile, "rb") as f:

        # Magic number
        b = f.read(4)
        magicnum = int.from_bytes(b, byteorder = "big")
        if(magicnum != IMAGEMAGICNUMBER):
            print("ERROR")

        # Image quantity
        b = f.read(4)
        qnt = int.from_bytes(b, byteorder = "big")

        # Image width
        b = f.read(4)
        width = int.from_bytes(b, byteorder = "big")

        # Image height
        b = f.read(4)
        height = int.from_bytes(b, byteorder = "big")

        for i in tqdm(range(0, qnt)):
            img = np.zeros((height, width), dtype=np.uint8)
            for row in range(0, width):
                for col in range(0, height):
                    b = f.read(1)
                    b = int.from_bytes(b, byteorder = "big")
                    img[row][col] = b

            img = Image.fromarray(img, 'L')
            imgname = os.path.join(DIR, "img_" + str(labels[i]) + "_" + str(i) + ".png")
            img.save(imgname)

if __name__ == "__main__":
    imgfile = input("image file name: ")
    labelfile = input("label file name: ")
    directory = input("directory name: ")
    readImages(imgfile, labelfile, directory)
