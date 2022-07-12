import tensorflow_datasets as tfds
import os as os
from PIL import Image,ImageDraw
import random
import matplotlib.pyplot as plt
labelKey=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'truck', 'train', 'boat', 'traffic-light', 'fire-hydrant', 'stop-sign',
'parking-meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
'frisbee','skis','snowboard','sports-ball','kite','baseball-bat','baseball-glove','skateboard','surfboard','tennis-racket','bottle','wineglass','cup',
'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot-dog','pizza','donut','cake','chair','couch','potted-plant',
'bed','dining-table','toilet','tv','laptop','mouse','remote','keyboard','cellphone','microwave','oven','toaster','sink','refridgerator','book','clock',
'vase','scissors','teddy-bear','hair-drier','toothbrush']
os.mkdir("sortedPics")
os.mkdir("sortedPics/forTraining")
for i in range(15,24):
   try:
      os.mkdir("sortedPics/forTraining/"+labelKey[i])
   except OSError as error:
      print(error)
try:
   os.mkdir("sortedPics/forTraining/teddy-bear")
except OSError as error:
   print(error)
coco_data = tfds.load('coco', split='train', shuffle_files=True)
for example in coco_data:
    image = example['image']
    imageID= example['image/id'].numpy()
    labels = example['objects']['label']
    bboxes = example['objects']['bbox']
    labelNums=labels.numpy()
    #print(imageID)
    #print(labelNums)
    shape=image.shape
    pil_image=Image.fromarray(image.numpy())
    imageForCrop=Image.fromarray(image.numpy())
    plt.imshow(pil_image)
    for count,bbox in enumerate(bboxes):
        if (labelNums[count]>14 and labelNums[count] < 24) or labelNums[count]==77:
            cord=bbox.numpy()
            
            i = ((cord[3]-cord[1])+(cord[2]-cord[0])/2)*0.025
            ymin = cord[0]
            xmin = cord[1]
            ymax = cord[2]
            xmax = cord[3]

            if cord[0]-i > 0:
                ymin = cord[0]-i
            if cord[1]-i > 0:
                xmin = cord[1]-i
            if cord[2]+i < 1:
                ymax = cord[2]+i
            if cord[3]+i < 1:
                xmax = cord[3]+i

            img=imageForCrop.crop((xmin*shape[1],ymin*shape[0],xmax*shape[1],ymax*shape[0]))
            # bounding box order (cord[]): ymin, xmin, ymax, xmax
            # crop order: xmin, ymin, xmax, ymax
            # shape[0] = height, shape[1] = width

            width,height=img.size
            if width>80 and height>80:
                filename="sortedPics/forTraining/"+labelKey[labelNums[count]]+"/image"+str(imageID)+"crop"+str(count)+".jpg"
                img.save(filename)