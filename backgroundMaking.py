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
try:
   os.mkdir("sortedPics/forTraining/background")
except OSError as error:
   print(error)
coco_data = tfds.load('coco', split='train', shuffle_files=True)
for example in coco_data:
   image = example['image']
   imageID= example['image/id'].numpy()
   labels = example['objects']['label']
   bboxes = example['objects']['bbox']
   labelNums=labels.numpy()
   shape=image.shape
   pil_image=Image.fromarray(image.numpy())
   imageForCrop=Image.fromarray(image.numpy())
   plt.imshow(pil_image)
    #make a background class patch from this image
   imageProblematic=False
   whileCounter=0
   numPatchesFromThisImage=0
   while(numPatchesFromThisImage<7 and not whileCounter>9999):
      whileCounter+=1
      h, w =image.shape[0], image.shape[1]
      x_max = w - 80
      xLeft = random.randint(0, x_max)
      xRight=random.randint(xLeft+80, w)
      y_max = h - 80
      yLower = random.randint(0, y_max)
      yUpper=random.randint(yLower+80, h)
      for count,bbox in enumerate(bboxes):
         if (labelNums[count]>14 and labelNums[count] < 24) or labelNums[count]==77:
            cord=bbox.numpy()
            if((yLower>cord[0]*h and yLower<cord[2]*h) or (yUpper>cord[0]*h and yUpper<cord[2]*h) or (xLeft>cord[1]*w and xLeft<cord[3]*w) or (xRight>cord[1]*w and xRight<cord[3]*w)or(yLower<cord[0]*h and yUpper> cord[2]*h and xLeft<cord[1]*w and xRight>cord[3]*w)):
               imageProblematic=True
               break
      if(not imageProblematic):
         img=imageForCrop.crop((xLeft,yLower,xRight,yUpper))
         filename="sortedPics/forTraining/background/image"+str(imageID)+"crop"+str(numPatchesFromThisImage)+".jpg"
         img.save(filename)
         numPatchesFromThisImage+=1