import os
import shutil
try:
    os.mkdir("sortedPics/forTesting")
except OSError as error:
    print(error)
for directory in os.listdir('sortedPics/forTraining'):
    newDirectoryName='sortedPics/forTesting/'+str(directory)
    oldDirectoryName='sortedPics/forTraining/'+str(directory)
    os.mkdir(newDirectoryName)
    counter=1
    for image in os.listdir(oldDirectoryName):
        if counter%10==1:
            oldFilePath=oldDirectoryName+'/'+str(image)
            newFilePath=newDirectoryName+'/'+str(image)
            shutil.move(oldFilePath,newFilePath)
            print("moved "+oldFilePath+' to '+newFilePath)
        counter+=1

