from cv2 import cv2
import uuid
import os
import time

# I am doing rock paper scissors
labels = ['rock', 'paper','scissors']
number_imgs = 6 # this is the number of images that will be collected for each class

# create images folder
IMAGES_PATH = os.path.join("images")

if not os.path.exists(IMAGES_PATH):
        os.mkdir(IMAGES_PATH)
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)

for label in labels:
    cap = cv2.VideoCapture(0) # this number might have to be changed depending on number of cameras in your workspace
    print('Collecting images for {}'.format(label))
    time.sleep(3)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        print('Got image {}'.format(imgnum))
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()
