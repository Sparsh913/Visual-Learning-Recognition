import cv2
import numpy as np
import os

data_path = '/home/uas-laptop/Kantor_Lab/nerfstudio/colmap_test/data/ur_pusher'

# Resize the images in the data_path by half and save them in a new folder
new_data_path = '/media/uas-laptop/KANTOR-LAB/colmap_data/ur_pusher'

for img in os.listdir(data_path):
    # read the image
    image = cv2.imread(os.path.join(data_path, img))
    # get the height and width of the image
    h, w = image.shape[:2]
    # resize the image by half
    resized_image = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_AREA)
    # save the resized image in the new_data_path
    cv2.imwrite(os.path.join(new_data_path, img), resized_image)
    
print('Images resized successfully')