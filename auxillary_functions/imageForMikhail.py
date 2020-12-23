import cv2
from functions import set_title

image1_path = 'D:/SuperGlueDir/GluePretr/Mikhail1.jpg'
out_path = './Mikhail2.jpg'
frame1 = cv2.imread(image1_path)
# print('frame1 shape: ', frame1.shape)
title = 'h_pred = ' + str(round(1232.80, 1)) + ' h label = ' + str(round(883.94, 1))
out_img = set_title(frame1, title)
cv2.imwrite(out_path, out_img)
