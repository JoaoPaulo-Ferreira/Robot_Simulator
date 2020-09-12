import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import atan2,degrees,sin,cos,radians
import rrt


branca = cv2.imread("branca.png", 0)
preta = cv2.bitwise_not(branca)
plt.imshow(branca,"gray"), plt.show()

# cv2.imwrite("robotmap.png", self.robotMap.imageMap)
kernel = np.ones((5,5),np.uint8)
global_map = cv2.morphologyEx(preta, cv2.MORPH_OPEN, kernel)
local_map = cv2.imread("robotmap.png",0)
plt.subplot(1, 2, 1), plt.imshow(global_map, 'gray')
plt.subplot(1, 2, 2), plt.imshow(local_map, 'gray')

plt.show()


# plt.imshow(preta,"gray"), plt.show()
# cv2.imwrite("preta.png", global_map)
# self.robot.setValues(self.img, x=self.start_x, y=self.start_y)

