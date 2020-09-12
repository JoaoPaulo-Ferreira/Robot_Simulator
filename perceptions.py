import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import atan2,degrees,sin,cos,radians
# import operator

class Perceptions:
    imageMap = np.zeros(0)
    def getWhitePoints(self,contours):
        dots = np.zeros(0)
        for contour in contours:
            dots = np.append(dots,(contour))
        dots = np.reshape(dots,(-1,2))
        return dots

    def AngleBtw2Points(self,pointA, pointB):
        changeInY = pointB[1] - pointA[1]
        changeInX = pointB[0] - pointA[0]
        theta = degrees(atan2(changeInY,changeInX))
        theta = (theta + 360) % 360
        return float("{:.2f}".format(theta))

    def DistBtw2Points(self,pointA, pointB):
        temp = pointB - pointA
        rho = np.linalg.norm(temp)
        return float("{:.2f}".format(rho))

    def Cartesian2Polar(self, pointB, pointA):
        pointA = np.asarray(pointA)
        pointB = np.asarray(pointB)
        rho = self.DistBtw2Points(pointA, pointB)
        theta = self.AngleBtw2Points(pointA, pointB)
        # print(theta)
        return np.asarray((rho,theta))

    def Polar2Cartesian(self, Polar):
        rho = Polar[0]
        theta = Polar[1]
        x = float("{:.2f}".format(rho*cos(radians(theta))))
        y = float("{:.2f}".format(rho*sin(radians(theta))))
        return np.asarray((x,y))

    def Preprocess(self, polarDots):
        cartDots = np.zeros(0)
        polarTheta = np.around( polarDots[:,1] )
        # processedDots = np.vstack((B, A)).T
        processedDots = np.full(360, -1)
        for i in range(polarTheta.shape[0]):
            if(polarTheta[i] == 360):
                polarTheta[i] = 0
            if(processedDots[int(polarTheta[i])] == -1):
                processedDots[int(polarTheta[i])] = polarDots[i,0]
            elif (polarDots[i,0] < processedDots[int(polarTheta[i])]):
                processedDots[int(polarTheta[i])] = polarDots[i,0]
        
        for i in range(360):
            if(processedDots[i] != -1):
                cart = self.Polar2Cartesian((processedDots[i],i))
                cartDots = np.append(cartDots, cart)
        cartDots = np.reshape(cartDots, (-1,2))
        return cartDots

    # def PlotMap(self,cartPoints, imsize, initial_point):
    #     y0 = initial_point[0]
    #     x0 = initial_point[1]
    #     new_map = np.zeros(imsize)
    #     for (x,y) in cartPoints:
    #         # new_map[int(y + y0) ,int(x + x0)] = 255
    #         new_map = cv2.circle(new_map, (int(x + x0),int(y + y0)), 5, 255, -1)
    #     plt.imshow(new_map, 'gray'), plt.show()

    def Extract_imagePoins(self,planta,initial_point):
        lin = initial_point[0]
        col = initial_point[1]
        cont = 10

        alldots=np.zeros(0)
        while cont < max(col, lin, (planta.shape[1]-col), (planta.shape[0]-lin)): # distância até a borda mais próxima
            
            white = np.zeros((planta.shape), dtype=np.uint8)
            white = cv2.circle(white, (col,lin), cont, 255, -1)
            white = cv2.circle(white, (col,lin), (cont-2), 0, -1)
            
            cont += 5
            showimg = cv2.bitwise_and(planta,planta, mask = white)
            contours, _ = cv2.findContours(showimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            alldots = np.append(alldots, self.getWhitePoints(contours))
        alldots = np.reshape(alldots, (-1,2))
        return alldots


    def Readings_Sim(self, planta, initial_point):
        alldots=np.zeros(0)
        polarDots=np.zeros(0)    
        initial_point = np.asarray(initial_point)
        alldots = self.Extract_imagePoins(planta, initial_point)

        for (x,y) in alldots:   # x = coluna y = linha
            polar = self.Cartesian2Polar((x,y),(x0,y0))
            polarDots = np.append(polarDots, polar)
        polarDots = np.reshape(polarDots, (-1,2))    

        return polarDots

    def UpdateMap(self,planta, initial_point):
        polarDots = np.zeros(0)
        cartDots = np.zeros(0)
        polarDots = self.Readings_Sim(planta,(initial_point))
        cartDots = self.Preprocess(polarDots)
        # PlotMap(cartDots,img.shape,(y0,x0))
        self.imageMap = np.zeros(planta.shape)
        y0 = initial_point[0]
        x0 = initial_point[1]
        for (x,y) in cartDots:
            # print(x,y)
            # self.imageMap[int(y + y0) ,int(x + x0)] = 255
            self.imageMap = cv2.circle(self.imageMap, (int(x + x0),int(y + y0)), 5, 255, -1)
        plt.imshow(self.imageMap,'gray'),plt.show()
# y0 = int(input("linha "))
# x0 = int(input("coluna "))
y0 = 500
x0 = 300

img = cv2.imread('planta_baixa.jpg',0)
plt.imshow(img, 'gray');plt.show()
_,planta  = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
        
kernel = np.ones((3,3),np.uint8)
planta = cv2.morphologyEx(planta, cv2.MORPH_OPEN, kernel)
sim = Perceptions()
plt.imshow(planta, 'gray'), plt.show()
sim.UpdateMap(planta, (y0,x0))

# polarDots = np.zeros(0)
# cartDots = np.zeros(0)
# polarDots = Readings_Sim(img,(y0,x0))
# cartDots = Preprocess(polarDots)
# PlotMap(cartDots,img.shape,(y0,x0))