import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import atan2,degrees,sin,cos,radians
import rrt
# https://github.com/PavanproJack/Mobile-Robotics-Repository

class Perceptions:
    imageMap = np.zeros(0)

    def initMap(self, imShape):
        self.imageMap = np.zeros(imShape.shape, dtype=np.uint8)

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
        y0 = initial_point[0]
        x0 = initial_point[1]
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
        # self.imageMap = np.zeros(planta.shape)
        y0 = initial_point[0]
        x0 = initial_point[1]
        for (x,y) in cartDots:
            self.imageMap = cv2.circle(self.imageMap, (int(x + x0),int(y + y0)), 5, 255, -1)
        # self.imageMap = cv2.circle(self.imageMap, (int(x + x0),int(y + y0)), 5, 255, -1)
        # temp = self.imageMap.copy()
        # temp = cv2.arrowedLine(temp, (x0,y0), (x0+14,y), color, thickness)
        # plt.imshow(temp,'gray'),plt.show()
        # plt.imshow(self.imageMap,'gray'),plt.show()

class Robot:
    w = 0
    x_robot = 0
    y_robot = 0
    theta = 0
    robotMap = Perceptions()    

    def UpdateMap(self, planta):
        self.robotMap.UpdateMap(planta, (int(self.y_robot),int(self.x_robot)))

    def setValues(self, planta, w=0.5, x=0, y=0, theta=0):
        self.w = w
        self.x_robot = x
        self.y_robot = y
        self.theta = theta
        self.robotMap.initMap(planta)

    def UpdateRobotPosition(self, vr, vl, t):
        print("vr = ", vr, " vl = ", vl, " t = ",t)
        print(self.x_robot, self.y_robot, self.theta)
        #Straight Line Trajectory
        if(vr == vl):
            # print("eq")
            self.x_robot = self.x_robot + vr*t*cos(radians(self.theta))
            self.y_robot = self.y_robot + vr*t*sin(radians(self.theta))
             # theta_n = theta
        # Spin in Place
        elif ((vr + vl) == 0):
            # print("opos")
            # print(vr , vl)
            if(vr > 0):
                self.theta += (2*vr*t)/self.w
            else:
                self.theta += (2*vl*t)/self.w

        #Circular Trajectory
        else:
            # print("diff")
            # Instantaneous Center of Curvature (ICC)
            # Let R be the radius from ICC
            # and l be the distance between the wheels
            
            R = (self.w/2) * ((vr + vl) / (vr - vl))
            
            ICC_x, ICC_y = self.x_robot - R*np.sin(self.theta), self.y_robot + R*np.cos(self.theta)
            
            #Change in Theta is proportional to the difference between vr and vl
            omega = (vr - vl) / self.w
            delta_Theta = omega * t
            
            self.x_robot = (self.x_robot - ICC_x) * np.cos(delta_Theta) - (self.y_robot - ICC_y) * np.sin(delta_Theta) + ICC_x
            self.y_robot = (self.x_robot - ICC_x) * np.sin(delta_Theta) + (self.y_robot - ICC_y) * np.cos(delta_Theta) + ICC_y
            self.theta += delta_Theta
            
            # self.x_robot = -((vr+vl)*np.sin(self.theta))*t/2
            # self.y_robot = ((vr+vl)*np.cos(self.theta))*t/2
            # self.theta = (vr-vl)/self.w
        print("/t",self.x_robot, self.y_robot, self.theta)

    def PlotMap(self):
        temp = self.robotMap.imageMap.copy()
        x,y = self.robotMap.Polar2Cartesian((20,-(self.theta)))
        x,y = int(x), int(y)
        x0,y0 = int(self.x_robot), int(self.y_robot)
        temp = cv2.arrowedLine(temp, (x0,y0), (x0+x, y0+y), 255, 3, line_type = cv2.LINE_AA, tipLength=1)
        plt.imshow(temp,'gray'),plt.show()



img = cv2.imread('planta_baixa.jpg',0)
# plt.imshow(img, 'gray');plt.show()
_,planta  = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
        
kernel = np.ones((5,5),np.uint8)
planta = cv2.morphologyEx(planta, cv2.MORPH_OPEN, kernel)
plt.imshow(planta, 'gray'), plt.show()


robot = Robot()
x = int(input("x = "))
y = int(input("y = "))
robot.setValues(planta, x=x,y=y)
robot.UpdateMap(planta)
robot.PlotMap()
while (True):
    a, b, c = input("vr, vl, t ").split()  
    vr = int(a)
    vl = int(b)
    t = int(c)
    robot.UpdateRobotPosition(vr, vl,t)
    robot.UpdateMap(planta)
    robot.PlotMap()
