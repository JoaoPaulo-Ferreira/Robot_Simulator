import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import atan2,degrees,sin,cos,radians
import rrt_star as rrt
import os
# https://github.com/PavanproJack/Mobile-Robotics-Repository
clear = lambda: os.system('clear')


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
            contours, _ = cv2.findContours(showimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
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
        y0 = initial_point[0]
        x0 = initial_point[1]
        for (x,y) in cartDots:
            self.imageMap = cv2.circle(self.imageMap, (int(x + x0),int(y + y0)), 2, 255, -1)
        
class Robot:
    w = 0
    x_robot = 0
    y_robot = 0
    theta = 0
    robotMap = Perceptions()    

    def UpdateMap(self, planta):
        self.robotMap.UpdateMap(planta, (int(self.y_robot),int(self.x_robot)))
        cv2.imwrite("robotmap.png", self.robotMap.imageMap)

    def setValues(self, planta, w=0.5, x=0, y=0, theta=0):
        self.w = w
        self.x_robot = x
        self.y_robot = y
        self.theta = theta
        self.robotMap.initMap(planta)

    def checkValidPosition(self):
        if(self.x_robot > self.robotMap.imageMap.shape[1] or self.x_robot < 0):
            print("Invalid x_robot : ", self.x_robot)
            quit()
        if(self.y_robot > self.robotMap.imageMap.shape[0] or self.y_robot < 0):
            print("Invalid y_robot : ", self.y_robot)
            quit()

    def UpdateRobotPosition(self, vr, vl, t):
        #Straight Line Trajectory
        if(vr == vl):
            # print("eq")
            self.x_robot = self.x_robot + round(vr*t*cos(radians(self.theta)))
            self.y_robot = self.y_robot + round(vr*t*sin(radians(self.theta)))
        # Spin in Place
        elif ((vr + vl) == 0):
            if(vr > 0):
                self.theta = (self.theta + (2*vr*t)/self.w)%360
            else:
                self.theta = (self.theta + (2*vl*t)/self.w)%360
        self.checkValidPosition()

    def PlotMap(self, path=None):
        temp = self.robotMap.imageMap.copy()
        x,y = self.robotMap.Polar2Cartesian((20,-(self.theta)))
        x,y = int(x), int(y)
        x0,y0 = int(self.x_robot), int(self.y_robot)
        temp = cv2.arrowedLine(temp, (x0,y0), (x0+x, y0+y), 255, 3, line_type = cv2.LINE_AA, tipLength=1)
        if(path != None):
            for i in range(len(path)-1):
                origem = (int(path[i][0]),int(path[i][1]))
                dest = (int(path[i+1][0]),int(path[i+1][1]))
                temp = cv2.line(temp, origem, dest, 255, lineType=cv2.LINE_AA)
        plt.imshow(temp,'gray'),plt.show()

    def is_collision(self, start, end):
        test = np.zeros((self.robotMap.imageMap.shape), dtype=np.uint8)
        test = cv2.line(test, start, end, 255, lineType=cv2.LINE_AA)
        result = cv2.bitwise_and(self.robotMap.imageMap, self.robotMap.imageMap, mask=test)
        if 255 in result:
            return True
        return False


class Controller:
    img = np.zeros(0)
    path = None
    path_update_cnt = 0
    traveled_path = []
    time_total = 0
    time_step = 0
    std_speed = 10
    robot = Robot()
    start_x = 300
    start_y = 500
    goal = (200,100)
    rrt = 0
    def __init__(self):
        self.img = cv2.imread('planta_baixa.jpg',0)
        _,planta  = cv2.threshold(self.img,80,255,cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((5,5),np.uint8)
        self.img = cv2.morphologyEx(planta, cv2.MORPH_OPEN, kernel)
        self.robot.setValues(self.img, x=self.start_x, y=self.start_y)

    def checkPosition(self, next_step):
        delta = 1
        delta_x = np.abs(next_step[0] - self.robot.x_robot)
        delta_y = np.abs(next_step[1] - self.robot.y_robot)
        if(delta_x <= 1):
            if(delta_y <= 1):
                return True
        return False

    def UpdateRobotPosition(self, start, next_step):
        while(not self.checkPosition(next_step)):
            polar = self.robot.robotMap.Cartesian2Polar(next_step, start)
            dist = int(polar[0])
            angle = int(polar[1])
            # print("start, next, rho, theta : ",start, next_step, dist, angle)
            vr, vl, t = self.gerPar(dist, angle)
            self.robot.UpdateRobotPosition(vr, vl, t)
            self.time_step += t
            print("state: ",self.robot.x_robot, self.robot.y_robot, self.robot.theta)

    def gerPar(self, dist, angle):
        if(angle == self.robot.theta):
            vr = vl = self.std_speed
            t = dist/self.std_speed
            print("str line", vr, vl, t)
        else:
            vr = self.std_speed
            vl = -self.std_speed
            t = np.abs((np.abs(angle - self.robot.theta) * self.robot.w)/(2*vr))
            print("spin", vr, vl, t)
        return vr, vl, t

    def summary(self):
        clear()
        print("         ###############		SUMMARY		###############")
        print("PATH ALG		    :	", "RRT_SMART")
        print("START POINT 	    :	", (self.start_x, self.start_y))
        print("GOAL POINT 	    :	", self.goal)
        print("TRAVEL TIME      :	", self.time_total)
        print("PATH UPDATES     :	", self.path_update_cnt)
        print("STEPS            :	", len(self.traveled_path))

    def Loop(self):
        while(True):
            self.robot.UpdateMap(self.img)
            self.robot.PlotMap(self.path)
            if self.path == None:
                # self.rrt = rrt.Rrt((self.robot.x_robot, self.robot.y_robot), self.goal, rrt.STEP_LEN, rrt.SAMPLE_RATE, rrt.ITER_MAX)
                self.rrt = rrt.RrtStar((self.robot.x_robot, self.robot.y_robot), self.goal, rrt.STEP_LEN, rrt.SAMPLE_RATE, 20, rrt.ITER_MAX)
                self.path = self.rrt.planning()
                self.path_update_cnt += 1
                self.path.reverse()

            else:
                if(len(self.path) < 2):
                    self.path = None
                else:
                    self.path.pop(0)
                    start = (int(self.robot.x_robot), int(self.robot.y_robot))
                    next_step = (int(self.path[0][0]), int(self.path[0][1]))
                    if (not self.robot.is_collision(start,next_step)):
                        self.UpdateRobotPosition(start, next_step)
                        if(self.checkPosition(next_step)):
                            self.time_total += self.time_step
                            self.traveled_path.append(next_step)
                            # self.path.pop(0)
                            if(next_step == self.goal):
                                
                                self.summary()
                                print("sucess")
                                break                    
                    else:
                        print("Colision")
                        self.path = None

c = Controller()
c.Loop()
 