"""
RRT_2D
@author: huiming zhou
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import cv2

OBSTACLE = 255
FREE_SPACE = 0

STEP_LEN = 50
SAMPLE_RATE = .01
ITER_MAX = 1000
# iter_max

class Env:
    def __init__(self):
        self.img = cv2.imread('robotmap.png', 0)
        self.img = cv2.bitwise_not(self.img)
        self.x_range = (0,self.img.shape[1])
        self.y_range = (0,self.img.shape[0])

# class Plotting:
#     def __init__(self, x_start, x_goal):
#         self.xI, self.xG = x_start, x_goal
#         self.env = Env()
#         self.obs_bound = self.env.obs_boundary
#         self.obs_circle = self.env.obs_circle
#         self.obs_rectangle = self.env.obs_rectangle

#     def animation(self, nodelist, path, name, animation=False):
#         self.plot_grid(name)
#         self.plot_visited(nodelist, animation)
#         self.plot_path(path)

#     def animation_connect(self, V1, V2, path, name):
#         self.plot_grid(name)
#         self.plot_visited_connect(V1, V2)
#         self.plot_path(path)

#     def plot_grid(self, name):
#         fig, ax = plt.subplots()

#         for (ox, oy, w, h) in self.obs_bound:
#             ax.add_patch(
#                 patches.Rectangle(
#                     (ox, oy), w, h,
#                     edgecolor='black',
#                     facecolor='black',
#                     fill=True
#                 )
#             )

#         for (ox, oy, w, h) in self.obs_rectangle:
#             ax.add_patch(
#                 patches.Rectangle(
#                     (ox, oy), w, h,
#                     edgecolor='black',
#                     facecolor='gray',
#                     fill=True
#                 )
#             )

#         for (ox, oy, r) in self.obs_circle:
#             ax.add_patch(
#                 patches.Circle(
#                     (ox, oy), r,
#                     edgecolor='black',
#                     facecolor='gray',
#                     fill=True
#                 )
#             )

#         plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
#         plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

#         plt.title(name)
#         plt.axis("equal")

#     @staticmethod
#     def plot_visited(nodelist, animation):
#         if animation:
#             count = 0
#             for node in nodelist:
#                 count += 1
#                 if node.parent:
#                     plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
#                     plt.gcf().canvas.mpl_connect('key_release_event',
#                                                  lambda event:
#                                                  [exit(0) if event.key == 'escape' else None])
#                     if count % 10 == 0:
#                         plt.pause(0.001)
#         else:
#             for node in nodelist:
#                 if node.parent:
#                     plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

#     @staticmethod
#     def plot_visited_connect(V1, V2):
#         len1, len2 = len(V1), len(V2)

#         for k in range(max(len1, len2)):
#             if k < len1:
#                 if V1[k].parent:
#                     plt.plot([V1[k].x, V1[k].parent.x], [V1[k].y, V1[k].parent.y], "-g")
#             if k < len2:
#                 if V2[k].parent:
#                     plt.plot([V2[k].x, V2[k].parent.x], [V2[k].y, V2[k].parent.y], "-g")

#             plt.gcf().canvas.mpl_connect('key_release_event',
#                                          lambda event: [exit(0) if event.key == 'escape' else None])

#             if k % 2 == 0:
#                 plt.pause(0.001)

#         plt.pause(0.01)

#     @staticmethod
#     def plot_path(path):
#         if len(path) != 0:
#             plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
#             plt.pause(0.01)
#         plt.show()

class Utils:
    def __init__(self):
        self.env = Env()

        self.delta = 0.5

    def is_collision(self, start, end):
        test = np.zeros((self.env.img.shape), dtype=np.uint8)
        test = cv2.line(test, (int(start.x),int(start.y)), (int(end.x),int(end.y)), 255, lineType=cv2.LINE_AA)
        img = cv2.bitwise_not(self.env.img)
        result = cv2.bitwise_and(img, img, mask=test)
        if 255 in result:
            return True
        
        return False
    

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = Env()
        # self.plotting = Plotting(s_start, s_goal)
        self.utils = Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

    def planning(self):
        # print("z")
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len:
                    self.new_state(node_new, self.s_goal)
                    # print(i)
                    return self.extract_path(node_new)

        return None

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            # print("inside")
            return Node((np.random.uniform(self.x_range[0], self.x_range[1]),
                         np.random.uniform(self.y_range[0], self.y_range[1])))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        # print(len(node_list))
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


# def main():
#     x_start = (300, 500)  # Starting node
#     x_goal = (400, 500)  # Goal node
    
#     rrt = Rrt(x_start, x_goal, 50, .01, 10000)
#     # rrt = Rrt()
#     path = rrt.planning()
#     print(path)

#     # if path:
#         # rrt.plotting.animation(rrt.vertex, path, "RRT", True)
#     # else:
#         # print("No Path Found!")


# if __name__ == '__main__':
#     main()
