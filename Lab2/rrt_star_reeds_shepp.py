
import copy
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import reeds_shepp_path_planning
except ImportError:
    raise

show_animation = True


class RRTStarReedsShepp():
    """
    Class for RRT star planning with Reeds Shepp path
    """

    class Node():
        """
        RRT Node
        """

        def __init__(self, x, y, theta):
            self.x = x
            self.y = y
            self.theta = theta
            self.parent = None
            self.path_x = []
            self.oath_y = []
            self.path_theta = []
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area,
                 leftway_list=[],
                 rightway_list=[],
                 max_iter=200,
                 goal_sample_rate=20,
                 connect_circle_dist=50.0
                 ):
        """
        Setting Parameter

        start:Start Position [x,y,theta]
        goal:Goal Position [x,y,theta]
        obstacleList:obstacle Positions [[x1,y1,w,h],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacle_list
        self.leftway_list = leftway_list
        self.rightway_list = rightway_list
        self.connect_circle_dist = connect_circle_dist

        self.curvature = 2.0  
        self.diff_goal_theta = np.deg2rad(1.0)
        self.diff_goal_dis = 0.5 

    def planning(self, animation=True, search_until_max_iter=True):
        """
        planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
#            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if new_node and self.check_collision(new_node, self.obstacle_list, self.leftway_list, self.rightway_list):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)
                    self.try_goal_path(new_node)

            if animation and i % 5 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def try_goal_path(self, node):

        goal = self.Node(self.end.x, self.end.y, self.end.theta)

        new_node = self.steer(node, goal)
        if new_node is None:
            return

        if self.check_collision(new_node, self.obstacle_list, self.leftway_list, self.rightway_list):
            self.node_list.append(new_node)

    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (x1, y1, w, h) in self.obstacle_list:
            self.plot_obstacle(x1, y1, w, h)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([0, 16, 0, 16])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)
    
    def plot_obstacle(self, x1, y1, w, h, color="k"):  
        x = [x1, x1 + w, x1 + w, x1]
        y = [y1, y1, y1 + h, y1 + h]
        plt.fill(x, y, facecolor=color,alpha=1.0)

    def plot_start_goal_arrow(self):
        reeds_shepp_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.theta)
        reeds_shepp_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.theta)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.theta,
            to_node.x, to_node.y, to_node.theta, self.curvature)

        if not px:
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.theta = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_theta = pyaw
        new_node.cost += sum([abs(l) for l in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.theta,
            to_node.x, to_node.y, to_node.theta, self.curvature)
        if not course_lengths:
            return float("inf")

        return from_node.cost + sum([abs(l) for l in course_lengths])

    def get_random_node(self):
                
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(-math.pi, math.pi))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.theta)
        return rnd
    
    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind
    
    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.sqrt(dx ** 2 + dy ** 2)
    
    def check_collision(self, node, obstacleList, leftwayList, rightwayList):
        collision_free = True
        if obstacleList:
            for (x, y) in zip(node.path_x, node.path_y):
                for ob in obstacleList:
                    x_low = ob[0] - 88/100
                    x_high = ob[0] + ob[2] + 88/100
                    y_low = ob[1] - 88/100
                    y_high = ob[1] + ob[3] + 88/100
                    if x > x_low and x < x_high and y > y_low and y < y_high:
                        collision_free = False
#        if leftwayList:
#            for (x, y, theta) in zip(node.path_x, node.path_y, node.path_theta):
#                for l in leftwayList:
#                    x_low = l[0]
#                    x_high = l[0] + l[2] 
#                    y_low = l[1] 
#                    y_high = l[1] + l[3] 
#                    if x > x_low and x < x_high and y > y_low and y < y_high:
#                        if theta <= np.deg2rad(90.0) and theta >= np.deg2rad(-90.0) \
#                        and 
                    
        return collision_free
    
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1 # avoid log0
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds
    
    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list, self.leftway_list, self.rightway_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node
    
    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list, self.leftway_list, self.rightway_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                self.propagate_cost_to_leaves(new_node)

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.diff_goal_dis:
                goal_indexes.append(i)
        print("goal_indexes:", len(goal_indexes))

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].theta - self.end.theta) <= self.diff_goal_theta:
                final_goal_indexes.append(i)

        print("final_goal_indexes:", len(final_goal_indexes))

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        print("min_cost:", min_cost)
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i
        return None
    
    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def generate_final_course(self, goal_index):
        path = [[self.end.x, self.end.y, self.end.theta]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy, itheta) in zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_theta)):
                path.append([ix, iy, itheta])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.theta])
        return path

def draw_final_traj(path, start, goal, obstacleList):
    ax = draw_obstacles_start_goal(obstacleList, start, goal)

    for pnt in path:
        theta = pnt[2]
        rec_corner = [pnt[0]+75/100*np.cos(theta+np.pi)+45/100*np.cos(theta-np.pi/2),
                      pnt[1]+75/100*np.sin(theta+np.pi)+45/100*np.sin(theta-np.pi/2)]
        robot = plt.Rectangle(rec_corner, 100/100, 90/100, np.degrees(theta), facecolor='w', edgecolor='b')
        ax.add_patch(robot)
    plt.show()
    
def draw_obstacles_start_goal(obstacleList, start, goal):
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(1,1,1)
    plt.xlim((0, 16))
    plt.ylim((0, 16))
    plt.grid()    
#    Plot initial (red) and goal (green) states
    plt.arrow(start[0], start[1], 0.4*np.cos(start[2]),
              0.4*np.sin(start[2]), color='r', width=0.1)
    plt.arrow(goal[0], goal[1], 0.4*np.cos(goal[2]),
              0.4*np.sin(goal[2]), color='g', width=0.1)              
    # Plot the obstacles
    for ob in obstacleList:
        obstacle = plt.Rectangle(ob[0:2], ob[2], ob[3], color = 'k')
        ax.add_patch(obstacle)
    return ax
    
def main():

    # ====Search Path with RRT====
#    obstacleList = [
#        (-2, -2, 1, 18),
#        (-1, -2, 17, 1),
#        (-1, 15, 17, 1),
#        (15,-1, 1, 16),
#        (4, 6, 6, 1),
#        (7, 7, 1, 4),
#        (7, 2, 1, 4),
#        (10, 2, 1, 10)
#    ]  # [x1,y1,w,h]
#    obstacleList = [
#            (0, 0, 50, 1000),
#            (50, 0, 950, 50),
#            (950, 50, 50, 950),
#            (50, 950, 900, 50)]
    obstacleList = [
            (0, 0, 0.5, 16),
            (0.5, 0, 15.5, 0.5),
            (15.5, 0.5, 0.5, 15.5),
            (0.5, 15.5, 15.0, 0.5),
            (3, 7.25, 10, 0.5),
            (3, 0.5, 1, 1),
            (4.5, 0.5, 1, 1),
            (10, 0.5, 1, 1),
            (7, 6, 1, 1),
            (14, 0.5, 1, 1),
            (9, 6, 1, 1),
            (4, 14, 1, 1),
            (6, 14, 1, 1),
            (9, 14, 1, 1),
            (8, 8, 1, 1)]
    leftwayList = []
    rightwayList = []

    # Set Initial parameters
    start = [2.0, 4.0, np.deg2rad(30.0)]
    goal = [12.0, 14.0, np.deg2rad(180.0)]
    max_iter = 500
    connect_circle_dist = 50.0
    rand_area = [0.0, 15.0]
    
    

    rrt_star_reeds_shepp = RRTStarReedsShepp(start, goal,
                                             obstacleList,                               
                                             rand_area, max_iter=max_iter,
                                             leftway_list=leftwayList,
                                             rightway_list=rightwayList,
                                             connect_circle_dist=connect_circle_dist)
    path = rrt_star_reeds_shepp.planning(animation=show_animation)

    # Draw final path
    if path and show_animation:  # pragma: no cover
#        rrt_star_reeds_shepp.draw_graph()
#        plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
#        plt.grid(True)
#        plt.pause(0.001)
#        plt.show()
#        plt.clf()
#
#        for (x1, y1, w, h) in obstacleList:
#            rrt_star_reeds_shepp.plot_obstacle(x1, y1, w, h)
#        
#        plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
#        reeds_shepp_path_planning.plot_arrow(
#            start[0], start[1], start[2])
#        reeds_shepp_path_planning.plot_arrow(
#            goal[0], goal[1], goal[2])
#
#        plt.plot(start[0], start[1], "xb")
#        plt.plot(goal[0], goal[1], "xr")
#        plt.axis([-2, 16, -2, 16])
#        plt.grid(True)
#        plt.pause(0.01)
#        plt.show()
        draw_final_traj(path, start, goal, obstacleList)
        
if __name__ == '__main__':
    main()
