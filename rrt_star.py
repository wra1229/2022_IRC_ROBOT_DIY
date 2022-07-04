import time
import math
import tqdm
import numpy as np

from tool import env, plotting, utils, Node


class RrtStar:
    def __init__(self, start_point, end_point, step_length, search_radius, sample_rate, max_iter, env_instance):
        '''
        :param start_point: start point of the robot
        :param end_point: end point of the robot
        :param step_length: step length of the robot
        :param search_radius: frnn radius
        :param sample_rate: sample rate for finding new node
        :param max_iter: maximum iteration
        :param env_instance:
        '''
        self.st_point = Node(start_point)
        self.ed_point = Node(end_point)
        self.step_length = step_length
        self.search_radius = search_radius
        self.sample_rate = sample_rate
        self.max_iter = max_iter
        self.nodes = [self.st_point]

        # initialize the environment
        self.env = env_instance
        self.plotting = plotting.Plotting(start_point, end_point, self.env)
        self.utils = utils.Utils(self.env)
        self.utils.fix_random_seed()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        # record
        self.iter_num = -1
        self.time_start = -1
        self.time_end = -1
        self.dist = -1

    def planning(self):
        path = None
        self.time_start = time.time()
        # converge = False
        for i in tqdm.tqdm(range(self.max_iter)):
            self.iter_num = i + 1
            node_rand = self.generate_random_node()  # generate new node
            node_near = self.nearest_neighbor(node_rand)  # find the nearest node of the new node in the tree
            node_new = self.new_state(node_near,
                                      node_rand)
            n_d,_=self.get_distance_and_angle(node_new,node_near)
            node_new.cost = node_near.cost + n_d
            # generate the new node by the direction defined by the random node
            if node_new and not self.utils.is_collision(node_near,node_new):
                near_inds = self.find_near_nodes(node_new)
                node_with_updated_parent = self.choose_parent(
                    node_new, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.nodes.append(node_with_updated_parent)
                else:
                    self.nodes.append(node_new)
                if ((not self.max_iter)
                        and node_new):  # if reaches goal
                    last_index = self.search_best_goal_node()
                    if last_index is not None:
                        path = self.extract_path(node_new)
                        self.dist = self.path_distance(path)
                        break


        self.time_end = time.time()
        # return final path
        # implement extract_path func maybe help
        # path = extract_path(...)
        # self.dist = self.path_distance(path)
        return path


    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = self.nodes[i]
            edge_node = self.new_state(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = not self.utils.is_collision(
                edge_node, near_node)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y)
                             for n in self.nodes]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.step_length
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.new_state(self.nodes[goal_ind], self.ed_point)
            if not self.utils.is_collision(
                    t_node, self.ed_point):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.nodes[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.nodes[i].cost == min_cost:
                return i

        return None

    def get_distance_and_angle(self, node_start, node_end):
        '''
        get the distance and angle between two nodes
        :param node_start:
        :param node_end:
        :return:
        '''
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return np.linalg.norm([dx, dy]), np.arctan2(dy, dx)

    def search_goal_parent(self):
        dist_list = [np.linalg.norm(n.x - self.ed_point.x, n.y - self.ed_point.y) for n in self.nodes]
        node_index = [dist_list.index(i) for i in dist_list if i <= self.step_length]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.nodes[i].cost for i in node_index
                         if not self.utils.is_collision(self.nodes[i], self.ed_point)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.nodes) - 1

    def generate_random_node(self):
        '''
        generate a random node (map range as the boundary)
        :return:
        '''
        delta = self.utils.delta

        # uniform sample new node
        if np.random.random() > self.sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        # sample end point as new node
        return self.ed_point

    def nearest_neighbor(self, n):
        '''
        find the nearest node in the tree
        :param n:
        :return:
        '''
        return self.nodes[int(np.argmin([np.linalg.norm([nd.x - n.x, nd.y - n.y])
                                         for nd in self.nodes]))]
    # def get_nearest_node_index(self,rnd):
    def find_near_nodes(self, new_node):

        node_num=len(self.nodes)+1
        dist=[]
        r = min(self.search_radius*math.sqrt((math.log(node_num)/ node_num)),self.step_length)
        dist.append([np.linalg.norm([nd.x - new_node.x, nd.y - new_node.y])
                     for nd in self.nodes])
        dist=np.array(dist).reshape(-1)
        near_inds = [i for i in range(len(dist)) if dist[i] <= r]
        return near_inds

    def choose_parent(self, new_node, near_inds):

        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.nodes[i]
            t_node = self.new_state(near_node, new_node)
            if t_node and not self.utils.is_collision(
                    t_node, new_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.new_state(self.nodes[min_ind], new_node)
        new_node.cost = min_cost

        return new_node



    def calc_dist_to_goal(self, x, y):
        dx = x - self.ed_point.x
        dy = y - self.ed_point.y
        return math.hypot(dx, dy)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.get_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.nodes:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


    def new_state(self, node_start, node_end):
        '''
        generate the new node by the direction defined by the random node
        :param node_start:
        :param node_end:
        :return:
        '''
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = np.min([self.step_length, dist])
        node_new = Node((node_start.x + dist * np.cos(theta),
                         node_start.y + dist * np.sin(theta)))
        node_new.parent = node_start

        return node_new
    def extract_path(self, node_end):
        '''
        extract the path from the end node by backtracking
        :param node_end:
        :return:
        '''
        path = [(self.ed_point.x, self.ed_point.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    def path_distance(self, path):
        '''
        get the distance of the path
        :param path:
        :return:
        '''
        dist_sum = 0
        for i in range(len(path) - 1):
            dist_sum += np.linalg.norm([path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]])
        return dist_sum


def env1_planning(eval_time=1):
    x_start = (5, 5)  # st node
    x_goal = (49, 16)  # end node

    # visualization
    if eval_time == 1:
        rrt = RrtStar(x_start, x_goal, 0.5, 0.05, 0.1, 10000, env.EnvOne())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
        rrt.plotting.animation(rrt.nodes, path, "RRT_STAR_ENV1", True)
        return
    # evaluation
    time_sum = list()
    iter_sum = list()
    dist_sum = list()
    for i in range(eval_time):
        rrt = RrtStar(x_start, x_goal, 0.5, 0.05, 0.1, 10000, env.EnvOne())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("###### Evaluation: {} ######".format(i + 1))
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
            dist_sum.append(rrt.dist)
        print("-----------------------------------------------------")
        time_sum.append(rrt.time_end - rrt.time_start)
        iter_sum.append(rrt.iter_num)

    # average time
    print("Average Time: {:.3f} s".format(np.mean(time_sum)))
    # average iteration
    print("Average Iteration: {:.0f}".format(np.mean(iter_sum)))
    # average distance
    if len(dist_sum) > 0:
        print("Average Distance: {:.3f}".format(np.mean(dist_sum)))


def env2_planning(eval_time=1):
    x_start = (5, 20)  # st node
    x_goal = (67, 40)  # end node

    # visualization
    if eval_time == 1:
        rrt = RrtStar(x_start, x_goal, 0.5, 0.2, 0.1, 10000, env.EnvTwo())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
        rrt.plotting.animation(rrt.nodes, path, "RRT_STAR_ENV2", True)
        return
    # evaluation
    time_sum = list()
    iter_sum = list()
    dist_sum = list()
    for i in range(eval_time):
        rrt = RrtStar(x_start, x_goal, 0.5, 0.2, 0.1, 10000, env.EnvTwo())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("###### Evaluation: {} ######".format(i + 1))
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
            dist_sum.append(rrt.dist)
        print("-----------------------------------------------------")
        time_sum.append(rrt.time_end - rrt.time_start)
        iter_sum.append(rrt.iter_num)

    # average time
    print("Average Time: {:.3f} s".format(np.mean(time_sum)))
    # average iteration
    print("Average Iteration: {:.0f}".format(np.mean(iter_sum)))
    # average distance
    if len(dist_sum) > 0:
        print("Average Distance: {:.3f}".format(np.mean(dist_sum)))


def env3_planning(eval_time=1):
    x_start = (5, 2)  # st node
    x_goal = (18, 18)  # end node

    # visualization
    if eval_time == 1:
        rrt = RrtStar(x_start, x_goal, 0.5, 0.1, 0.2, 10000, env.EnvThree())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        print("Distance:{:.3f}".format(rrt.dist))
        rrt.plotting.animation(rrt.nodes, path, "RRT_STAR_ENV3", True)
        return
    # evaluation
    time_sum = list()
    iter_sum = list()
    dist_sum = list()
    for i in range(eval_time):
        rrt = RrtStar(x_start, x_goal, 0.5, 0.2, 0.1, 10000, env.EnvThree())
        path = rrt.planning()
        if not path:
            print("No Path Found!")
        print("###### Evaluation: {} ######".format(i + 1))
        print("Time: {:.3f} s".format(rrt.time_end - rrt.time_start))
        print("Iteration:", rrt.iter_num)
        if path:
            print("Distance:{:.3f}".format(rrt.dist))
            dist_sum.append(rrt.dist)
        print("-----------------------------------------------------")
        time_sum.append(rrt.time_end - rrt.time_start)
        iter_sum.append(rrt.iter_num)

    # average time
    print("Average Time: {:.3f} s".format(np.mean(time_sum)))
    # average iteration
    print("Average Iteration: {:.0f}".format(np.mean(iter_sum)))
    # average distance
    if len(dist_sum) > 0:
        print("Average Distance: {:.3f}".format(np.mean(dist_sum)))


if __name__ == '__main__':
    env1_planning(eval_time=1)
    # env2_planning(eval_time=1)
    # env3_planning(eval_time=1)
