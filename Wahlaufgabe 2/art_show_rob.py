import openpyxl
from enum import Enum
import time
from random import randint
import random
import pandas as pd
import pygame

class Action(Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"


class ShowRobotPlot:

    def __init__(self, pathFloorPicture, pathRobotPicture):
        pygame.init()
        self.screen = pygame.display.set_mode((1250, 500))
        pygame.display.set_caption("Art Show Robot")
        self.running = True
        self.floor = pygame.image.load(pathFloorPicture)
        self.robot = pygame.image.load(pathRobotPicture)
    
    def get_plot_position(self, state):
        if state == 0:
            return (200,25)
        elif state == 1:
            return (365, 25)
        elif state == 2:
            return (470, 25)
        elif state == 3:
            return (600, 25)
        elif state == 4:
            return (725, 25)
        elif state == 5:
            return (850, 25)
        elif state == 6:
            return (1000, 25)
        elif state == 7:
            return (100, 225)
        elif state == 8:
            return (365, 150)
        elif state == 9:
            return (600, 150)
        elif state == 10:
            return (850, 150)
        elif state == 11:
            return (1100, 225)
        elif state == 12:
            return (15, 225)
        elif state == 13:
            return (200, 225)
        elif state == 14:
            return (365, 225)
        elif state == 15:
            return (470, 225)
        elif state == 16:
            return (600, 225)
        elif state == 17:
            return (725, 225)
        elif state == 18:
            return (850, 225)
        elif state == 19:
            return (1000, 225)
        elif state == 20:
            return (1180, 225)
        elif state == 21:
            return (365, 350)
        elif state == 22:
            return (600, 350)
        elif state == 23:
            return (850, 350)
        elif state == 24:
            return (200, 425)
        elif state == 25:
            return (365, 425)
        elif state == 26:
            return (470, 425)
        elif state == 27:
            return (600, 425)
        elif state == 28:
            return (725, 425)
        elif state == 29:
            return (850, 425)
        elif state == 30:
            return (1000, 425)
    
    def redraw_everything(self, currentStateID, currentPathList, font, text, textRect):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.floor, (0, 0))
        self.screen.blit(self.robot, self.get_plot_position(currentPathList[currentStateID]))
        text = font.render("Moves: " + str(currentStateID) + " / " + str(len(currentPathList)-1), True, (255, 0, 0))
        self.screen.blit(text, textRect)

    def plot(self, path, title):
        self.floor = pygame.transform.scale(self.floor, (1250, 500))
        self.robot = pygame.transform.scale(self.robot, (50, 50))

        self.screen.blit(self.floor, (0, 0))
        self.screen.blit(self.robot, self.get_plot_position(12))
        
        self.running = True
        stateID = 0
        

        font = pygame.font.Font(None, 36)
        text = font.render("Moves: " + str(stateID) + " / " + str(len(path)-1), True, (255, 0, 0))
        textRect = text.get_rect()
        self.screen.blit(text, textRect)
        
        pygame.display.set_caption(title)

        while self.running:
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    self.running = False
                elif i.type == pygame.KEYDOWN:
                    if i.key == pygame.K_ESCAPE:
                        self.running = False
                    elif i.key == pygame.K_RIGHT:
                        stateID = stateID + 1 if stateID < len(path)-1 else stateID
                    elif i.key == pygame.K_LEFT:
                        stateID = stateID - 1 if stateID > 0 else stateID
                self.redraw_everything(stateID, path, font, text, textRect)
                    
            pygame.display.update()

class ShowRobot:

    def __init__(self, filePath):
        self.reward_matrix = self.read_xslx_into_two_dimensional_list(filePath)
        self.q_table, self.state_table = self.create_q_table_and_state_table(self.reward_matrix)
        self.paths = []
        self.states = [x for x in range(31)]
    
    def read_xslx_into_two_dimensional_list(self, file_path=""):
        result = []
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        for row in sheet.iter_rows(values_only=True):
            result.append(list(row))
        result = [inner[2:] for inner in result[2:]]
        return result

    def create_q_table_and_state_table(self, reward_matrix):
        dict = {}

        for i, inner_list in enumerate(reward_matrix):
            dict[i] = {}
            for j, value in enumerate(inner_list):
                if value != -1:
                    dict[i][j] =  value
        q_table = {}
        state_table = {}

        for i in dict.keys():
            q_table[i] = {}
            state_table[i] = {}
            for j in dict[i].keys():
                if i == 12:
                    possible_keys = dict[i].keys()
                    if 7 in possible_keys:
                        q_table[i]["R"] = dict[i][7]
                        state_table[i]["R"] = 7
                    continue
                elif i == 20:
                    possible_keys = dict[i].keys()
                    if 11 in possible_keys:
                        q_table[i]["L"] = dict[i][11]
                        state_table[i]["L"] = 11
                    continue
                elif i == 7:
                    possible_keys = dict[i].keys()
                    if 12 in possible_keys:
                        q_table[i]["L"] = dict[i][12]
                        state_table[i]["L"] = 12
                    if 13 in possible_keys:
                        q_table[i]["R"] = dict[i][13]
                        state_table[i]["R"] = 13
                    if 0 in possible_keys:
                        q_table[i]["U"] = dict[i][0]
                        state_table[i]["U"] = 0
                    if 24 in possible_keys:
                        q_table[i]["D"] = dict[i][24]
                        state_table[i]["D"] = 24
                    continue
                elif i == 11:
                    possible_keys = dict[i].keys()
                    if 20 in possible_keys:
                        q_table[i]["R"] = dict[i][20]
                        state_table[i]["R"] = 20
                    if 19 in possible_keys:
                        q_table[i]["L"] = dict[i][19]
                        state_table[i]["L"] = 19
                    if 6 in possible_keys:
                        q_table[i]["U"] = dict[i][6]
                        state_table[i]["U"] = 6
                    if 30 in possible_keys:
                        q_table[i]["D"] = dict[i][30]
                        state_table[i]["D"] = 30
                    continue
                if j-1 == i:
                    q_table[i]["R"] = dict[i][j]
                    state_table[i]["R"] = j
                elif j+1 == i:
                    q_table[i]["L"] = dict[i][j]
                    state_table[i]["L"] = j
                elif j+1 < i:
                    q_table[i]["U"] = dict[i][j]
                    state_table[i]["U"] = j
                elif j-1 > i:
                    q_table[i]["D"] = dict[i][j]
                    state_table[i]["D"] = j

        return q_table, state_table     

    def create_random_action(self, possible_actions = ["L", "R", "U", "D"]):
        """
        :return: Random possible action for robot to execute.
        """
        rand_number = randint(0, len(possible_actions)-1)
        return possible_actions[rand_number]

    def is_action_valid(self, current_state_number, action):
        """
        Validates the next move of the roboter.
        :param current_state_number: Integer from [0 , ... , 30]
        :param action: String
        :return: boolean (Is action valid -> true, is action invalid -> false
        """
        for key in self.state_table[current_state_number]:
            if key == action:
                return True
        return False

    def remove_reward_from_matrix(self, state):
        """
        :param reward_matrix: The given xlsx reward matrix
        :param action: The action the robot executes
        :return: new_reward_matrix -> Updated xlsx with the new rewards
        """
        column_index = state
        new_reward_matrix = self.reward_matrix
        for row in new_reward_matrix:
            if row[column_index] != -1:
                row[column_index] = 0
        return new_reward_matrix

    def get_next_best_action(self, current_state_number):
        """
        :param current_state_number: The current state of the robot.
        :return: the maximum reward possible for the robot (greedy method).
        """
        best_reward = 0
        best_action = 0
        for key in self.q_table[current_state_number]:
            value = self.q_table[current_state_number][key]
            if value >= best_reward:
                best_reward = value
                best_action = key

        if best_reward == 0:
            best_action = self.create_random_action()
            while not self.is_action_valid(current_state_number, best_action):
                best_action = self.create_random_action()
        return best_action, best_reward

    def get_next_best_action_with_depth(self, current_state_number, depth):
        if depth == 0:
            return self.get_next_best_action(current_state_number)
        
        bestAction = None
        bestReqard = float('-inf')

        for action in ["L", "R", "D", "U"]:
            if not self.is_action_valid(current_state_number, action):
                continue
            reward = self.q_table[current_state_number][action]

            if depth > 0:
                next_state = self.state_table[current_state_number][action]
                next_action = self.get_next_best_action_with_depth(next_state, depth-1)
                while not self.is_action_valid(next_state, next_action):
                    next_action = self.get_next_best_action_with_depth(next_state, depth-1)
                if next_action is not None:
                    reward += 0.9 * self.q_table[next_state][next_action]
            if reward > bestReqard:
                bestReqard = reward
                bestAction = action
        return bestAction

    def create_random_service_task(self):
        """
        This is a method for phase 3. Creates a random "service task" and updates the q-table regarding the reward.
        :return: the updated q-table with new "services"
        """
        rand_number_state = randint(0, 30)
        rand_number_reward = random.choice([1, 5, 10])

        
        for row in range(len(self.reward_matrix)):
            if self.reward_matrix[row][rand_number_state] != -1:
                self.reward_matrix[row][rand_number_state] = rand_number_reward
        self.q_table, self.state_table = self.create_q_table_and_state_table(self.reward_matrix)
        
    def count_service_tasks(self, reward_matrix):
        """
        :param reward_matrix: The given xlsx reward matrix
        :return: the number of service tasks in the matrix
        """
        count = 0
        service_tasks = []

        for state in self.states:
            for row in reward_matrix:
                if row[state] > 0:
                    service_tasks.append(state)
                    break

        return service_tasks

    def find_path(self, start_state, goal_state):
        copyQ = self.q_table
        copyStates = self.state_table
        visited_states = set()
        path = []

        def dfs(state):
            if state == goal_state:
                return True
            visited_states.add(state)
            for action, next_state in copyStates[state].items():
                if next_state not in visited_states and dfs(next_state):
                    path.append((state, action))
                    return True
            return False

        if dfs(start_state):
            path.reverse()
            return path
        return None

    def optimal_strategy_function(self, current_timestep, current_phase_number, current_state_number, current_reward_matrix_dataframe):
        """
        Implements the optimal strategy function for three given phases.
        Phase 1: No service requests.
        Phase 2: Available service requests given via Matrix.
        Phase 3: Service requests are always available and random new requests are handled.

        :param current_timestep: Integer increment
        :param current_phase_number: [1,2,3]
        :param current_state_number: [0, ... , 30]
        :param current_reward_matrix_dataframe: 31x31 Matrix
        :return: the optimal strategy path as an array [0....30] the initial steps
        """
        exit_state = 20

        current_reward_matrix_dataframe = current_reward_matrix_dataframe.values.tolist()

        # Phase 1
        if current_phase_number == 1:  # Walk until EXIT or time expires and than print out the explored (probably bad path)
            state_after_action = current_state_number
            
            if current_timestep == 100:
                return current_timestep, 2, 12, pd.DataFrame(current_reward_matrix_dataframe)

            current_timestep += 1
            possible_actions = ["L", "R", "U", "D"]
            random_action = self.create_random_action()
            while not self.is_action_valid(current_state_number, random_action):
                possible_actions.remove(random_action)
                random_action = self.create_random_action(possible_actions)
            state_after_action = self.state_table[current_state_number][random_action]

            if state_after_action == exit_state:
                return current_timestep, 2, 12, pd.DataFrame(current_reward_matrix_dataframe)
            return current_timestep, 1, state_after_action, pd.DataFrame(current_reward_matrix_dataframe)

        # Phase 2
        elif current_phase_number == 2:
            current_timestep += 1
            tasks = self.count_service_tasks(current_reward_matrix_dataframe)
            if len(tasks) == 0:
                return 0, 3, 12, pd.DataFrame(current_reward_matrix_dataframe)

            best_action, best_reward = self.get_next_best_action(current_state_number)
            while not self.is_action_valid(current_state_number, best_action):
                best_action, best_reward = self.get_next_best_action(current_state_number)

            new_state_number = self.state_table[current_state_number][best_action]
            current_reward_matrix_dataframe = self.remove_reward_from_matrix(current_state_number)
            self.q_table, self.state_table = self.create_q_table_and_state_table(current_reward_matrix_dataframe)
                
            return current_timestep, 2, new_state_number, pd.DataFrame(current_reward_matrix_dataframe)
                
        # Phase 3
        else:
            current_timestep += 1
            self.create_random_service_task()
            best_action, best_reward = self.get_next_best_action(current_state_number)
            while not self.is_action_valid(current_state_number, best_action):
                best_action, best_reward = self.get_next_best_action(current_state_number)
            new_state_number = self.state_table[current_state_number][best_action]
            current_reward_matrix_dataframe = self.remove_reward_from_matrix(current_state_number)
            self.q_table, self.state_table = self.create_q_table_and_state_table(current_reward_matrix_dataframe)
            if current_timestep > 33:
                return -1, 3, new_state_number, pd.DataFrame(current_reward_matrix_dataframe)
            return current_timestep, 3, new_state_number, pd.DataFrame(current_reward_matrix_dataframe)
            
showRobot = ShowRobot(filePath="./Reward_Matrix_Show_Snapshot.xslx")
path = [12]
current_phase = 1
paths= []
timestep = 0
state = 12

firstPhaseCompleted, secondPhaseCompleted = False, False
reward_matrix = pd.DataFrame(showRobot.reward_matrix)

while True:
    timestep, current_phase, state, reward_matrix = showRobot.optimal_strategy_function(timestep, current_phase, state, reward_matrix)
    path.append(state)
    if current_phase == 2 and not firstPhaseCompleted:
        firstPhaseCompleted = True
        paths.append(path[:-1])
        path = [path[-1]]
    elif current_phase == 3 and not secondPhaseCompleted:
        secondPhaseCompleted = True
        paths.append(path[:-1])
        path = [path[-1]]
    if (current_phase == 3 and timestep == -1):	
        paths.append(path)
        break
    

plotter = ShowRobotPlot(pathFloorPicture="./Messe.png", pathRobotPicture="./robot.png")

for id, path in enumerate(paths):
    if len(path) > 1:
        plotter.plot(path, "Art Show Robot - Phase " + str(id+1))

"""
Controls for the plot:
Right arrow: next step
Left arrow: previous step
ESC: Close the plot
"""



    

