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

"""
"C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\Messe.png"
"C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\robot.png"
"""

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

    def plot(self, pathLists):
        self.floor = pygame.transform.scale(self.floor, (1250, 500))
        self.robot = pygame.transform.scale(self.robot, (50, 50))

        self.screen.blit(self.floor, (0, 0))
        self.screen.blit(self.robot, self.get_plot_position(12))
        

        stateID = 0

        for pathList in pathLists:
            font = pygame.font.Font(None, 36)
            text = font.render("Moves: " + str(stateID) + " / " + str(len(pathList)-1), True, (255, 0, 0))
            textRect = text.get_rect()
            self.screen.blit(text, textRect)
            print(pathList)
            while self.running:
                for i in pygame.event.get():
                    if i.type == pygame.QUIT:
                        self.running = False
                    elif i.type == pygame.KEYDOWN:
                        if i.key == pygame.K_ESCAPE:
                            self.running = False
                        elif i.key == pygame.K_RIGHT:
                            stateID = stateID + 1 if stateID < len(pathList)-1 else stateID
                        elif i.key == pygame.K_LEFT:
                            stateID = stateID - 1 if stateID > 0 else stateID
                    self.redraw_everything(stateID, pathList, font, text, textRect)
                        
                pygame.display.update()

class ShowRobot:

    def __init__(self):
        self.reward_matrix = self.read_xslx_into_two_dimensional_list()
        self.q_table, self.state_table = self.create_q_table_and_state_table(self.reward_matrix)
        self.paths = []
        self.states = [x for x in range(31)]
    
    def read_xslx_into_two_dimensional_list(self, file_path="C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\Reward_Matrix_Show_Snapshot.xlsx"):
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

    def create_random_action(self):
        """
        :return: Random possible action for robot to execute.
        """
        possible_action = {1: 'L', 2: 'R', 3: 'U', 4: 'D'}
        rand_number = randint(1, 4)
        return possible_action[rand_number]

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
                row[column_index] = -1
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

    def create_random_service_task(self):
        """
        This is a method for phase 3. Creates a random "service task" and updates the q-table regarding the reward.
        :return: the updated q-table with new "services"
        """
        rand_number_state = randint(0, 30)
        rand_number_reward = random.choice([1, 5, 10])
        possible_actions = []

        for key in self.state_table[rand_number_state]:
            possible_actions.append(key)

        while self.state_table[rand_number_state] != 0:
            rand_number_state = randint(0, 30)
        
        for row in possible_actions:
            self.reward_matrix[row][rand_number_state] = rand_number_reward
        self.q_table, self.state_table = self.create_q_table_and_state_table(self.reward_matrix)
        
    def count_service_tasks(self, reward_matrix):
        """
        :param reward_matrix: The given xlsx reward matrix
        :return: the number of service tasks in the matrix
        """
        count = 0

        for state in self.states:
            for row in reward_matrix:
                if row[state] > 0:
                    count += 1
                    break

        return count

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
        #self.reward_matrix = current_reward_matrix_dataframe.to_numpy()
        exit_state = 20

        # Phase 1
        if current_phase_number == 1:  # Walk until EXIT or time expires and than print out the explored (probably bad path)
            if current_timestep < 100:
                current_timestep += 1
                random_action = self.create_random_action()
                while not self.is_action_valid(current_state_number, random_action):
                    random_action = self.create_random_action()
                state_after_action = self.state_table[current_state_number][random_action]

                if current_state_number == exit_state:
                    return current_timestep, 1, state_after_action, self.reward_matrix
            return current_timestep, 2, current_state_number, self.reward_matrix
            
            

        # Phase 2
        elif current_phase_number == 2:
            current_timestep += 1
            best_action, best_reward = self.get_best_action_from_q_table(new_state_number)

            if self.is_action_valid(new_state_number, best_action):
                old_state = new_state_number
                new_state_number = self.state_table[new_state_number][best_action]
                new_reward_matrix = self.remove_reward_from_matrix(old_state)
                self.q_table, self.state_table = self.create_q_table_and_state_table(new_reward_matrix)

            # go to new service state, when there are no service_tasks left
            if self.count_service_tasks(new_reward_matrix) == 0:
                return 0, 3, current_state_number, self.reward_matrix
            return current_timestep, 2, current_state_number, self.reward_matrix
                
        # Phase 3
        else:
            current_timestep += 1
            self.create_random_service_task()
            best_action, best_reward = self.get_next_best_action(current_state_number)
            if self.is_action_valid(current_state_number, best_action):
                new_state_number = self.state_table[current_state_number][best_action]
                new_reward_matrix = self.remove_reward_from_matrix(current_state_number)
                self.q_table, self.state_table = self.create_q_table_and_state_table(new_reward_matrix)
            if current_timestep > 25:
                return 0, 1, current_state_number, self.reward_matrix
            return current_timestep, 3, current_state_number, self.reward_matrix
            

showRobot = ShowRobot()
path = [12]

for i in range(200):
    timestep, phase, state, reward_matrix = showRobot.optimal_strategy_function(0, 1, 12, showRobot.reward_matrix)
    path.append(state)

plotter = ShowRobotPlot(pathFloorPicture="C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\Messe.png", pathRobotPicture="C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\robot.png")
plotter.plot([path])



    

