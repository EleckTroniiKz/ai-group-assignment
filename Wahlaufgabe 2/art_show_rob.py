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

def get_plot_position(state):
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

def plot_movements(moveList):
    if len(moveList) == 0:
        print("LIST IS EMPTY")
        return
    pygame.init()
    screen = pygame.display.set_mode((1250, 500))
    pygame.display.set_caption("Art Show Robot")
    clock = pygame.time.Clock()
    running = True
    imp = pygame.image.load("C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\Messe.png")
    imp = pygame.transform.scale(imp, (1250, 500))
    screen.blit(imp, (0, 0))

    robot = pygame.image.load("C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\robot.png")
    robot = pygame.transform.scale(robot, (50, 50))
    screen.blit(robot, get_plot_position(moveList[0]))
    pygame.display.flip()

    id = 0
   
    while running:
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                running = False
            elif i.type == pygame.KEYDOWN:
                if i.key == pygame.K_ESCAPE:
                    running = False
                elif i.key == pygame.K_RIGHT:
                    id = id + 1 if id < len(moveList) else id
                    screen.fill((255, 255, 255))
                    screen.blit(imp, (0, 0))
                    screen.blit(robot, get_plot_position(moveList[id]))
                elif i.key == pygame.K_LEFT:
                    id = id - 1 if id > 0 else id
                    screen.fill((255, 255, 255))
                    screen.blit(imp, (0, 0))
                    screen.blit(robot, get_plot_position(moveList[id]))
        pygame.display.update()
    pygame.quit()       

def read_xslx_into_array(file_path="C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\Reward_Matrix_Show_Snapshot.xlsx"):
    result = []
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    for row in sheet.iter_rows(values_only=True):
        result.append(list(row))
    result = [inner[2:] for inner in result[2:]]
    return result

def create_q_table_and_state_table(reward_matrix):
    
 
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


reward_matrix = read_xslx_into_array()
q_table, state_table = create_q_table_and_state_table(reward_matrix)
new_q_table, new_state_table = q_table, state_table
new_reward_matrix = reward_matrix



"""
    q_table will return the reward that will be granted, when being at state x, and doing action y -> q_table[x][y]. All available actions can be received by calling q_table[x].keys()
    state_table will return the state that will be reached, when being at state x, and doing action y -> state_table[x][y]. All available actions can be received by calling state_table[x].keys(). 
"""

def create_random_action():
    """
    :return: Random possible action for robot to execute.
    """
    possible_action = {1: 'L', 2: 'R', 3: 'U', 4: 'D'}
    rand_number = randint(1, 4)
    return possible_action[rand_number]

def is_action_valid(current_state_number, action):
    """
    Validates the next move of the roboter.
    :param current_state_number: Integer from [0 , ... , 30]
    :param action: String
    :return: boolean (Is action valid -> true, is action invalid -> false
    """
    for key in state_table[current_state_number]:
        if key == action:
            return True
        else:
            continue
    return False


def create_new_reward_matrix(reward_matrix, action):
    """
    :param reward_matrix: The given xlsx reward matrix
    :param action: The action the robot executes
    :return: new_reward_matrix -> Updated xlsx with the new rewards
    """
    column_index = action
    new_reward_matrix = reward_matrix
    for row in new_reward_matrix:
        if row[column_index] != -1:
            row[column_index] = -1
    return new_reward_matrix


def get_best_action_from_q_table(current_state_number):
    """
    :param current_state_number: The current state of the robot.
    :return: the maximum reward possible for the robot (greedy method).
    """
    best_reward = 0
    best_action = 0
    for key in new_q_table[current_state_number]:
        value = new_q_table[current_state_number][key]
        if value >= best_reward:
            best_reward = value
            best_action = key

    if best_reward == 0:
        best_action = create_random_action()
    #print(best_action)
    #print(best_reward)
    return best_action, best_reward


def create_random_service_task():
    """
    This is a method for phase 3. Creates a random "service task" and updates the q-table regarding the reward.
    :return: the updated q-table with new "services"
    """
    global new_reward_matrix, new_q_table
    rand_number_state = randint(0, 30)
    rand_number_reward = random.choice([1, 5, 10])
    possible_actions = []
    for value in state_table[rand_number_state].values():
        possible_actions.append(value)
    for row_index in possible_actions:
        if new_reward_matrix[row_index][rand_number_state] == -1:
            new_reward_matrix[row_index][rand_number_state] = rand_number_reward
    create_q_table_and_state_table(new_reward_matrix)


def optimal_strategy_function(current_timestep, current_phase_number, current_state_number, current_reward_matrix_dataframe):
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

    list_of_rewards = []
    new_state_number = current_state_number
    global new_reward_matrix, new_q_table, new_state_table
    new_reward_matrix = current_reward_matrix_dataframe.to_numpy()
    path = [current_state_number]
    exit_state = 20

    # Phase 1
    if current_phase_number == 1:  # Walk until EXIT or time expires and than print out the explored (probably bad path)
        print("PHASE 1 EXECUTED")
        starting_time = current_timestep
        while True:
            current_time = time.time()
            elapsed_time = current_time - starting_time
            if elapsed_time >= 60:  # Try to execute phase one for a minute...
                break
            random_action = create_random_action()
            is_valid = is_action_valid(new_state_number, random_action)
            if is_valid:
                reward = q_table[new_state_number][random_action]
                list_of_rewards.append(reward)
                new_state_number = state_table[new_state_number][random_action]
                path.append(new_state_number)

            if new_state_number == exit_state:
                print('Total Path:', path)
                print('Reward List:', list_of_rewards)
                return path

    # Phase 2
    elif current_phase_number == 2:
        print("PHASE 2 EXECUTED")
        while True:
            best_action, best_reward = get_best_action_from_q_table(new_state_number)
            is_valid = is_action_valid(new_state_number, best_action)
            if is_valid:
                list_of_rewards.append(best_reward)
                new_state_number = state_table[new_state_number][best_action]
                path.append(new_state_number)
                new_reward_matrix = create_new_reward_matrix(new_reward_matrix, new_state_number)
                new_q_table, new_state_table = create_q_table_and_state_table(new_reward_matrix)
                #print(pd.DataFrame(new_reward_matrix))
                #print(new_q_table)
            print('Total Path:', path)
            print('Reward List:', list_of_rewards)
            exit_condition = -1
            # Iteriere über alle Zeilen
            for i in range(len(new_reward_matrix)):
                # Iteriere über alle Spalten
                for j in range(len(new_reward_matrix[i])):
                    if new_reward_matrix[i][j] != -1:
                        exit_condition = new_reward_matrix[i][j]

            if exit_condition == -1:
                break
        return path

    # Phase 3
    else:
        print("PHASE 3 EXECUTED")
        starting_time = current_timestep
        while True:
            current_time = time.time()
            elapsed_time = current_time - starting_time
            if elapsed_time >= 6:  # Try to execute phase three for six seconds...
                break
            create_random_service_task()
            best_action, best_reward = get_best_action_from_q_table(new_state_number)
            is_valid = is_action_valid(new_state_number, best_action)
            if is_valid:
                list_of_rewards.append(best_reward)
                new_state_number = state_table[new_state_number][best_action]
                path.append(new_state_number)
                new_reward_matrix = create_new_reward_matrix(new_reward_matrix, new_state_number)
                new_q_table, new_state_table = create_q_table_and_state_table(new_reward_matrix)
                #print(pd.DataFrame(new_reward_matrix))
                #print(new_q_table)
            print('Total Path:', path)
            print('Reward List:', list_of_rewards)
            return path


start_time = time.time()
new_path = []
for index in range(1, 4):
    path = optimal_strategy_function(start_time, 2, 0, pd.DataFrame(reward_matrix))
    print("RETURNED PATH:")
    new_path = path

print("FINAL PATH:", new_path)
plot_movements(new_path)


"""
    Beurteilen Sie, inwiefern der Einsatz von Dataset Aggregation zur Effizienzsteigerung Ihres Roboters bei einer realen Kunstmesse beitragen kann.

    Dataset Aggregation allows us to combine Data from different sources to train the model. In the case how we learned it in class, it is by "demonstrating" to the robot. And then iteratively letting the robot try, and find out how to maximize his reward during the process.
    DAgger can help to reduce overfitting, because we have more Data available. In this case the people's input. And also in the context of the art gallery, it can help to make the robot more dynamic and flexible. It can learn from the people's input and adapt to the environment, even when the layout changes. 
    
    So all in all, can the robot learn from the people's input and its already existing knowledge, and be more efficient with those, and prevent overfitting. 

"""