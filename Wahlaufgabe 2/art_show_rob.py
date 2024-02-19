import openpyxl
from enum import Enum

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
                    id = id + 1 if id < 30 else id
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
                q_table[i]["R"] = dict[i][j]
                state_table[i]["R"] = 7
                continue
            elif i == 20:
                q_table[i]["L"] = dict[i][j] 
                state_table[i]["L"] = 11
                continue
            elif i == 7:
                q_table[i]["L"] = dict[i][12]
                q_table[i]["R"] = dict[i][13]
                q_table[i]["U"] = dict[i][0]
                q_table[i]["D"] = dict[i][24]
                state_table[i]["L"] = 12
                state_table[i]["R"] = 13
                state_table[i]["U"] = 0
                state_table[i]["D"] = 24
                continue
            elif i == 11:
                q_table[i]["L"] = dict[i][19]
                q_table[i]["R"] = dict[i][20]
                q_table[i]["U"] = dict[i][6]
                q_table[i]["D"] = dict[i][30]
                state_table[i]["L"] = 19
                state_table[i]["R"] = 20
                state_table[i]["U"] = 6
                state_table[i]["D"] = 30
                continue
            elif i == 13:
                q_table[i]["L"] = dict[i][7]
                q_table[i]["R"] = dict[i][14]
                state_table[i]["L"] = 7
                state_table[i]["R"] = 14
                continue
            elif i == 19:
                q_table[i]["L"] = dict[i][18]
                q_table[i]["R"] = dict[i][11]
                state_table[i]["L"] = 18
                state_table[i]["R"] = 11
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

def remove_reward_from_matrix(matrix, state):
    """
    might work already. NOts ure
    """
    for i in range(len(matrix)):
        matrix[i][state] = 0
    return matrix

reward_matrix = read_xslx_into_array()
q_table, state_table = create_q_table_and_state_table(reward_matrix)

plot_movements([0, 5, 8, 12, 20, 1])

"ich berechne den nächsten move"
# remove_reward_from_matrix(reward_matrix, state) # state is the state that the robot is currently in
# q_table, state_table = create_q_table_and_state_table(reward_matrix)



"""
    q_table will return the reward that will be granted, when being at state x, and doing action y -> q_table[x][y]. All available actions can be received by calling q_table[x].keys()
    state_table will return the state that will be reached, when being at state x, and doing action y -> state_table[x][y]. All available actions can be received by calling state_table[x].keys(). 
"""