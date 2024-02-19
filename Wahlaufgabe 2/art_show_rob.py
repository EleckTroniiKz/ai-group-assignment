import openpyxl
from enum import Enum

class Action(Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"



def read_xslx_into_array(file_path="C:\\Users\\Can\\Documents\\Programming\\ai-project\\ai-group-assignment\\Wahlaufgabe 2\\Reward_Matrix_Show_Snapshot.xlsx"):
    result = []
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    for row in sheet.iter_rows(values_only=True):
        result.append(list(row))
    result = [inner[2:] for inner in result[2:]]
    return result

def create_q_table_and_state_table():
    nested_list = read_xslx_into_array()
    dict = {}

    for i, inner_list in enumerate(nested_list):
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
            elif i == 7:
                q_table[i]["L"] = dict[i][12]
                q_table[i]["R"] = dict[i][13]
                state_table[i]["L"] = 12
                state_table[i]["R"] = 13
            elif i == 11:
                q_table[i]["L"] = dict[i][19]
                q_table[i]["R"] = dict[i][20]
                state_table[i]["L"] = 19
                state_table[i]["R"] = 20
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

q_table, state_table = create_q_table_and_state_table()



"""
    q_table will return the reward that will be granted, when being at state x, and doing action y -> q_table[x][y]. All available actions can be received by calling q_table[x].keys()
    state_table will return the state that will be reached, when being at state x, and doing action y -> state_table[x][y]. All available actions can be received by calling state_table[x].keys(). 
"""