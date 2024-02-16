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

def create_q_table():
    nested_list = read_xslx_into_array()
    dict = {}

    for i, inner_list in enumerate(nested_list):
        dict[i] = {}
        for j, value in enumerate(inner_list):
            if value != -1:
                dict[i][j] =  value
    q_table = {}

    for i in dict.keys():
        q_table[i] = {}
        for j in dict[i].keys():
            if i == 12:
                q_table[i]["R"] = dict[i][j]
                continue
            elif i == 20:
                q_table[i]["L"] = dict[i][j] 
            elif i == 7:
                q_table[i]["L"] = dict[i][12]
                q_table[i]["R"] = dict[i][13]
            elif i == 11:
                q_table[i]["L"] = dict[i][19]
                q_table[i]["R"] = dict[i][20]
            if j-1 == i:
                q_table[i]["R"] = dict[i][j]
            elif j+1 == i:
                q_table[i]["L"] = dict[i][j]
            elif j+1 < i:
                q_table[i]["U"] = dict[i][j]
            elif j-1 > i:
                q_table[i]["D"] = dict[i][j] 

    return q_table          

q_table = create_q_table()

# to use q_table, you have to call q_table[state]][action] for example. The Actions are defined in the Action Enum

"""
    e.g. q_table[5][Action.UP] -> 5 is the state and Action.UP is the action
"""