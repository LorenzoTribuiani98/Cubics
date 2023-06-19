import numpy as np
from copy import deepcopy

# block mapping: shape -> int
BLOCK_MAP = {
    (1,1) : 0,
    (1,2) : 1,
    (1,3) : 2,
    (2,1) : 3,
    (2,2) : 4,
    (2,3) : 5,
    (3,1) : 6,
    (3,2) : 7,
    (3,3) : 8,
}

REVERSE_BLOCK_MAP_ = {
    0 : [1,1],
    1 : [1,2],
    2 : [1,3],
    3 : [2,1],
    4 : [2,2],
    5 : [2,3],
    6 : [3,1],
    7 : [3,2],
    8 : [3,3],
}

REVERSE_BLOCK_MAP = {
    0 : [[0,0,0],
         [0,0,0],
         [1,0,0]],
    1 : [[0,0,0],
         [0,0,0],
         [1,1,0]],
    2 : [[0,0,0],
         [0,0,0],
         [1,1,1]],
    3 : [[0,0,0],
         [1,0,0],
         [1,0,0]],
    4 : [[0,0,0],
         [1,1,0],
         [1,1,0]],
    5 : [[0,0,0],
         [1,1,1],
         [1,1,1]],
    6 : [[1,0,0],
         [1,0,0],
         [1,0,0]],
    7 : [[1,1,0],
         [1,1,0],
         [1,1,0]],
    8 : [[1,1,1],
         [1,1,1],
         [1,1,1]],
}
# possible actions mapping: int -> (block code, rotation)
ACTION_MAP = {
    0 : (0,0),
    1 : (1,0),
    2 : (2,0),
    3 : (3,0),
    4 : (4,0),
    5 : (5,0),
    6 : (6,0),
    7 : (7,0),
    8 : (8,0),
    9 : (9,0),
    10: (0,1),
    11: (1,1),
    12: (2,1),
    13: (3,1),
    14: (4,1),
    15: (5,1),
    16: (6,1),
    17: (7,1),
    18: (8,1),
    19: (9,1),
}

def compute_reward(field: np.ndarray, block: int, action: int,  bad_placing: bool, row_n:int = 3, col_n:int = 10) -> int:
    """
    Compute the reward for the current state-action
    
    Parameters:
    -------------
    
    - return_dict: a dictionary storing previous informations about the observation
    - game: the current game state
    - bad_placing: positioning of the block
    
    Returns:
    -------------
    
    - reward: the reward value
    """
    return_dict = create_reward_dict(field, row_n = row_n)
    field, _ = create_new_state(field, block, action)
    if field is None:
        return 0
    completed_lines = 0
    consecutive_rows = []
    longest_row = 0
    for row in field[-row_n:]:
        if np.all(row):
            completed_lines += 1

        if row[0] != 0:
            consecutive_rows.append(row)
            consecutive_len = 0
            for i in range(col_n):
                if row[i] != 0:
                    consecutive_len += 1
                if row[i] == 0:
                    if consecutive_len > longest_row:
                        longest_row = consecutive_len
                    break
            
    max_height = 0
    unreachable = 0
    previous = 0
    bump = []
    for i, col in enumerate(field.T):
        try:
            temp = 20 - np.min(np.nonzero(col)) 
        except:
            temp = 0
        if i==0:
            previous = temp
        else:
            bump.append(abs(previous - temp))
            previous = temp

        if temp > max_height:
            max_height = temp 
        
        flag = False        
        for i in range(20-row_n, 20):
            if col[i] != 0:
                flag = True
                continue
            if flag and col[i] == 0:
                unreachable += 1
            if flag and col[i] != 0:
                flag = False
                continue
    
    bumpiness = np.mean(bump)
    reward = completed_lines *10 + \
        longest_row - \
        unreachable - bumpiness
        #(max_height - return_dict["height"])*0.2 - \
    
    # if bad_placing:
    #     reward = -10
    
    return reward


def create_reward_dict(field, row_n=3, col_n=10):

    """
    Given the current state returns the dictionary used by the compute reward function (only for develop)"""
    
    return_dict = {}
    completed_lines = 0
    consecutive_rows = []
    longest_row = 0
    for row in field[-row_n:]:
        if np.all(row):
            completed_lines += 1

        if row[0] != 0:
            consecutive_rows.append(row)
            consecutive_len = 0
            for i in range(col_n):
                if row[i] != 0:
                    consecutive_len += 1
                if row[i] == 0:
                    if consecutive_len > longest_row:
                        longest_row = consecutive_len
                    break
            
    max_height = 0
    unreachable = 0
    for i, col in enumerate(field.T):
        try:
            temp = 20 - np.min(np.nonzero(col)) 
        except:
            temp = 0

        if temp > max_height:
            max_height = temp 
        
        flag = False        
        for i in range(20-row_n, 20):
            if col[i] != 0:
                flag = True
                continue
            if flag and col[i] == 0:
                unreachable += 1
            if flag and col[i] != 0:
                flag = False
                continue
 
    return_dict["height"] = max_height
    return_dict["longest"] = 0 if longest_row == 10 else longest_row 
    return_dict["consecutives"] = len(consecutive_rows)
    return_dict["unreachable"] = unreachable

    return return_dict


def create_new_state(field_, block, action):
    field = deepcopy(field_)
    block_shape = [index for index in BLOCK_MAP if BLOCK_MAP[index]==block][0]
    if action[1]:
        block_shape = (block_shape[1], block_shape[0])

    bad_placing = False                                     # if bad placed a random x is assigned
    if not(0 <= action[0] < 10 - block_shape[0] + 1):
        bad_placing=True
        x = np.random.randint(0,10 - block_shape[0] + 1)
    else:
        x = action[0]

    nonzero = np.nonzero(field.T[x:x+block_shape[0]])[1]
    if len(nonzero) > 0:
        y = min(nonzero)
    else:
        y = 20
    y = y - block_shape[1]
    if y < 0:
        return (None, False)
    
    field[y:y+block_shape[1], x:x+block_shape[0]] = 1

    return(field, bad_placing)

def check_lines(field):
    
    score = 0
    for i in range(field.shape[0]):
        if sum(field[i, :]) == 10:
            score += 10
            field = np.delete(field, i, axis=0)
            field = np.vstack((np.zeros([10]), field))

    return (field, score)
