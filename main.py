import pygame
from Cubics import Cubics, colors
import minizinc
from datetime import timedelta
import numpy as np
import os
import threading
from copy import deepcopy
from Paradigms.q_learning.Q_table import q_table_solve, create_observation, load_table
from Paradigms.utils.utils import ACTION_MAP, BLOCK_MAP, REVERSE_BLOCK_MAP
import torch
from Paradigms.q_learning.DeepQNet import Agent



width = 700
height = 650
gameWidth = 100  # meaning 300 // 10 = 30 width per block
gameHeight = 400  # meaning 600 // 20 = 20 height per blo ck
blockSize = 20
 
topLeft_x = (width - gameWidth) // 2
topLeft_y = height - gameHeight - 50

font_small_size = 18
font_medium_size = 35
font_big_size = 60
font_bigger_size = 80

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Cubics")
pygame.init()

font_small = pygame.font.SysFont('hpsimplified', font_small_size, False, False) 
font_medium = pygame.font.SysFont('hpsimplified', font_medium_size, False, False) 
font_big = pygame.font.SysFont('hpsimplified', font_big_size, False, False)
font_bigger = pygame.font.SysFont('hpsimplified', font_bigger_size, True, False)

##_________UI & Utils____________

class Button:

    def __init__(self, rect, text, function, kwargs=None):
        self.rect = rect
        self.text = text
        self.function = function
        self.args = kwargs
        self.background = "#202020"

    def on_click(self, event):
        x, y = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                if self.rect.collidepoint(x, y):
                    if self.args == None:
                        self.function()
                    else:
                        self.function(**self.args)

    def show(self):
        draw_box(self.rect, self.background)
        rendered_text = font_medium.render(self.text, True, "#A5C9CA")
        displace_h = rendered_text.get_bounding_rect()[2] // 2
        displace_v = rendered_text.get_bounding_rect()[3] // 2

        screen.blit(rendered_text, (self.rect[0] + self.rect[2]//2 - displace_h, self.rect[1] + self.rect[3]//2 - displace_v))

    def on_hover(self):
        x, y = pygame.mouse.get_pos()
        if self.rect.collidepoint(x,y):
            self.background = "#404040"
        else:
            self.background = "#202020"

def draw_box(rect, background):
    pygame.draw.rect(
        screen,
        background,
        rect
    )
    pygame.draw.rect(
        screen,
        "#E7F6F2",
        [rect[0] - 1, rect[1] - 1, rect[2] + 2, rect[3] + 3],
        1
    )

def update_UI(game):
    cubics_label = font_big.render("Cubics", True, '#A5C9CA')
    screen.blit(cubics_label, [(width // 2) - 60, 30])

    draw_box([game.x + game.width * game.zoom + 50, game.y + 50, 260, 50], "#202020")
    score_text = font_medium.render("Score", True, '#A5C9CA')
    score_points = font_medium.render(str(game.score),True, "#A5C9CA")
    screen.blit(score_text, [game.x + game.width * game.zoom + 140, game.y + 8])
    screen.blit(score_points, [game.x + game.width * game.zoom + 60, game.y + 58])
        
    label = font_medium.render("Next Shape", True, '#A5C9CA')        
    draw_box([game.x + game.width * game.zoom + 50, game.y + 199, 260, 300], "#202020")
    screen.blit(label, [game.x + game.width * game.zoom + 100, game.y + 148])    
        
    sy =  game.x + game.width * game.zoom + 180 - ((game.next_block.width * 50) // 2)
    sx = game.y + 350 - ((game.next_block.height * 50) // 2)

    for i in range(game.next_block.width):
        for j in range(game.next_block.height):
            pygame.draw.rect(screen, '#202020', [sy + 50*i, sx + 50*j, 50, 50], 1)
            pygame.draw.rect(screen, colors[game.next_block.color],[sy + i*50 + 1, sx + j*50 + 1, 48, 49])

def update_field_UI(game):
    #displaying field
        screen.fill('#2C3333')
        pygame.draw.rect(
            screen,
            "#E7F6F2",
            [game.x - 1, game.y - 1,game.width * game.zoom + 3, game.height * game.zoom + 3],
            1
        )
        pygame.draw.rect(
            screen,
            "#202020",
            [game.x, game.y,game.width * game.zoom, game.height * game.zoom ],
        )

        #displaying blocks
        for i in range(game.height):
            for j in range(game.width):
                pygame.draw.rect(screen, '#202020', [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
                if 0 < game.field[i][j] < 10:
                    pygame.draw.rect(screen, colors[game.field[i][j]],
                                     [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 1, game.zoom - 1])

def display_game_over(game):

    draw_box([50,100, width-100, height-200], "#2C3333")
    button1 = Button(pygame.Rect(70, height - 190, 120, 70), "Replay", game.__init__,kwargs={"width": 10, "height": 20})
    button1.on_hover()
    game_over_txt = font_bigger.render("Game Over", True, '#A5C9CA')
    screen.blit(game_over_txt, [width//2 - game_over_txt.get_bounding_rect()[2]//2, 120])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        button1.on_click(event)

    button1.show()
    return False


#regular game
def start_game():    
    done = False
    clock = pygame.time.Clock()
    fps = 25
    game = Cubics(10,20)
    counter = 0

    while not done:
        if game.state == "start":
            #checks for current block to be initialized
            if game.current_block is None:
                game.gen_new_block()
            if game.next_block is None:
                game.gen_next_block()
        
            #keep track of time
            counter += 1 
            if counter > 100000:
                counter = 0

            #move piece down based on level
            if counter % (fps // game.level // 2) == 0:
                if game.state == "start":
                    game.move_down() 
        
            #event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        game.rotate()
                    if event.key == pygame.K_LEFT:
                        game.move_hor(-1)
                    if event.key == pygame.K_RIGHT:
                        game.move_hor(1)
                    if event.key == pygame.K_ESCAPE:
                        game.__init__(20, 10)

                    #UI updates
            update_field_UI(game)        
            update_UI(game)        

        else:
            #done = display_game_over(game)
            print(game.score)

        pygame.display.flip()
        clock.tick(fps)

#send the informations to the minizinc solver
def solve(return_storing, game, solver, model):
    
    temp_blocks = deepcopy(game.blocks)
    temp_blocks.pop()
    n = len(game.blocks) - 1
    
    blocks_x = [block.x  for block in temp_blocks]
    blocks_y = [(19 - block.y - block.height + 1) for block in temp_blocks]
    blocks_w = [block.width for block in temp_blocks]
    blocks_h = [block.height for block in temp_blocks]
    widths = [game.current_block.width, game.next_block.width]
    heights = [game.current_block.height, game.next_block.height]
    
    inst = minizinc.Instance(solver, model)
    inst["n"] = n
    inst["blocks_x"] = blocks_x
    inst["blocks_y"] = blocks_y
    inst["blocks_w"] = blocks_w
    inst["blocks_h"] = blocks_h
    inst["widths"] = widths
    inst["heights"] = heights    
    
    out = inst.solve(timeout=timedelta(seconds=300), free_search=True)

    return_storing["pos_x"] = out.solution.pos_x[0]
    return_storing["rotation"] = out.solution.rotations[0]
    return_storing["pos_y"] = out.solution.pos_y[0] - 19

def start_q_table_solver(level):
    print("loading table")
    load_table()
    done = False
    clock = pygame.time.Clock()
    fps = 50
    game = Cubics(10,20)
    game.gen_new_block()
    game.gen_next_block()
    game.level = level
    counter = 0
    return_storing = {}
    
    field_obs, block_code = create_observation(game.field_no_curr, (game.current_block.width, game.current_block.height))
    thread = threading.Thread(target = q_table_solve, args = (return_storing, field_obs, block_code))
    thread.start()
    rotated = False
    checked = False       

    while not done:
        if game.state == "start":        
            #keep track of time
            counter += 1 
            if counter > 100000:
                counter = 0

            #move piece down based on level
            if counter % (fps // game.level // 2) == 0:
                if game.state == "start":
                    state = game.move_down()
                    if state:                            
                        field_obs, block_code = create_observation(game.field_no_curr, (game.current_block.width, game.current_block.height))
                        rotated = False
                        checked = False
                        thread = threading.Thread(target=q_table_solve, args=(return_storing, field_obs, block_code))
                        thread.start()
                        
        
            #event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if not thread.is_alive() and not checked: 
                x = return_storing["action"][0]
                rotate = return_storing["action"][1]
                if rotate and not rotated:
                    game.rotate()
                    rotated = True
                game.move_to_x(x) 
                checked=True


            #UI updates
            update_field_UI(game)        
            update_UI(game)
        else:
            done = True

        pygame.display.flip()
        clock.tick(fps)    

def start_q_learning_solver(level):
    done = False
    clock = pygame.time.Clock()
    fps = 50
    game = Cubics(10,20)
    game.gen_new_block()
    game.gen_next_block()
    game.level = level
    counter = 0
    return_storing = {}
    
    agent = Agent(hidden_layers=2)
    agent.load_state_dict(torch.load("target.pt"))
    action_index = agent.act(
        game.get_normalized_field().flatten(), 
        REVERSE_BLOCK_MAP[BLOCK_MAP[(game.current_block.width, game.current_block.height)]])
        
    action = ACTION_MAP[action_index]
    rotated = False
    checked = False       

    while not done:
        if game.state == "start":        
            #keep track of time
            counter += 1 
            if counter > 100000:
                counter = 0

            #move piece down based on level
            if counter % (fps // game.level // 2) == 0:
                if game.state == "start":
                    state = game.move_down()
                    if state:              
                                     
                        action_index = agent.act(
                            game.get_normalized_field(), 
                            REVERSE_BLOCK_MAP[BLOCK_MAP[(game.current_block.width, game.current_block.height)]])
                            
                        checked = False
                        rotated = False
        
            #event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if  not checked: 
                x = action[0]
                rotate = action[1]
                if rotate and not rotated:
                    game.rotate()
                    rotated = True
                game.move_to_x(x) 
                checked=True


            #UI updates
            update_field_UI(game)        
            update_UI(game)
        else:
            done = True

        pygame.display.flip()
        clock.tick(fps)    

#automatic computer playing
def start_game_solver(counter_inn, level):
    with open("data.csv", mode='a') as f:
        done = False
        clock = pygame.time.Clock()
        fps = 25
        game = Cubics(10,20)
        game.gen_new_block()
        game.gen_next_block()
        game.level = level
        counter = 0
        return_storing = {}

        model_path = os.path.join(
            os.path.dirname(__file__),
            "SOLVER.mzn"
        )
        model = minizinc.Model(model_path)
        solver = minizinc.Solver.lookup('chuffed')    
        
        #creates the main thread for solving and start it
        thread = threading.Thread(target = solve, args=(return_storing, game, solver, model))
        thread.start()
        while not done:
            if game.state == "start":        
                #keep track of time
                counter += 1 
                if counter > 100000:
                    counter = 0

                #move piece down based on level
                if counter % (fps // game.level // 2) == 0:
                    if game.state == "start":
                        game.move_down() 
            
                #event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

                if not thread.is_alive(): 
                    x = return_storing["pos_x"]
                    rotate = return_storing["rotation"]
                    if rotate:
                        game.rotate()
                    y = abs(return_storing["pos_y"]) - game.current_block.height + 1
                    game.move_to(x,y) 

                    #update view

                    thread = threading.Thread(target=solve, args=(return_storing, game, solver, model))
                    thread.start()

                #UI updates
                update_field_UI(game)        
                update_UI(game)        

            else:
                #display_game_over(game)
                print(game.score)
                if counter_inn+1 < 10 and level <= 4:
                    f.write("{}, {} \n".format(game.level, game.score))
                    f.flush()
                    start_game_solver(counter_inn+1)
                elif counter_inn+1 == 10 and level <= 4:
                    f.write("{}, {} \n".format(game.level, game.score))
                    f.flush()
                    start_game_solver(0, game.level+1)
                else:
                    pygame.quit()

            pygame.display.flip()
            clock.tick(fps)

run = True
while run:    
    
    screen.fill("#2C3333")

    title = font_bigger.render("Cubics", True, "#A5C9CA")
    screen.blit(title, [width//2 - title.get_bounding_rect()[2]//2, 50])
    
    button1 = Button(
        pygame.Rect(width//2 - 125, 250, 250, 70),
        "Play", 
        start_game
    )
    button1.on_hover()

    button2 = Button(                                  #q_table
        pygame.Rect(width//2 - 125, 370, 250, 70), 
        "Computer",
        start_q_table_solver,
        kwargs={"level": 10}
    )
    # button2 = Button(                                 #deep q learning
    #     pygame.Rect(width//2 - 125, 370, 250, 70), 
    #     "Computer",
    #     start_q_learning_solver,
    #     kwargs={"level": 10}
    # )
    button2.on_hover()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        button1.on_click(event)
        button2.on_click(event)

    button1.show()
    button2.show()

    bottom_page = font_small.render("Lorenzo Tribuiani | Matteo Rossi Reich - 2022", True, "#A5C9CA")
    screen.blit(bottom_page, [width//2 - bottom_page.get_bounding_rect()[2]//2, 600])
    pygame.display.update()
pygame.quit()