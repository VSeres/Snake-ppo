from typing import Tuple
from numpy.core.fromnumeric import size
import pygame, sys, random, gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class Snake2(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, width: int=24, height: int=24, ticks=10):
        super(Snake2,self).__init__()
    
        self.POSSIBLE_ACTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN']
        self.width = width
        self.height = height
        self.thickness = 30
        self.size = width*height
        self.action_space = spaces.Discrete(len(self.POSSIBLE_ACTIONS))
        self.observation_space = spaces.Box(low=-1, high=32767, shape=(30,), dtype=np.int16)
        self.max_distance : np.ndarray = np.array([self.height-1, (min(self.height,self.width)-1)*2, self.width-1, (min(self.height,self.width)-1)*2]*2)
        self.playSurface = None
        self.ticks = ticks

        self.red = pygame.Color(255, 0, 0) # snake head
        self.green = pygame.Color(0, 255, 0) # snake body
        self.white = pygame.Color(255, 255, 255) # food
        self.black = pygame.Color(0, 0, 0) # background

        self.reset()

    def reset(self) -> np.ndarray:
        # (width, height)
        middle = (int(self.width/2), int(self.height/2))
        self.snake = [ middle, (middle[0]+1, middle[1]), (middle[0]+2, middle[1])]
        self.head_direction = np.zeros(4)
        self.head_direction[3] = 1
        self.tail_direction = np.zeros(4)
        self.tail_direction[3] = 1
        self.food = None
        self.spawn_food()
        self.game_over = False
        self.score = 0
        self.steps = int(self.size/2)
        return self.get_state()

    def init_interface(self) -> None:
        pygame.init()
        self.fps_controller = pygame.time.Clock()
        self.playSurface = pygame.display.set_mode((self.width*self.thickness, (self.height*self.thickness)+25))
        pygame.display.set_caption("Snake")

    def close(self) -> None:
        pygame.quit()
        sys.exit()

    def spawn_food(self) -> None:
        food = (random.randrange(0,self.width), random.randrange(0,self.width))
        while food in self.snake:
            food = (random.randrange(0,self.width), random.randrange(0,self.width))
        self.food = food

    def check_game_over(self) -> bool:
        # bounds
        if self.snake[0][0] < 0 or self.snake[0][0] > self.width-1:
            return True
        if self.snake[0][1] < 0 or self.snake[0][1] > self.height-1:
            return True
        # self hit
        if self.snake[0] in self.snake[1:]:
            return True

        return False

    def distance(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        distance_to_wall = []
        distance_to_self = self.max_distance.copy()
        direction_to_food = [0, 0, 0, 0, 0]
        # uppwards distance to wall
        toLT = min(head)
        toRT = min(self.width-1-head[0], head[0])
        toLB = min(head[0], self.height-1-head[1])
        toRB = min(self.width-1-head[0], self.height-1-head[1])

        distance_to_wall.append(head[1])
        distance_to_wall.append(self.distance((head[0]-toLT, head[1]-toLT), head))
        distance_to_wall.append(self.width-1-head[0])
        distance_to_wall.append(self.distance((head[0]+toRT, head[1]-toRT), head))
        distance_to_wall.append(self.height-1-head[1])
        distance_to_wall.append(self.distance((head[0]-toLB, head[1]+toLB), head))
        distance_to_wall.append(head[0])
        distance_to_wall.append(self.distance((head[0]+toRB, head[1]+toRB), head))

        for part in self.snake[1:]:
            diagonal = self.diagonal(part, head)
            distance = self.distance(head, part)
            if diagonal > 0 and distance_to_self[diagonal] > distance:
                distance_to_self[diagonal] = distance
                
            if head[0] == part[0]:
                if part[1]-head[1] < 0:
                    if distance_to_self[0] > distance:
                        distance_to_self[0] = distance
                else:
                    if distance_to_self[4] > distance:
                        distance_to_self[4] = distance
            elif head[1] == part[1]:
                if part[0]-head[0] > 0:
                    if distance_to_self[2] > distance:
                        distance_to_self[2] = distance
                else:
                    if distance_to_self[6] > distance:
                        distance_to_self[6] = distance

        food_direction = (self.food[0] - head[0], self.food[1] - head[1])

        if food_direction[0] < 0:
            direction_to_food[3] = 1
        else:
            direction_to_food[1] = 1
        if food_direction[1] < 0:
            direction_to_food[0] = 1
        else:
            direction_to_food[2] = 1
        direction_to_food[4] = self.distance(head, self.food)

        return np.concatenate((distance_to_self, distance_to_wall, direction_to_food, self.head_direction, self.tail_direction, [self.steps])).astype(np.int16)

    def is_on_food(self) -> bool:
        return self.snake[0] == self.food

    def diagonal(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int:
        if pos1[0] == pos2[0] or pos1[1] == pos2[1]:
            return -1
        difference = (pos1[0] - pos2[0], pos1[1] - pos2[1])
        absulte_value = tuple(map(abs, difference))

        if(absulte_value[0] == absulte_value[1]):
            if difference[0] > 0 and difference[1] < 0:
                return 1
            if difference[0] > 0 and difference[1] > 0:
                return 3
            if difference[0] < 0 and difference[1] > 0:
                return 5
            if difference[0] < 0 and difference[1] < 0:
                return 7
        else:
            return -1
        # 7 -> \/ <- 1
        # 5 -> /\ <- 3

    def step(self, action: int):
        self.steps -= 1
        info = {'steps_remaining': self.steps, 'score': self.score}
        if self.steps < 0:
            self.game_over = True
            return self.get_state(), -20, self.game_over, info
        x = self.snake[0][0]
        y = self.snake[0][1]
        if action == 0:
            y -= 1
        if action == 1:
            x += 1
        if action == 2:
            y += 1
        if action == 3:
            x -= 1
        
        self.head_direction = np.zeros(4)
        self.head_direction[action] = 1
        self.snake.insert(0, (x, y))

        reward = 0
        if self.check_game_over():
            self.game_over = True
            reward = -2

        elif self.is_on_food():
            self.score += 1
            self.steps = int(self.size*0.75+self.score*1.5)
            if len(self.snake) == self.size:
                reward = 10
                self.game_over = True
            else:
                reward = 1
                self.spawn_food()
        else:
            self.snake.pop()
            # reward = 1/self.distance(self.snake[0],self.food)
        
        direction_x = self.snake[-2][0]-self.snake[-1][0]
        direction_y = self.snake[-2][0]-self.snake[-1][0]
        self.tail_direction = np.zeros(4)
        if direction_y < 0:
            self.tail_direction[0] = 1
        elif direction_x > 0:
            self.tail_direction[1] = 1
        elif direction_y > 0:
            self.tail_direction[2] = 1
        else:
            self.tail_direction[3] = 1

        info['score'] = self.score
        return self.get_state(), reward, self.game_over, info

    def draw(self) -> None:
        self.playSurface.fill(self.black)
        score_font = pygame.font.SysFont('consolas', 20)
        score_surface = score_font.render(f'Pontok: {self.score}',True, self.white)
        self.playSurface.blit(score_surface, (5,5))
        self.draw_rect(self.food, self.white)
        for pos in self.snake[1:]:
            self.draw_rect(pos, self.green)
        self.draw_rect(self.snake[0], self.red)
    
    def render(self) -> None:
        if self.playSurface is None:
            self.init_interface()
        if len(self.snake) == self.width*self.height:
            print('BINGO!!')
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.draw()
        pygame.display.update()
        self.fps_controller.tick(self.ticks)
        pygame.display.set_caption(f'Snake - {self.fps_controller.get_fps():.0f} fps')
        if self.game_over:
            pygame.time.delay(2000)

    def draw_rect(self, pos, color) -> None:
        pos = list(pos)
        pos[0] = max(pos[0], 0)
        pos[0] = min(pos[0], self.width-1)
        pos[1] = max(pos[1], 0)
        pos[1] = min(pos[1], self.height-1)
        pygame.draw.rect(self.playSurface, color, pygame.Rect((pos[0]*self.thickness)+5, (pos[1]*self.thickness)+25, self.thickness-5, self.thickness-5))

def main():
    import main
    
    game = Snake2(9,9,ticks=100)
    model = PPO.load('./model/main')
    for _ in range(3):
        done = False
        obs = game.reset() 
        while not done:
            prev_obs = obs
            action, _state = model.predict(obs)
            obs, _reward, done, info = game.step(action=action)
            game.render()
        print(info)
    return None
if __name__ == '__main__':
    main()