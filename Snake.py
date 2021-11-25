from typing import Tuple
import pygame, sys, random, gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import colorsys

UP = 0
RIGHTUP = 1
RIGHT = 2
RIGHTDOWN = 3
DOWN = 4
LEFTDOWN = 5
LEFT = 6
LEFTUP = 7

class Snake2(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, size: int = 6, ticks : int = 10):
        super(Snake2,self).__init__()
    
        self.POSSIBLE_ACTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN']
        self.thickness = 30
        self.dimension = size
        self.map_size = size**2
        self.action_space = spaces.Discrete(len(self.POSSIBLE_ACTIONS))
        self.observation_space = spaces.Box(low=-1, high=32767, shape=(30,), dtype=np.int16)
        self.max_distance : np.ndarray = np.array([self.dimension -1, (min(self.dimension ,self.dimension )-1)*2, self.dimension -1, (min(self.dimension ,self.dimension )-1)*2]*2)
        self.playSurface = None
        self.ticks = ticks

        self.colors = {
            "head": pygame.Color(255, 0, 0),
            "body": pygame.Color(0, 255, 0),
            "food": pygame.Color(255, 255, 255),
            "wall": pygame.Color(255, 255, 255),
            "text":pygame.Color(255, 255, 255),
            "background": pygame.Color(0, 0, 0)
        }

        self.reset()

    def random_colors(self) -> None:
        rgb = [random.randrange(0,256)/255 for _ in range(3)]
        h, l, s = colorsys.rgb_to_hls(*rgb)
        angle_change = 360/4
        angle = 0
        for key in self.colors.keys():
            if key == 'background' or key == 'text': continue
            hue = h + angle / 360
            angle += angle_change
            color = [round(x*255) for x in colorsys.hls_to_rgb(hue, l, s)]
            self.colors[key] = pygame.Color(*color)

    def reset(self) -> np.ndarray:
        # (width, height)
        middle = (int(self.dimension /2), int(self.dimension /2))
        self.snake = [ middle, (middle[0]+1, middle[1]), (middle[0]+2, middle[1])]
        self.head_direction = np.zeros(4)
        self.head_direction[3] = 1
        self.tail_direction = np.zeros(4)
        self.tail_direction[3] = 1
        self.food = None
        self.spawn_food()
        self.game_over = False
        self.score = 0
        self.steps = self.map_size*1.5
        self.step_count = 0
        return self.get_state()

    def init_interface(self) -> None:
        pygame.init()
        self.fps_controller = pygame.time.Clock()
        self.playSurface = pygame.display.set_mode((self.dimension *self.thickness+60, self.dimension *self.thickness+80))
        pygame.display.set_caption("Snake")

    def close(self) -> None:
        pygame.quit()
        sys.exit()

    def spawn_food(self) -> None:
        food = (random.randrange(0,self.dimension ), random.randrange(0,self.dimension ))
        while food in self.snake:
            food = (random.randrange(0,self.dimension ), random.randrange(0,self.dimension ))
        self.food = food

    def check_game_over(self) -> bool:
        # bounds
        if self.snake[0][0] < 0 or self.snake[0][0] > self.dimension -1:
            return True
        if self.snake[0][1] < 0 or self.snake[0][1] > self.dimension -1:
            return True
        # self hit
        if self.snake[0] in self.snake[1:]:
            return True

        return False

    def distance(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        head_x = head[0]
        head_y = head[1]
        distance_to_wall = np.ndarray((8,), dtype=np.int16)
        distance_to_self = self.max_distance.copy()
        direction_to_food = np.zeros((5,), dtype=np.int16)
        # uppwards distance to wall
        toLT = min(head)
        toRT = min(self.dimension -1-head_x, head_x)
        toLB = min(head_x, self.dimension -1-head_y)
        toRB = min(self.dimension -1-head_x, self.dimension -1-head_y)

        distance_to_wall[UP] = head_y
        distance_to_wall[RIGHTUP] = self.distance((head_x-toLT, head_y-toLT), head)
        distance_to_wall[RIGHT] = self.dimension -1-head_x
        distance_to_wall[RIGHTDOWN] = self.distance((head_x+toRT, head_y-toRT), head)
        distance_to_wall[DOWN] = self.dimension -1-head_y
        distance_to_wall[LEFTDOWN] = self.distance((head_x-toLB, head_y+toLB), head)
        distance_to_wall[LEFT] = head_x
        distance_to_wall[LEFTUP] = self.distance((head_x+toRB, head_y+toRB), head)

        for part in self.snake[1:]:
            diagonal = self.diagonal(part, head)
            distance = self.distance(head, part)
            part_x = part[0]
            part_y = part[1]
            if diagonal > 0 and distance_to_self[diagonal] > distance:
                distance_to_self[diagonal] = distance
                continue

            if head_x == part_x:
                if part_y-head_y < 0:
                    if distance_to_self[UP] > distance:
                        distance_to_self[UP] = distance
                else:
                    if distance_to_self[DOWN] > distance:
                        distance_to_self[DOWN] = distance
            elif head_y == part_y:
                if part_x-head_x > 0:
                    if distance_to_self[RIGHT] > distance:
                        distance_to_self[RIGHT] = distance
                else:
                    if distance_to_self[LEFT] > distance:
                        distance_to_self[LEFT] = distance

        food_direction = (self.food[0] - head_x, self.food[1] - head_y)

        if food_direction[0] < 0:
            direction_to_food[3] = 1
        else:
            direction_to_food[1] = 1
        if food_direction[1] < 0:
            direction_to_food[0] = 1
        else:
            direction_to_food[2] = 1
        direction_to_food[4] = self.distance(head, self.food)

        return np.concatenate((distance_to_self, distance_to_wall, direction_to_food, self.head_direction, self.tail_direction, [self.score])).astype(np.int16)

    def is_on_food(self) -> bool:
        return self.snake[0] == self.food

    def diagonal(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int:
        if pos1[0] == pos2[0] or pos1[1] == pos2[1]:
            return -1
        difference = (pos1[0] - pos2[0], pos1[1] - pos2[1])
        absulte_value = (abs(difference[0]), abs(difference[1]))

        if(absulte_value[0] == absulte_value[1]):
            if difference[0] > 0 and difference[1] < 0:
                return RIGHTUP
            if difference[0] > 0 and difference[1] > 0:
                return RIGHTDOWN
            if difference[0] < 0 and difference[1] > 0:
                return LEFTDOWN
            if difference[0] < 0 and difference[1] < 0:
                return LEFTUP
        else:
            return -1
        # 7 -> \/ <- 1
        # 5 -> /\ <- 3

    def step(self, action: int):
        self.steps -= 1
        info = {'steps_remaining': self.steps, 'score': self.score, 'step_count': self.step_count}
        if self.steps < 0:
            self.game_over = True
            return self.get_state(), -20, self.game_over, info
        self.step_count += 1
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
            reward = -4

        elif self.is_on_food():
            self.score += 1
            self.steps = int(self.map_size*2+self.score)
            if len(self.snake) == self.map_size:
                reward = (2*self.map_size**2) / self.step_count
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
        info['step_count'] = self.step_count
        return self.get_state(), reward, self.game_over, info

    def draw(self) -> None:
        self.playSurface.fill(self.colors["background"])
        score_font = pygame.font.SysFont('consolas', 20)
        score_surface = score_font.render(f'Pontok: {self.score}',True, self.colors["text"])
        self.playSurface.blit(score_surface, (5,5))
        self.draw_rect(self.food, self.colors["food"])

        # border top
        pygame.draw.rect(self.playSurface, self.colors['wall'], pygame.Rect(25, 25, self.thickness*(self.dimension +2)-50, 25))
        # border bottom
        pygame.draw.rect(self.playSurface, self.colors['wall'], pygame.Rect(25, (self.dimension +1)*self.thickness+25, self.thickness*(self.dimension +2)-50, 25))
        # border left
        pygame.draw.rect(self.playSurface, self.colors['wall'], pygame.Rect(0, 25, 30, (self.dimension +2)*self.thickness))
        # border right
        pygame.draw.rect(self.playSurface, self.colors['wall'], pygame.Rect(self.thickness*(self.dimension +2)-25, 25, 25, (self.dimension +2)*self.thickness))

        for pos in self.snake[1:]:
            self.draw_rect(pos, self.colors["body"])
        self.draw_rect(self.snake[0], self.colors["head"])
    
    def render(self) -> None:
        if self.playSurface is None:
            self.init_interface()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.draw()
        pygame.display.update()
        self.fps_controller.tick(self.ticks)
        pygame.display.set_caption(f'Snake - {self.fps_controller.get_fps():.0f} fps')
        if self.game_over:
            pygame.time.delay(2000)

    def draw_rect(self, pos: Tuple[int, int], color: pygame.Color) -> None:
        pos = list(pos)
        pos[0] += 1
        pos[1] += 1
        pygame.draw.rect(self.playSurface, color, pygame.Rect((pos[0]*self.thickness)+5, (pos[1]*self.thickness)+25, self.thickness-5, self.thickness-5))

def main():
    game = Snake2(9, ticks=50)
    model = PPO.load('./model/main16-16')
    for _ in range(3):
        game.random_colors()
        done = False
        obs = game.reset() 
        while not done:
            action, _state = model.predict(obs)
            obs, _reward, done, info = game.step(action=action)
            game.render()
    return None
if __name__ == '__main__':
    main()