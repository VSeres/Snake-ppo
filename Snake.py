from typing import Any, Tuple, Union
import pygame
import sys
import random
import gym
from gym import spaces
import numpy as np
import operator
import colorsys

from player import Agent, HumanAgent, PPOAgent

UP = 0
RIGHTUP = 1
RIGHT = 2
RIGHTDOWN = 3
DOWN = 4
LEFTDOWN = 5
LEFT = 6
LEFTUP = 7


class Snake2(gym.Env):
    """
    Snake játék megvalósítása.
    Ez van felhasználva a tanuláshoz és a játék futtatásához.
    OpenAi gym környezetet kiterjeszti

    :ivar size: pályamérete, csak a tanitáshoz szükséges
    :ivar thickness: blokkok mérete
    :ivar dimension: a pálya mérete
    :ivar colors: Szinek a különbözö elemekhez
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, size: int = None):
        super(Snake2, self).__init__()

        self.POSSIBLE_ACTIONS = ['RIGHT', 'LEFT', 'UP', 'DOWN']
        self.thickness = 30
        self.dimension = size or 6
        self.action_space = spaces.Discrete(len(self.POSSIBLE_ACTIONS))
        self.observation_space = spaces.Box(
            low=-1, high=32767, shape=(29,), dtype=np.int16)
        self._set_game_variabels()
        self.screen = None
        self.colors = {
            "head": pygame.Color(255, 0, 0),
            "body": pygame.Color(0, 255, 0),
            "food": pygame.Color(255, 255, 255),
            "wall": pygame.Color(255, 255, 255),
            "text": pygame.Color(255, 255, 255),
            "background": pygame.Color(0, 0, 0)
        }

        self.reset()

    def _set_game_variabels(self):
        """
        Beállítja a pálya méretét és a maximum távolságokat
        """
        self.map_size = self.dimension**2
        self.max_distance: np.ndarray = np.array(
            [self.dimension - 1, (min(self.dimension, self.dimension)-1)*2, self.dimension - 1, (min(self.dimension, self.dimension)-1)*2]*2)

    def random_colors(self):
        """
        Véletlenszerű szinek generálása a pályához
        """
        rgb = [random.randrange(0, 256)/255 for _ in range(3)]
        h, l, s = colorsys.rgb_to_hls(*rgb)
        angle_change = 360/4
        angle = 0
        for key in self.colors.keys():
            if key == 'background' or key == 'text':
                continue
            hue = h + angle / 360
            angle += angle_change
            color = [round(x*255) for x in colorsys.hls_to_rgb(hue, l, s)]
            self.colors[key] = pygame.Color(*color)

    def reset(self) -> np.ndarray:
        """
        Visszaállítja a környezetet a kiindulási állapotba

        :return: a környezet állapotát
        """
        # (width, height)
        middle = (int(self.dimension / 2), int(self.dimension / 2))
        self.snake = [middle, (middle[0]+1, middle[1]),
                      (middle[0]+2, middle[1])]
        self.food = None
        self.spawn_food()
        self.game_over = False
        self.score = 0
        self.steps = self.map_size*1.5
        self.step_count = 0
        return self.get_state()

    def init_interface(self):
        """
        A pygame ablakhoz szükséges beállítások
        """
        pygame.init()
        Button.env = self
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((500, 500))
        pygame.display.set_caption("Snake")

    def close(self):
        """
        A programbol való kilépés
        Automatikusan meg van hiva amikor a környezet garbage collected
        """
        pygame.quit()
        sys.exit()

    def spawn_food(self):
        """
        Lehelyez a pályán egy új ennivalót
        """
        food = (random.randrange(0, self.dimension),
                random.randrange(0, self.dimension))
        while food in self.snake:
            food = (random.randrange(0, self.dimension),
                    random.randrange(0, self.dimension))
        self.food = food

    def check_game_over(self) -> bool:
        """
        Le ellenörzi hogy a kigyó feje kint van a pályáról vagy nekiment a kigyo testének

        :return: neki ment e valaminek
        """
        # bounds
        if self.snake[0][0] < 0 or self.snake[0][0] > self.dimension - 1:
            return True
        if self.snake[0][1] < 0 or self.snake[0][1] > self.dimension - 1:
            return True
        # self hit
        if self.snake[0] in self.snake[1:]:
            return True

        return False

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Visszaadja két pont manhattan távolságát

        :param pos1: első pont
        :param pos2: második pont

        :return: két pont közötti távolság
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_state(self) -> np.ndarray:
        """
        Kiszámolja az agentnek a pálya állapotát

        :return: pálya állapota
            8 irányba a távolságot magátol és a pálya szélétől 
            merre van és milyen mesze az enivaló
            a kinhó fejének és farkának haladási iránya

        """
        head = self.snake[0]
        head_x = head[0]
        head_y = head[1]
        distance_to_wall = np.ndarray((8,), dtype=np.int16)
        distance_to_self = self.max_distance.copy()
        direction_to_food = np.zeros((5,), dtype=np.int16)

        # distance to wall
        toLT = min(head)
        toRT = min(self.dimension - 1-head_x, head_x)
        toLB = min(head_x, self.dimension - 1-head_y)
        toRB = min(self.dimension - 1-head_x, self.dimension - 1-head_y)

        distance_to_wall[UP] = head_y
        distance_to_wall[RIGHTUP] = self.distance(
            (head_x-toLT, head_y-toLT), head)
        distance_to_wall[RIGHT] = self.dimension - 1-head_x
        distance_to_wall[RIGHTDOWN] = self.distance(
            (head_x+toRT, head_y-toRT), head)
        distance_to_wall[DOWN] = self.dimension - 1-head_y
        distance_to_wall[LEFTDOWN] = self.distance(
            (head_x-toLB, head_y+toLB), head)
        distance_to_wall[LEFT] = head_x
        distance_to_wall[LEFTUP] = self.distance(
            (head_x+toRB, head_y+toRB), head)

        # distance to body
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

        # food direction
        if food_direction[0] < 0:
            direction_to_food[3] = 1
        else:
            direction_to_food[1] = 1
        if food_direction[1] < 0:
            direction_to_food[0] = 1
        else:
            direction_to_food[2] = 1
        direction_to_food[4] = self.distance(head, self.food)

        # tail direction
        tail_direction = Snake2.direction(self.snake[-1], self.snake[-2])
        # head direction
        head_direction = Snake2.direction(self.snake[1], head)
        return np.concatenate((distance_to_self, distance_to_wall, direction_to_food, head_direction, tail_direction)).astype(np.int16)

    def is_on_food(self) -> bool:
        """:return: a kigyó feje ennivalón van-e"""
        return self.snake[0] == self.food

    def diagonal(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Kiszámolja hogy meilyik átlón van az egyok ponthoz képes a másik

        :return: -1 ha nem átolósak a pontok, 1 jobbra fel, 3 jobbra le, 5 balra le, 7 balra fel

        """
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

    def step(self, action: int) -> Union[tuple[np.ndarray, float, bool, dict[str, Any]], None]:
        """
        Egy lépés hajt végre a környezeten
        Ha a kör véget ért te felelsz a reset() meghivásáért

        :param action: az agent által biztosított müvelet

        :returns: observation (ndarray): A pálya állapota, reward (float): A műveletér járo jutalom, done (bool): A kör végetért-e, info (dict): egyéb adatokat tratalmaz
        """
        self.steps -= 1
        info = {'steps_remaining': self.steps,
                'score': self.score, 'step_count': self.step_count}
        if self.steps < 0:
            self.game_over = True
            return self.get_state(), -20, self.game_over, info
        self.step_count += 1
        x = self.snake[0][0]
        y = self.snake[0][1]
        if action == 0:
            y -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y += 1
        elif action == 3:
            x -= 1

        self.snake.insert(0, (x, y))

        reward = 0
        if self.check_game_over():
            self.game_over = True
            reward = -4
            self.snake.pop()

        elif self.is_on_food():
            self.score += 1
            self.steps = int(self.map_size*2+self.score)
            if len(self.snake) == self.map_size:
                reward = int(round((1.6*self.map_size**2) / self.step_count))
                reward = min(reward, 4)
                self.game_over = True
            else:
                reward = 1
                self.spawn_food()
        else:
            self.snake.pop()
            # reward = 1/self.distance(self.snake[0],self.food)

        info['score'] = self.score
        info['step_count'] = self.step_count
        return self.get_state(), reward, self.game_over, info

    def draw_text(self, text, size, x, y):
        """
        Szöveg kiírása a képernyőre

        :param text: kiirandó szöveg
        :param size: kiirandó szöveg betűmérete
        :param x: szöveg középpontjának X pozíciója
        :param y: szöveg középpontjának Y pozíciója
        """
        font = pygame.font.SysFont('consolas', size)
        text_surface: pygame.Surface = font.render(
            text, True, self.colors['text'])
        text_rect = text_surface.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(text_surface, text_rect)

    def direction(pos1: tuple[int, int], pos2: tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Visszaadja a pos1 key képes melyik irányba van a pos2

        :param pos1: x, y koordinátája a referencia pontnak
        :param pos2: x, y koordinátája a vizsgálandó pontnak

        :return: Az első eleme a felfelé, óramutató irányba haladnak az irányok
        """
        dir_x = pos2[0] - pos1[0]
        dir_y = pos2[1] - pos1[1]
        direction = [0, 0, 0, 0]
        if dir_y < 0:
            direction[0] = 1
        elif dir_y > 0:
            direction[2] = 1
        if dir_x > 0:
            direction[1] = 1
        elif dir_x < 0:
            direction[3] = 1
        return tuple(direction)

    def render(self):
        """
        A pályát kirajzolja a képernyőre
        """
        self.screen.fill(self.colors["background"])
        centerx = self.screen.get_size()[0]
        self.draw_text(
            f'Score: {self.score}   Speed: {self.ticks}', 20, centerx/2, 15)
        self.draw_rect(self.food, self.colors["food"])

        # border top
        pygame.draw.rect(self.screen, self.colors['wall'], pygame.Rect(
            25, 25, self.thickness*(self.dimension + 2)-50, 25))
        # border bottom
        pygame.draw.rect(self.screen, self.colors['wall'], pygame.Rect(
            25, (self.dimension + 1)*self.thickness+25, self.thickness*(self.dimension + 2)-50, 30))
        # border left
        pygame.draw.rect(self.screen, self.colors['wall'], pygame.Rect(
            0, 25, 30, (self.dimension + 2)*self.thickness))
        # border right
        pygame.draw.rect(self.screen, self.colors['wall'], pygame.Rect(
            self.thickness*(self.dimension + 2)-25, 25, 25, (self.dimension + 2)*self.thickness))

        line = {'direction': None, 'length': 0, 'start': None}
        for i, pos in enumerate(self.snake[1:]):
            pos_direction = Snake2.direction(self.snake[i], pos)
            direction = pos_direction.index(1)
            if line['direction'] == direction:
                line['length'] += 1
            else:
                self.draw_line(
                    self.colors["body"], line['start'], line['length'], line['direction'])
                line['direction'] = direction
                line['length'] = 1
                line['start'] = pos
        self.draw_line(self.colors["body"], line['start'],
                       line['length'], line['direction'])
        self.draw_rect(self.snake[0], self.colors["head"])
        pygame.display.update()
        pygame.display.set_caption(
            f'Snake - {self.clock.get_fps():.0f} ticks')

    def draw_rect(self, pos: Tuple[int, int], color: pygame.Color):
        """
        Egy blokk kirajzolása szolgál

        :param pos: a jobb felső pontja a bloknak
        :param color: a blokk szine
        """
        pos = list(pos)
        pos[0] += 1
        pos[1] += 1
        pygame.draw.rect(self.screen, color, pygame.Rect(
            (pos[0]*self.thickness)+5, (pos[1]*self.thickness)+25, self.thickness-5, self.thickness-5))

    def draw_line(self, color: pygame.Color, start_pos, length, direction):
        """
        Kigyó testét alkotó vonalakat rajozolja ki

        :param color: A megjeleniteshez használandó szín
        :param start_pos: A vonal kiindulópontja
        :param length: Vonal hossza
        :param direction: A start_pos hoz képest merre húza a vonalat
          0 felfelé,
          1 jobbra,
          2 lefelé,
          3 balra 
        """
        if start_pos is None:
            return
        start_pos = list(start_pos)
        start_pos[0] += 1
        start_pos[1] += 1
        if direction == 0:
            start_pos[1] -= length-1
            start = ((start_pos[0]*self.thickness)+5,
                     (start_pos[1]*self.thickness)+25)
            end = self.thickness - 5, length*self.thickness
        elif direction == 1:
            start = ((start_pos[0]*self.thickness),
                     (start_pos[1]*self.thickness)+25)
            end = length*self.thickness, self.thickness-5
        elif direction == 2:
            start = ((start_pos[0]*self.thickness)+5,
                     (start_pos[1]*self.thickness)+20)
            end = self.thickness - 5, length*self.thickness
        elif direction == 3:
            start_pos[0] -= length-1
            start = ((start_pos[0]*self.thickness)+5,
                     (start_pos[1]*self.thickness)+25)
            end = length*self.thickness, self.thickness-5

        pygame.draw.rect(self.screen, color, pygame.Rect(start, end))

    def delay(self, delay: float):
        """
        x másodpercig vár az. Feldolgoza a bezárási eventet.

        :param delay: idő másodperben
        """
        end = pygame.time.get_ticks() + delay * 1000
        while end > pygame.time.get_ticks():
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

    def game_loop(self, agent: Agent, random_color: bool = True) -> Tuple[bool, int]:
        """
        A játék fó ciklusa ez csak akkor érhetó el ha play() el inditjuk ell a pogramot

        :param agent: az agent amelyik játszik
        :param random_color: véletlenszerű szinek

        :return: az agent megnyerte a kört, elért pontszám
        """
        if random_color:
            self.random_colors()
        self.screen = pygame.display.set_mode(
            (self.dimension * self.thickness+60, self.dimension * self.thickness+80))
        obs = self.reset()
        self.render()
        self._count_down()
        while not self.game_over:
            action = agent.step(obs)
            obs, _reward, _done, info = self.step(action)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP_PLUS:
                        self.ticks += 1
                    if event.key == pygame.K_KP_MINUS:
                        self.ticks -= 1
                        if self.ticks <= 0:
                            self.ticks = 1
            self.render()
            self.clock.tick(self.ticks)
        self._end_screen()
        return len(self.snake) == self.map_size, info['score'],

    def _end_screen(self):
        """
        A kör végén a pontok megjelenitése
        """
        text_size = 40 if self.dimension == 6 else 60
        center = list(self.screen.get_rect().center)
        self.draw_text('GAME OVER', text_size, *center)
        center[1] += 30 if self.dimension == 6 else 50
        self.draw_text(f'Score: {self.score}', text_size-20, *center)
        self.delay(3)

    def _count_down(self):
        """
        Kör kezdete előtti visszaszámlálás
        """
        for i in range(3, 0, -1):
            self.draw_text(str(i), 80, *self.screen.get_rect().center)
            self.delay(1)
            self.render()

    def play(self, model: str = '16x16-8'):
        """
        Ezel a metódussal lehet elindítani a játékot
        megjeleníti a főmenüt

        :param agent: az AI által használandó model
        """
        self.init_interface()
        self.diff_index = 1
        self.map_index = 0
        self.player_index = 0
        click = False
        start_button = Button('Start game', 40, 50, 50, 400, 175)
        quit_button = Button('Exit', 40, 50, 275, 400, 175)
        while True:
            self.screen.fill(self.colors['background'])

            if start_button.hover():
                if click:
                    self.settings(model)
                    self.screen = pygame.display.set_mode((500, 500))
                    click = False
            if quit_button.hover():
                if click:
                    self.close()

            start_button.draw()
            quit_button.draw()

            click = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        click = True

            pygame.display.update()
            pygame.display.set_caption(
                f'Snake - {self.clock.get_fps():.0f} ticks')
            self.clock.tick(60)

    def settings(self, model: str):
        """
        A kör elinditása előtti menü megjelenítése
        Itt lehet beállítani a játékos, pálya méretét és a nehézséget

        :param agent: az AI által használandó model
        """
        player = [PPOAgent, HumanAgent]
        map_size = [6, 9, 12, 16]
        difficulty = [2, 5, 8]
        diff_name = ['Easy', 'Medium', 'Hard']
        click = False
        player_button = Button(
            f'Player: {player[self.player_index].name}', 30, 50, 20, 400, 100)
        size_button = Button(
            f'Map size: {map_size[self.map_index]}', 30, 50, 140, 400, 100)
        difficulty_button = Button(
            f'Difficulty: {diff_name[self.diff_index]}', 30, 50, 260, 400, 100)
        start_button = Button('Start game', 30, 50, 380, 400, 100)
        while True:
            self.screen.fill(self.colors['background'])

            if player_button.hover():
                if click:
                    self.player_index = (self.player_index+1) % len(player)
                    player_button.text = f'Player: {player[self.player_index].name}'
            if size_button.hover():
                if click:
                    self.map_index = (self.map_index+1) % len(map_size)
                    size_button.text = f'Map size: {map_size[self.map_index]}'

            if start_button.hover():
                if click:
                    self.dimension = map_size[self.map_index]
                    model = player[self.player_index](model)
                    self._set_game_variabels()
                    self.ticks = difficulty[self.diff_index]
                    self.game_loop(model)
                    return
            if difficulty_button.hover():
                if click:
                    self.diff_index = (self.diff_index+1) % len(difficulty)
                    difficulty_button.text = f'Difficulty: {diff_name[self.diff_index]}'

            player_button.draw()
            size_button.draw()
            difficulty_button.draw()
            start_button.draw()

            click = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        click = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        return

            pygame.display.update()
            pygame.display.set_caption(
                f'Snake - {self.clock.get_fps():.0f} ticks')
            self.clock.tick(60)


class Button(pygame.Rect):
    """
    Gombokat a megvalósito osztály

    :cvar env: jelenlegi Snake instance
    :ivar text: gombon szereplő szöveg
    :ivar font_size: szöveg betűméret
    """
    env: Snake2

    def __init__(self, text: str, font_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = (200, 50, 60)
        self.text = text
        self.font_size = font_size

    def hover(self) -> bool:
        """
        Visszaadja hogy az egér ezen gombon van
        ha ezen a gombon van az egér átszinezi

        :return: ezen gombon van-e az egér
        """
        mouse = pygame.mouse.get_pos()
        if self.collidepoint(mouse):
            self.color = (182, 46, 46)
            return True
        self.color = (200, 50, 60)
        return False

    def draw(self):
        """
        Kirajzolja a gombot0 
        """
        pygame.draw.rect(self.env.screen, self.color, self)
        Button.env.draw_text(self.text, self.font_size, *self.center)


def main():
    game = Snake2()
    game.play(model='20x16x8')


if __name__ == '__main__':
    main()
