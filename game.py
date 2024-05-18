import pygame
import sys
import random
from snake import Snake

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.WIDTH, self.HEIGHT = 800, 600
        self.SNAKE_SIZE = 20
        self.SNAKE_SPEED = self.SNAKE_SIZE  # make sure the snake's speed is the same as the grid cell size
        self.snake = Snake([100//self.SNAKE_SIZE*self.SNAKE_SIZE, 50//self.SNAKE_SIZE*self.SNAKE_SIZE], [self.SNAKE_SPEED, 0], self.SNAKE_SIZE)
        self.generate_food()
        self.score = 0
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

    def game_over(self):
        pygame.quit()
        sys.exit()

    def get_direction_from_input(self, keys):
        if keys[pygame.K_UP]:
            return [0, -self.SNAKE_SPEED]
        elif keys[pygame.K_DOWN]:
            return [0, self.SNAKE_SPEED]
        elif keys[pygame.K_LEFT]:
            return [-self.SNAKE_SPEED, 0]
        elif keys[pygame.K_RIGHT]:
            return [self.SNAKE_SPEED, 0]
        else:
            return self.snake.direction.get()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over()
                keys = pygame.key.get_pressed()
            direction = self.get_direction_from_input(keys)
            self.snake.move(direction)

            if self.snake.get_head_pos() == self.food_pos:
                self.score += 1
                self.food_pos = [random.randrange(1, self.WIDTH//self.SNAKE_SIZE) * self.SNAKE_SIZE, random.randrange(1, self.HEIGHT//self.SNAKE_SIZE) * self.SNAKE_SIZE]
            else:
                self.snake.shrink()

            head_pos = self.snake.get_head_pos()
            if head_pos[0] < 0 or head_pos[0] > self.WIDTH-20 or head_pos[1] < 0 or head_pos[1] > self.HEIGHT-20:
                self.game_over()
            if self.snake.collides_with_self():
                self.game_over()
            if self.snake.get_head_pos() == self.food_pos:
                self.score += 1
                self.generate_food()

            self.screen.fill(pygame.Color(0, 0, 0))

            for body_part in self.snake.get_body():
                pygame.draw.rect(self.screen, pygame.Color(0,255,0), pygame.Rect(*body_part))

            pygame.draw.rect(self.screen, pygame.Color(255,0,0), pygame.Rect(self.food_pos[0], self.food_pos[1], self.SNAKE_SIZE, self.SNAKE_SIZE))

            pygame.display.flip()

            self.clock.tick(60)

    def generate_food(self):
        self.food_pos = [random.randrange(self.WIDTH//self.SNAKE_SIZE) * self.SNAKE_SIZE, random.randrange(self.HEIGHT//self.SNAKE_SIZE) * self.SNAKE_SIZE]

if __name__ == "__main__":
    game = SnakeGame()
    game.run()