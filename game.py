import numpy as np
import pygame
import sys
import random
from snake import Snake
from ga_brain import GABrain

class SnakeGame:
    def __init__(self, brain=None, width=800, height=600, snake_size=20, display=True):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.WIDTH, self.HEIGHT = width, height
        self.SNAKE_SIZE = snake_size
        self.SNAKE_SPEED = self.SNAKE_SIZE  # make sure the snake's speed is the same as the grid cell size
        self.snake = Snake([100 // self.SNAKE_SIZE * self.SNAKE_SIZE, 50 // self.SNAKE_SIZE * self.SNAKE_SIZE],
                           [self.SNAKE_SPEED, 0], self.SNAKE_SIZE)
        self.generate_food()
        if brain is not None:
            self.brain = brain
        else:
            self.brain = GABrain(input_nodes=25, hidden_nodes=16, output_nodes=4, hidden_layers=2)
        self.score = 0
        self.display = display
        if self.display:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

    def game_over(self):
        print("Game Over!")
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

    def get_state(self):
        # Get the direction of the snake (a single value)
        direction = self.snake.direction.get()

        # Get the vision of the snake
        vision = self.look()

        # Flatten the vision list
        vision_flat = [item for sublist in vision for item in sublist]

        # Ensure direction is a single value
        if isinstance(direction, list):
            direction = direction[0]

        # Return direction as a single value followed by vision
        return [direction] + vision_flat

    def map_direction_to_velocity(self, direction):
        if direction == 0:
            return [0, -self.SNAKE_SPEED]
        elif direction == 1:
            return [0, self.SNAKE_SPEED]
        elif direction == 2:
            return [-self.SNAKE_SPEED, 0]
        elif direction == 3:
            return [self.SNAKE_SPEED, 0]

    def look(self):
        # Define the 8 directions
        directions = [
            (-self.SNAKE_SIZE, 0),  # Left
            (-self.SNAKE_SIZE, -self.SNAKE_SIZE),  # Up-left
            (0, -self.SNAKE_SIZE),  # Up
            (self.SNAKE_SIZE, -self.SNAKE_SIZE),  # Up-right
            (self.SNAKE_SIZE, 0),  # Right
            (self.SNAKE_SIZE, self.SNAKE_SIZE),  # Down-right
            (0, self.SNAKE_SIZE),  # Down
            (-self.SNAKE_SIZE, self.SNAKE_SIZE),  # Down-left
        ]

        # Look in each direction
        vision = []
        for direction in directions:
            vision.append(self.look_in_direction(direction))

        return vision

    def look_in_direction(self, direction):
        # Initialize the look array
        look = [0, 0, 0]

        # Get the position of the snake's head
        pos = list(self.snake.get_head_pos())

        # Initialize the distance
        distance = 0

        # Initialize the food found and body found flags
        food_found = False
        body_found = False

        # Move in the direction until hitting a wall
        while not self.wall_collide(pos):
            pos[0] += direction[0]
            pos[1] += direction[1]
            distance += 1

            # Check for food
            if not food_found and self.food_collide(pos):
                food_found = True
                look[0] = 1

            # Check for body
            if not body_found and self.body_collide(pos):
                body_found = True
                look[1] = 1

        # Set the distance to the wall
        look[2] = 1 / distance if distance != 0 else 1

        return look

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over()

            # Get the current state of the game
            state = self.get_state()

            # Debug: Print state to ensure it is being calculated correctly
            print("State:", state)

            # Convert state to a one-dimensional array
            state = np.array(state, dtype=float)

            # Get the direction from the AI
            direction = self.brain.forward(state)

            # Debug: Print direction to ensure AI is making a decision
            print("Direction:", direction)

            # Map the direction to a velocity
            direction = self.map_direction_to_velocity(direction)

            self.snake.move(direction)

            if self.snake.get_head_pos() == self.food_pos:
                self.score += 1
                print("Food Eaten! Score:", self.score)
                self.generate_food()
            else:
                self.snake.shrink()

            head_pos = self.snake.get_head_pos()
            if head_pos[0] < 0 or head_pos[0] > self.WIDTH - 20 or head_pos[1] < 0 or head_pos[1] > self.HEIGHT - 20:
                print("Wall Collision")
                self.game_over()
            if self.snake.collides_with_self():
                print("Self Collision")
                self.game_over()

            if self.display:
                self.screen.fill(pygame.Color(0, 0, 0))

                for body_part in self.snake.get_body():
                    pygame.draw.rect(self.screen, pygame.Color(0, 255, 0), pygame.Rect(*body_part))

                pygame.draw.rect(self.screen, pygame.Color(255, 0, 0),
                                 pygame.Rect(self.food_pos[0], self.food_pos[1], self.SNAKE_SIZE, self.SNAKE_SIZE))

                pygame.display.flip()

            self.clock.tick(60)

    def generate_food(self):
        self.food_pos = [random.randrange(self.WIDTH // self.SNAKE_SIZE) * self.SNAKE_SIZE,
                         random.randrange(self.HEIGHT // self.SNAKE_SIZE) * self.SNAKE_SIZE]

    def wall_collide(self, pos):
        """Check if a given position collides with the wall."""
        return pos[0] < 0 or pos[0] > self.WIDTH - 20 or pos[1] < 0 or pos[1] > self.HEIGHT - 20

    def food_collide(self, pos):
        """Check if a given position collides with the food."""
        return pos == self.food_pos

    def body_collide(self, pos):
        """Check if a given position collides with the snake's body."""
        return pos in self.snake.get_body()
