import numpy as np
import pygame
import random
from snake import Snake
from ga_brain import GABrain
import pickle

class SnakeGame:
    def __init__(self, brain=None, width=800, height=600, snake_size=20, display=False):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.WIDTH, self.HEIGHT = width, height
        self.SNAKE_SIZE = snake_size
        self.SNAKE_SPEED = self.SNAKE_SIZE  # make sure the snake's speed is the same as the grid cell size
        self.snake = Snake(self.random_position(), [self.SNAKE_SPEED, 0], self.SNAKE_SIZE)
        self.food_pos = self.generate_food()
        if brain is not None:
            self.brain = brain
        else:
            self.brain = GABrain()
        self.score = 0
        self.display = display
        if self.display:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.game_over = False
        self.record = []
        
    def random_position(self):
        mid_width = self.WIDTH // 2
        mid_height = self.HEIGHT // 2
        return [mid_width - (mid_width % self.SNAKE_SIZE), mid_height - (mid_height % self.SNAKE_SIZE)]

    def reset(self):
        self.snake = Snake(self.random_position(), [self.SNAKE_SPEED, 0], self.SNAKE_SIZE)
        self.food_pos = self.generate_food()
        self.score = 0
        self.game_over = False
        self.display = False

    def end_game(self):
        self.game_over = True

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
        
    def euclidean_distance_to_food(self):
        head_pos = np.array(self.snake.get_head_pos())
        food_pos = np.array(self.food_pos)
        return np.linalg.norm(head_pos - food_pos)

    def get_state(self):

        state = []

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
        return state + vision_flat

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
        distance = 1
        food_distance = -1

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

        look[2] = 1/distance
        

        return look
    
    # an attempt to make the look function more efficient
    def new_look_in_direction(self, direction):
        # Initialize the look array
        look = [0, 0, 0]

        # Get the position of the snake's head
        pos = list(self.snake.get_head_pos())

        # Initialize the distance
        distance = 1

        # Move in the direction until hitting a wall
        while not self.wall_collide(pos):
            pos[0] += direction[0]
            pos[1] += direction[1]
            distance += 1

            # Check for food
            if self.food_collide(pos):
                look[0] = 1
                look[2] = distance  # Store the distance to the food
                return look

            # Check for body
            if self.body_collide(pos):
                look[1] = 1
                look[2] = distance  # Store the distance to the body
                return look

        # If no food or body was found, store the distance to the wall
        look[2] = distance
        return look

    def game_loop(self):
        while not self.game_over:

            if self.snake.steps == 0:
                self.end_game()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_game()

            # Get the current state of the game
            state = self.get_state()

            # Convert state to a one-dimensional array
            state = np.array(state, dtype=float)

            # Get the direction from the AI
            direction = self.brain.forward(state)

            # Map the direction to a velocity
            direction = self.map_direction_to_velocity(direction)

            self.snake.move(direction)

            if self.snake.get_head_pos() == self.food_pos:
                self.score += 1000
                self.snake.grow()
                # print("Food Eaten! Score:", self.score)
                self.food_pos = self.generate_food()
                # print("New Food Position:", self.food_pos)

            head_pos = self.snake.get_head_pos()
            if head_pos[0] < 0 or head_pos[0] > self.WIDTH - self.SNAKE_SIZE or head_pos[1] < 0 or head_pos[1] > self.HEIGHT - self.SNAKE_SIZE:
                # print("Wall Collision")
                self.end_game()
            if self.snake.collides_with_self():
                # print("Self Collision")
                self.end_game()

            if self.display:
                if not hasattr(self, 'screen'):
                    self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                    
                self.screen.fill(pygame.Color(0, 0, 0))

                for body_part in self.snake.get_body():
                    pygame.draw.rect(self.screen, pygame.Color(0, 255, 0), pygame.Rect(*body_part))

                pygame.draw.rect(self.screen, pygame.Color(255, 0, 0),
                                 pygame.Rect(self.food_pos[0], self.food_pos[1], self.SNAKE_SIZE, self.SNAKE_SIZE))

                pygame.display.flip()
                
                self.clock.tick(30)
            
            self.record.append((list(self.snake.get_body()), list(self.food_pos)))

        # score and snake age
        achieved_snake_age  = self.snake.age
        achieved_food_eaten = self.snake.food_eaten
        achieved_score = self.score
        
        self.reset()

        return achieved_snake_age, achieved_score, achieved_food_eaten
    
    def play_back(self, filename):
        # Load the recorded game states from the file
        if(False):
            self.load_game_states(filename)

            self.display = True

            # Loop through the recorded game states
            for state in self.record:
                # Update the snake's body and food position
                self.snake.set_body(state[0])
                self.food_pos = state[1]

                # Update the game display
                if self.display:
                    if not hasattr(self, 'screen'):
                        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

                    self.screen.fill(pygame.Color(0, 0, 0))

                    for body_part in self.snake.get_body():
                        pygame.draw.rect(self.screen, pygame.Color(0, 255, 0), pygame.Rect(*body_part))

                    pygame.draw.rect(self.screen, pygame.Color(255, 0, 0),
                                    pygame.Rect(self.food_pos[0], self.food_pos[1], self.SNAKE_SIZE, self.SNAKE_SIZE))

                    pygame.display.flip()

                    self.clock.tick(30)
                

    def wait_until_over(self):
        # Wait for the game to finish
        self.game_thread.join()

    def run(self):
        snake_age, score, food_eaten =  self.game_loop()
        return snake_age, score, food_eaten

    def generate_food(self):
        return  [random.randrange(self.WIDTH // self.SNAKE_SIZE) * self.SNAKE_SIZE,
                         random.randrange(self.HEIGHT // self.SNAKE_SIZE) * self.SNAKE_SIZE]

    def wall_collide(self, pos):
        """Check if a given position collides with the wall."""
        return pos[0] < 0 or pos[0] > self.WIDTH - self.SNAKE_SIZE or pos[1] < 0 or pos[1] > self.HEIGHT - self.SNAKE_SIZE

    def food_collide(self, pos):
        """Check if a given position collides with the food."""
        return pos == self.food_pos

    def body_collide(self, pos):
        """Check if a given position collides with the snake's body."""
        return pos in self.snake.get_body()
    
    def save_game_states(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.record, f)

    def load_game_states(self, filename):
        with open(filename, 'rb') as f:
            self.record = pickle.load(f)
