class Snake:
    def __init__(self, initial_pos, speed, size, initial_steps=250):
        self.pos = [list(initial_pos)]
        self.speed = speed
        self.size = size
        self.direction = Direction()
        self.steps = initial_steps
        self.food_bonus = initial_steps
        self.age = 0
        self.food_eaten = 0
        self.grow_next_move = False
        

    def move(self, direction):
        self.direction.set(direction)
        new_head = [self.pos[0][0] + self.direction.get()[0], self.pos[0][1] + self.direction.get()[1]]
        self.pos.insert(0, new_head)
        if not self.grow_next_move:  # Add this line
            self.pos.pop()  # Add this line
        else:
            self.grow_next_move = False  # Reset the flag after growing
        self.steps -= 1
        self.age += 1

    def grow(self):
        self.grow_next_move = True
        self.food_eaten += 1
        self.steps = self.food_bonus

    def collides_with_self(self):
        for block in self.pos[1:]:
            if self.pos[0] == block:
                return True
        return False
    
    def get_body(self):
        return [(pos[0], pos[1], self.size, self.size) for pos in self.pos]
    
    def get_head_pos(self):
        return self.pos[0] if self.pos else [0, 0]

class Direction:
    def __init__(self):
        self.current = [0, 0]

    def set(self, direction):
        # Ignore direction change that would make the snake move in the opposite direction
        if direction[0] == -self.current[0] and direction[1] == -self.current[1]:
            return
        self.current = direction

    def get(self):
        return self.current