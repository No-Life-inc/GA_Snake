from game import SnakeGame

#path to pickled snake
path = 'Gen_95_snake.pkl'

game = SnakeGame(display=True)

game.play_back(path)