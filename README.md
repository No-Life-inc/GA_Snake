# GA_SNAKE

## Contributors

- Morten Bendeke
- Betül Iskender
- Yelong Hartl-He
- Zack Ottesen

## General Use

This repo is for creating a Genetic Algorithm that can learn to play snake.<br>
The goal is to try and find the best combination of crossover and selection methods to acheive a snake with the highest score.
The best snake from each generation is saved in the best_snakes folder.


## How To Run


Pip install poetry -> poetry install -> poetry shell<br>
under main.py set the population size, generations, crossover and selection methods
Lastly, use the following command:

```bash
python main.py
```

## Dependencies

Poetry is used to handle dependencies.
Pip install poetry -> poetry install -> poetry shell

## Acknowledgements
- Game State method inspired by Greer Viau https://github.com/greerviau/SnakeAI
- Game State method also inspired by Ali Akbar https://github.com/aliakbar09a/AI_plays_snake 
