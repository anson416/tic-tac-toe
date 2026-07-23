<h1 align="center">Tic Tac Toe with Reinforcement Learning</h1>

<p align="center">
  <a href="https://anson416.github.io/tic-tac-toe/"><strong>Project Page</strong></a>
</p>

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

### Train an AI agent (i.e., create a Q-table)

```text
python train.py [-s SIZE] \
                [-e EPOCHS] \
                [-lr LEARNING_RATE] \
                [-df DISCOUNT_FACTOR] \
                [-rs RANDOM_SEARCH] \
                [-me MIN_EPSILON] \
                [-ed EPSILON_DECAY] \
                [--seed SEED]

optional arguments:
  -s SIZE, --size SIZE
  -e EPOCHS, --epochs EPOCHS
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -df DISCOUNT_FACTOR, --discount_factor DISCOUNT_FACTOR
  -rs RANDOM_SEARCH, --random_search RANDOM_SEARCH
  -me MIN_EPSILON, --min_epsilon MIN_EPSILON
  -ed EPSILON_DECAY, --epsilon_decay EPSILON_DECAY
  --seed SEED
```

You can find the created Q-tables under "qtables/".

### Test the AI agent

```text
python test.py pkl_path \
               [-s SIZE] \
               [-e EPOCHS] \
               [--seed SEED]

positional arguments:
  pkl_path

optional arguments:
  -s SIZE, --size SIZE
  -e EPOCHS, --epochs EPOCHS
  --seed SEED
```

### Play (w/wo AI agent)

```text
python play.py [-s SIZE] \
               [-p PKL_PATH] \
               [-a1] \
               [-a2] \
               [-r RANDOM_PROB] \
               [--seed SEED]

optional arguments:
  -s SIZE, --size SIZE
  -p PKL_PATH, --pkl_path PKL_PATH
  -a1, --agent_1
  -a2, --agent_2
  -r RANDOM_PROB, --random_prob RANDOM_PROB
  --seed SEED
```
