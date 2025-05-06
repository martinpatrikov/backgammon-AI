# backgammon-AI

A command-line implementation of backgammon that lets you play against yourself, a friend, or an AI opponent.

## Prerequisites

You’ll need an Anaconda installation available on your machine.

### Installing the gym-backgammon Environment

This project uses the [gym-backgammon](https://github.com/dellalibera/gym-backgammon) package both to train the reinforcement-learning agent and to render the board. To add it to your setup:

```bash
git clone https://github.com/dellalibera/gym-backgammon.git
cd gym-backgammon/
pip3 install -e .
```

You’ll also require PyTorch:

```bash
pip install torch torchvision
```

## Launching a Game

Start a new session by running:

```bash
python play.py -begin
```

An ASCII representation of the backgammon board will appear, for example:

```
| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |
|--------Outer Board----------|     |-------P=O Home Board--------|     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |  X |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |  X |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |    |     |
|  X |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|  X |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|-----------------------------|     |-----------------------------|     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |  O |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |  O |     |
|--------Outer Board----------|     |-------P=X Home Board--------|     |
| 11 | 10 |  9 |  8 |  7 |  6 | BAR |  5 |  4 |  3 |  2 |  1 |  0 | OFF |
```

You may choose to alternate turns with another human or hand control over to the AI.

## Human Moves

If you need dice rolls, generate them with:

```bash
python roll_dice.py
```

To submit your own move(s), use:

```bash
python play.py -human <source> <destination> [<source2> <destination2> …]
```

- `<source>`: the point your checker starts on  
- `<destination>`: where you want to move it  

### Example

You roll a 6 and a 1 as player X and wish to move from point 12 to 6, then point 7 to 6:

```bash
python play.py -human 12 6 7 6
```

If you roll doubles (e.g. 6 6), enter four moves:

```bash
python play.py -human 12 6 12 6 23 17 23 17
```

## Entering Moves from the Bar

Captured checkers sit on the bar; to re-enter them, use `bar` as the source:

```bash
python play.py -human bar 23 bar 20
```

*(This example places two X checkers back into play on rolls of 1 and 4.)*

## Bearing Off

Once all your checkers are in your home quadrant, remove them with destination codes:

- **X** pieces: `-1`  
- **O** pieces: `24`  

```bash
python play.py -human 5 -1 3 -1   # X bearing off on rolls 6 and 4
python play.py -human 18 24 20 24 # O bearing off on rolls 6 and 4
```

## Skipping a Turn

If your dice roll leaves you with no legal moves (and it’s your turn):

```bash
python play.py -skip
```

## Letting the AI Play

Hand control to the AI by supplying your roll:

```bash
python play.py -ai <die1> <die2>
```

For instance, on a 1 and 6:

```bash
python play.py -ai 1 6
```

The AI will choose and display its optimal move(s).  
*Note:* if you roll doubles, you still only pass two numbers; the AI will execute four moves automatically. If the roll is unplayable, the AI will correctly detect and skip.

---

## How the AI Was Trained

The agent improved by playing 200,000 self-play games using a reinforcement-learning approach, resulting in a model capable of challenging human opponents.
