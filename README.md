### Newest update

I'm not sure which environment I was even using for running this! And its taking me a minute to see which one now! 
Label what environment I am using and everything esle I need to get stuff going 


### Update:

I merged the files over from `ppo-harvest`... (did I??? Coming back to this now I can't see it)

I also changed the gym `regiser` function so that its directly within the 
scripts that run. For some reason I was getting path issues potentially, 
I'm not too sure tbh. But its working now! 

-----

This is an OpenAI gym implementation of the Commons Game, a multi-agent environment proposed in [A multi-agent reinforcement learning model of common-pool resource appropriation](https://arxiv.org/abs/1707.06600) using [pycolab](https://github.com/deepmind/pycolab) as game engine.

## Installation

To install `cd` to the directory of the repository and run `pip install -e .`

## Usage

The file `example.py` contains a simple usage example where you can modify the number of agents and the size of its field of vision. To run the example `cd` to the directory of the repository and run `python example.py`. You should see something like this:

![](example.gif)
