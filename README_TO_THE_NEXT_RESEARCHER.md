Hi!

If you're the next researcher working on this, and you just want to dive into the code; I'd recommend looking at the README's. This file is going to describe a bit of the history, ideas, and the methodology behind a lot of this code.

The repo is basically split into three main parts: The antarctic ladder, latency analysis, and working paper authorship prediction; all of which can be found in their associated folders.

The antarctic ladder metrics are essentially finished (imo), but if you want to add more metrics please look at the associated readme.



Latency analysis has gone through a bunch of changes to get to its current state. Originally the latency analysis code matched working papers to their nearest neighbour measure (in embedding space); but that turned out to produce a bunch of matches which were impossible (Working Papers published after a measure supposed influencing said measure).

The current method we're using for latency analysis is a threshold method, this threshold was qualitiatively chosen based on the results of lag_distributions.py; depending on the analysis, a Measure is said to have matched with:
a) The most recent Working Paper whose cosine similarity is above the threshold.
or
b) The full set of Working Papers whose cosine similarities are above the threshold.



Working Paper Authorship implements models that predict who authored a Working Paper based on its position in embedding space. The README there spells out a little bit of the history and reasoning.


A significant chunk of the code in this repo is now written by Claude. If you want to continue doing this (and I would highly recommend it if you want to run quick experiments), I would recommend doing the following:
1) Ask for tests with new code changes. Review tests to make sure they are testing important code features.
2) Read the changes made, pay special attention to whether or not the code Claude is writing properly matches the modelling you want implemented.
3) Do the modelling yourself, but ask Claude to implement the machinery of modelling.