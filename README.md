# slither-io-CNN
An experimental game-playing agent for Slither.io, exploring computer vision and reinforcement learning with a CNN-GRU policy trained using PPO.


envs/ handles the game. The simulator is useful for early training; the browser environment can capture the screen and send mouse/keyboard inputs.
cnn_gru.py turns recent screen frames into an action choice and a value estimate.
rollout.py records observations, actions, rewards, and outcomes while the agent plays.
ppo.py uses those recorded games to update the model.
rewards.py defines what counts as progress, such as staying alive or collecting pellets.
train.py connects the environment, model, and PPO training loop. play.py loads a saved checkpoint and runs the agent.

A useful interface for each environment is reset() to start a game and step(action) to apply an action and return the next screen, reward, and whether the game ended. For a first version, keep actions simple: choose a steering direction and whether to boost.

Put generated checkpoints and logs in .gitignore so they don’t clutter the repository.