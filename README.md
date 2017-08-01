# ERL
Exploration based Reinforcement Learning

# PixelCNN DQN
A tensorflow implementation of ['Count-Based Exploration with Neural Density Model'](https://arxiv.org/abs/1703.01310). It uses PixelCNN to measure the sparsity of one state and give corresponding bonus.

# DQN
DQN is modified from ['DQN-tensorflow'](https://github.com/devsisters/DQN-tensorflow)

# PixelCNN
PixelCNN is modified from ['gated-pixel-cnn'](https://github.com/jakebelew/gated-pixel-cnn)

# Attention
The project doesn't follow the paper strictly. I noticed that there can be Q-value explosion and I considered that it's due to the fact that both opposite transition may have positive rewards. As result, the Q value boost, so it's very unstable. Temporally, I only choose the top 5 bonus in one episode (maybe harmful) and combine it with Q-value update with Monte-Carlo Return.

# Reflection
Though code achieved average score 2451 in OpenAI gym board, but it seems not that stable like the other high ranking code.
<ol>
<li>If the epsilon is set to 0, it can't work well and may stuck at some local points.</li>
<li>I didn't experiment with many different hyperparameter.</li>
<li>I also integrated it into the <a href="https://github.com/miyosuda/async_deep_reinforce">A3C</a> code. It's actually hard to combine pixelCNN's gradients from different threads but I managed to do it and also achieved 2500 as the highest score.
</ol>
