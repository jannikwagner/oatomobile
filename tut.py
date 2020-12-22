import oatomobile

# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01")

# Makes an initial observation.
observation = environment.reset()
done = False

for i in range(100):
  # Selects a random action.
  action = environment.action_space.sample()
  observation, reward, done, info = environment.step(action)
  # Renders interactive display.
  #environment.render(mode="human")  # no available video device

# Book-keeping: closes
environment.close()


# Rule-based agents.
import oatomobile.baselines.rulebased

agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)
action = agent.act(observation)

# Imitation-learners.
import torch
import oatomobile.baselines.torch

models = [oatomobile.baselines.torch.ImitativeModel() for _ in range(4)]
ckpts = ... # Paths to the model checkpoints.
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
agent = oatomobile.baselines.torch.RIPAgent(
  environment=environment,
  models=models,
  algorithm="WCM",
)
action = agent.act(observation)

