import zarr
import numpy as np

file = "pusht_cchi_v7_replay.zarr"
file = "sample.zarr"

zar = zarr.open(file, 'r')
print(zar.tree())
# print(zar['data/state'])
# print(zar['data/action'].info)
# print(zar['meta/episode_ends'].info)

# extract numpy array from action. action has chunks of arrays in it -> 0.0, 1.0, 2.0, ...
action = zar['data/action']

# chunks
action_chunks = zar['data/action'].chunks
print("action_chunks: ", action_chunks)

state_chunks = zar['data/state'].chunks
print("state_chunks: ", state_chunks)

state = zar['data/state']
# print(action[0])
print("length of action: ", len(action))
print("zarr action: ", action[:10])
print("zarr state: ", state[:10])
print("length of meta/episode_ends: ", len(zar['meta/episode_ends']))
print("zar['meta']['episode_ends'][:]: ", zar['meta']['episode_ends'][:])