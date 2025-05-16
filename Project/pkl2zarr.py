import zarr
import numpy as np
import pickle
import os
from numcodecs import Blosc

# for pkl_file in os.listdir(pkl_file_folder):
#     if pkl_file.endswith(".pkl"):
#         pkl_file_path = os.path.join(pkl_file_folder, pkl_file)
#         zarr_file_path = pkl_file_path.replace(".pkl", ".zarr")
#         with open(pkl_file_path, "rb") as f:
#             data = pickle.load(f)
#             state = data["joint_positions"]
#             action = data["control"]
#             print("state shape: ", state.shape)
#             print("action shape: ", action.shape)
#             # break
#             zarr.save_group(zarr_file_path, state=state, action=action)
            
            
#         print(f"Converted {pkl_file_path} to {zarr_file_path}")

def load_pkl(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)
        # state = data["joint_positions"]
        # action = data["control"]
        return data
    
def convert_and_store(data, data_group):
    # for key, value in data.items():
    key = 'joint_positions'
    value = data[key]
    # zarr_group = group.create_group('state')
    data_group.create_dataset('state', data=value, chunks=True)
    
    key = 'control'
    value = data[key]
    # zarr_group = group.create_group('action')
    data_group.create_dataset('control', data=value, chunks=True)
        
def process_folder(folder_path, output_file, compressor):
    # Create zarr file
    z = zarr.open(output_file, mode ='w')
    
    # Create 'data' and 'meta' groups
    data_group = z.create_group('data')
    meta_group = z.create_group('meta')
    # meta_group = meta_group.create_group('episode_ends')    
    
    initial_shape_action = (1, 7)
    initial_shape_state = (1, 5)
    
    # Iterate over pickle files
    for i, file_name in enumerate(os.listdir(folder_path)):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)
            data = load_pkl(file_path)
            action_arr = data['control']
            # action_arr = np.random.rand(2)
            # state_arr = data['joint_positions']
            state_arr = np.random.rand(5)
            img_arr = np.random.rand(96, 96, 3)
            keypoint_arr = np.random.rand(9, 2)
            n_contacts_arr = np.random.rand(1)
            
            # episode_end_arr = np.random.randint(0, 2, (20,))
            
            if 'action' not in data_group:
                action_dataset = data_group.create_dataset("action",
                                                           shape = initial_shape_action,
                                                           dtype = float,
                                                           chunks = (30,7),
                                                           compressor = compressor,
                                                           order = 'C',
                                                           maxshape = (None, 7))
                
            else:
                action_dataset = data_group['action']
                            
            # print("action_dataset.shape: ", action_dataset.shape)
            action_dataset.append(action_arr.reshape(1, 7))
            
            if "state" not in data_group:
                state_dataset = data_group.create_dataset("state",
                                                          shape = initial_shape_state,
                                                          dtype = float,
                                                          chunks = (30,5),
                                                          compressor = compressor,
                                                          order = 'C',
                                                          maxshape = (None, 5))
            else:
                state_dataset = data_group['state']
                
            state_dataset.append(state_arr.reshape(1, 5))
            
            if "img" not in data_group:
                img_dataset = data_group.create_dataset("img",
                                                        shape = (1, 96, 96, 3),
                                                        dtype = float,
                                                        chunks = (30, 96, 96, 3),
                                                        compressor = compressor,
                                                        order = 'C',
                                                        maxshape = (None, 96, 96, 3))
            else:
                img_dataset = data_group['img']
                
            img_dataset.append(img_arr.reshape(1, 96, 96, 3))
            
            if "keypoint" not in data_group:
                keypoint_dataset = data_group.create_dataset("keypoint",
                                                             shape = (1, 9, 2),
                                                             dtype = float,
                                                             chunks = (30, 9, 2),
                                                             compressor = compressor,
                                                             order = 'C',
                                                             maxshape = (None, 9, 2))
            else:
                keypoint_dataset = data_group['keypoint']
                
            keypoint_dataset.append(keypoint_arr.reshape(1, 9, 2))
            
            if "n_contacts" not in data_group:
                n_contacts_dataset = data_group.create_dataset("n_contacts",
                                                              shape = (1, 1),
                                                              dtype = float,
                                                              chunks = (30, 1),
                                                              compressor = compressor,
                                                              order = 'C',
                                                              maxshape = (None, 1))
            else:
                n_contacts_dataset = data_group['n_contacts']
                
            n_contacts_dataset.append(n_contacts_arr.reshape(1, 1))
    
    episode_end_arr = np.arange(len(z['data/action']), 10, -20)
    # print("len(z['data/action'])", len(z['data/action']))
    len_episode_end_arr = len(episode_end_arr)  
    if "episode_ends" not in meta_group:
        episode_ends_dataset = meta_group.create_dataset("episode_ends",
                                                            shape = (len_episode_end_arr,),
                                                            dtype = int,
                                                            chunks = len_episode_end_arr + 2,
                                                            compressor = compressor,
                                                            order = 'C')
    else:
        episode_ends_dataset = meta_group['episode_ends']
        # print("z['meta']['episode_ends'][:]", z['meta']['episode_ends'][:])
    episode_ends_dataset.append(episode_end_arr.reshape(len_episode_end_arr,))
            
            
    if 'action' in data_group:
        data_group['action'].resize((len(data_group['action']), 7))
    if 'state' in data_group:
        data_group['state'].resize((len(data_group['state']), 5))
    if 'img' in data_group:
        data_group['img'].resize((len(data_group['img']), 96, 96, 3))
    if 'keypoint' in data_group:
        data_group['keypoint'].resize((len(data_group['keypoint']), 9, 2))
    if 'n_contacts' in data_group:
        data_group['n_contacts'].resize((len(data_group['n_contacts']), 1))
        
    if 'episode_ends' in meta_group:
        meta_group['episode_ends'].resize(len_episode_end_arr,)
        # reverse the episode_ends
        meta_group['episode_ends'] = episode_end_arr[::-1]
        
    print("z['meta']['episode_ends'][:]", z['meta']['episode_ends'][:])
            
            
    z.attrs.update(meta_group.attrs)
    


if __name__ == "__main__":
    
    pkl_file_folder = "0427_181723"
    output_file = "0427_181723.zarr"
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
    process_folder(pkl_file_folder, output_file, compressor)
    print(f"Converted {pkl_file_folder} to {output_file}")