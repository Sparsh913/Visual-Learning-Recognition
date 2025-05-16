import zarr
import numpy as np
import pickle
import os
from numcodecs import Blosc
import cv2

def load_pkl(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)
        # state = data["joint_positions"]
        # action = data["control"]
        return data
    
        
def process_folder(folder_path, output_file, compressor):
    # Create zarr file
    z = zarr.open(output_file, mode ='w')
    
    # Create 'data' and 'meta' groups
    data_group = z.create_group('data')
    meta_group = z.create_group('meta')
    # meta_group = meta_group.create_group('episode_ends')    
    
    initial_shape_action = (1, 2)
    initial_shape_state = (1, 2)
    episode_end_arr = []
    episode_idx = 0

    # iterate over folders in the data folder which contain pickle files
    for folder in os.listdir(folder_path):
        folder_pkl = os.path.join(folder_path, folder)
        if not os.path.isdir(folder_pkl):
            continue
        
        allfiles = [file for file in os.listdir(folder_pkl) if file.endswith(".pkl")]
        allfiles.sort()
        img_folder = os.path.join(folder_pkl, "images")
        print("img_folder: ", img_folder)
        img_files = [file for file in os.listdir(img_folder) if file.endswith(".png")]
        img_files.sort()
        # Iterate over pickle files
        i = 0
        for file_name, img in zip(allfiles, img_files):
            file_path = os.path.join(folder_pkl, file_name)
            data = load_pkl(file_path)
            action_arr = data['action'][:2]
            # action_arr = np.random.rand(2)
            state_arr = data['state'][:2]
            # state_arr = np.random.rand(5)
            # resize img to 240 x 320
            img = cv2.imread(os.path.join(img_folder, img))
            img = cv2.resize(img, (240, 320))
            # print("i", i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_arr = img # shape (240, 320, 3)
            keypoint_arr = np.random.rand(9, 2)
            n_contacts_arr = np.random.rand(1)
            
            # episode_end_arr = np.random.randint(0, 2, (20,))
            
            if 'action' not in data_group:
                action_dataset = data_group.create_dataset("action",
                                                        shape = initial_shape_action,
                                                        dtype = float,
                                                        chunks = (30,2),
                                                        compressor = compressor,
                                                        order = 'C',
                                                        maxshape = (None, 2))
                
            else:
                action_dataset = data_group['action']
                            
            # print("action_dataset.shape: ", action_dataset.shape)
            action_dataset.append(action_arr.reshape(1, 2))
            
            if "state" not in data_group:
                state_dataset = data_group.create_dataset("state",
                                                        shape = initial_shape_state,
                                                        dtype = float,
                                                        chunks = (30,2),
                                                        compressor = compressor,
                                                        order = 'C',
                                                        maxshape = (None, 2))
            else:
                state_dataset = data_group['state']
                
            state_dataset.append(state_arr.reshape(1, 2))
            
            if "img" not in data_group:
                img_dataset = data_group.create_dataset("img",
                                                        shape = (1, 240, 320, 3),
                                                        dtype = float,
                                                        chunks = (30, 240, 320, 3),
                                                        compressor = compressor,
                                                        order = 'C',
                                                        maxshape = (None, 240, 320, 3))
            else:
                img_dataset = data_group['img']
                
            img_dataset.append(img_arr.reshape(1, 240, 320, 3))
            
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
            
            i += 1
    
        episode_idx = len(data_group['action'])
        episode_end_arr.append(episode_idx)
    episode_end_arr = np.array(episode_end_arr)
    # episode_end_arr = np.arange(len(z['data/action']), 10, -20)
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
        data_group['action'].resize((len(data_group['action']), 2))
    if 'state' in data_group:
        data_group['state'].resize((len(data_group['state']), 2))
    if 'img' in data_group:
        data_group['img'].resize((len(data_group['img']), 240, 320, 3))
    if 'keypoint' in data_group:
        data_group['keypoint'].resize((len(data_group['keypoint']), 9, 2))
    if 'n_contacts' in data_group:
        data_group['n_contacts'].resize((len(data_group['n_contacts']), 1))
        
    if 'episode_ends' in meta_group:
        meta_group['episode_ends'].resize(len_episode_end_arr,)
        # reverse the episode_ends
        meta_group['episode_ends'] = episode_end_arr
        
    print("z['meta']['episode_ends'][:]", z['meta']['episode_ends'][:])
            
            
    z.attrs.update(meta_group.attrs)
    


if __name__ == "__main__":
    
    data_folder = "gello_data"
    output_folder = "sample.zarr"
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
    process_folder(data_folder, output_folder, compressor)
    print(f"Converted {data_folder} to {output_folder}")