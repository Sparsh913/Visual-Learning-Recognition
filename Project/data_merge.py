import os

data_paths = ['/media/uas-laptop/KANTOR-LAB/colmap_data/cic_ur5/stream_camera_1-20240518T072454Z-001_stream_camera_1', '/media/uas-laptop/KANTOR-LAB/colmap_data/cic_ur5/stream_camera_2-20240518T072551Z-001_stream_camera_2', '/media/uas-laptop/KANTOR-LAB/colmap_data/cic_ur5/stream-20240518T071636Z-001_stream']

# there are  3 folders in the data_path whose name starts with stream
# copy the files from these and merge them into a single folder but the filenames are similar as of now
# so rename them to have unique names

# create a new folder to store the merged data
merged_data_path = '/media/uas-laptop/KANTOR-LAB/colmap_data/cic_ur5/merged_data'
os.makedirs(merged_data_path, exist_ok=True)

# iterate over the data_paths
for data_path in data_paths:
    # get the list of files in the data_path
    files = os.listdir(data_path)
    for file in files:
        # copy the file to the merged_data_path without the folder. The files should be there directly in the merged_data folder
        # rename the file to have a unique name
        # the new name should be the 'stream_i_' + file where i is the index of the data_path in the data_paths list
        # for example, if the file is 'image_0001.jpg' and the data_path is data_paths[0], then the new name should be 'stream_0_image_0001.jpg'
        # copy the file to the merged_data_path
        
        # get the index of the data_path in the data_paths list
        index = data_paths.index(data_path)
        # get the new name
        new_name = 'stream_' + str(index) + '_' + file
        # copy the file to the merged_data_path
        os.system('cp ' + os.path.join(data_path, file) + ' ' + os.path.join(merged_data_path, new_name))
        
        
print('Data merged successfully')