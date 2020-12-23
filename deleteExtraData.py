import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ))

# path_to_target_file = root_dir + '/not_nan_target.csv'

# df_target = pd.read_csv(path_to_target_file)
# print('df size before: ', df_target.size)
# df_target = df_target.drop(df_target[df_target['target_value'] > 2000].index)
# print('df size after: ', df_target.size)
# df_target.to_csv(root_dir+'/paired_targetU2000NotNan.csv')




# img0_paths = df_target['image1']
# img1_paths = df_target['image2']
# h = df_target['target_value']
# h = np.log(h)
# h_max = np.max(h)
# pixels_norm_factor = 1920
# print('hmax: ', h_max)
# print('h_min: ', np.min(h))

path_to_target_file = root_dir + '/cbh_meters.csv'
df = pd.read_csv(path_to_target_file)
h = df['CBH_meters']
print('h max: ', np.max(h))
print('h min: ', np.min(h))
