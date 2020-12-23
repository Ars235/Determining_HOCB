import matplotlib.pyplot as plt
import pandas as pd

root_dir = 'D:/SuperGlueDir/'
df = pd.read_csv(root_dir+'not_nan_target.csv')

h = df['target_value']
plt.hist(x=h, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('VNGO (m)')
plt.ylabel('Frequency')
# plt.title(title_str)
plt.savefig(root_dir + "distribution.png")
