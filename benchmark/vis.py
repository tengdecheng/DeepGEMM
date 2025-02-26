import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'all_perf.csv'

df = pd.read_csv(file_path)

names = df['name'].unique()

bs = 0
seq_lens = []

for name in names:
  subset = df[df['name'] == name]
  seq_lens = subset["seqlen"]
  bs = df["batch"][0]
  plt.plot(subset['seqlen'], subset['bw'], label=name, )

# plt.title('bandwidth(bs{})'.format(bs))
# plt.xlabel('seqlen')
# plt.ylabel('BS(GB/s)')
# # plt.xticks(seq_lens)
# # plt.xscale('log')
# # plt.axvline(x=3, color='r', linestyle='--', label='Vertical Line at x=3')
# plt.legend()

# plt.savefig('bandwidth_vs_seqlen_bs{}.png'.format(bs))