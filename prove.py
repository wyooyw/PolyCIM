import islpy as isl
import numpy as np

count_time = np.load("count_time.npy")
max_size = np.load("max_size.npy")
mean_size = np.load("mean_size.npy")
sum_size = np.load("sum_size.npy")

print(count_time)
print(max_size)
print(mean_size)
print(sum_size)

import matplotlib.pyplot as plt
plt.scatter(max_size, count_time)

# 添加标题和标签
plt.title('Simple Scatter Plot')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')

# 显示图形
plt.show()