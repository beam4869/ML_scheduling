import matplotlib.pyplot as plt
import random
from collections import Counter 
import pandas as pd
# df = pd.read_csv("I:\\python\\PCA_loop\\first two PCA component plot.csv")
df = pd.read_csv("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/python/PCA_loop/first two PCA component plot.csv")
x0 = df["x0"]
y0 = df["y0"]
plt.figure(figsize=(10,5.2))
plt.scatter(x0, y0, alpha=0.3)
x1 = df["x1"]
y1 = df["y1"]
plt.scatter(x1, y1, alpha=0.3)
x2 = df["x2"]
y2 = df["y2"]
plt.scatter(x2, y2, alpha=0.3)
x3 = df["x3"]
y3 = df["y3"]
plt.scatter(x3, y3, alpha=0.3)
x4 = df["x4"]
y4 = df["y4"]
plt.scatter(x4, y4, alpha=0.3)
labels = ["[[1, 2], [3, 4]]", "[[4], [1, 2, 3]]","[[1, 4], [2, 3]]","[[1], [2, 3, 4]]", "[[2], [1, 3, 4]]"]
plt.legend(labels=labels, loc="upper left", frameon=False)
plt.xlabel('PCA component one', fontsize = 18)
plt.ylabel('PCA component two', fontsize = 18)
plt.ylim(-0.10, 1.2)
plt.xlim(-0.10, 1.2)
plt.show()