import json
import math

import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter
from nycu_cv_hw2.constants import DATA_DIR_PATH, OUTPUT_DIR_PATH

widths = []
heights = []
sizes = []
aspect_ratios = []

# 讀入資料
with open(DATA_DIR_PATH / "train.json") as f:
    data = json.load(f)
for annotation in data["annotations"]:
    w, h = annotation["bbox"][2:4]
    widths.append(w)
    heights.append(h)
    sizes.append(math.sqrt(w * h))
    aspect_ratios.append(w / h)

with open(DATA_DIR_PATH / "valid.json") as f:
    data = json.load(f)
for annotation in data["annotations"]:
    w, h = annotation["bbox"][2:4]
    widths.append(w)
    heights.append(h)
    sizes.append(math.sqrt(w * h))
    aspect_ratios.append(w / h)


# 畫圖 + 存檔工具
def plot_and_save(data, title, xlabel, filename):
    data = np.array(data)
    mean_val = np.mean(data)
    std_val = np.std(data)

    plt.figure()
    plt.hist(data, bins=50, color="skyblue", edgecolor="black")
    plt.axvline(
        mean_val,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    plt.axvline(
        mean_val + std_val,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"+1σ: {mean_val + std_val:.2f}",
    )
    plt.axvline(
        mean_val - std_val,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"-1σ: {mean_val - std_val:.2f}",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # 印出統計數值
    print(f"{title}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Std:  {std_val:.2f}")
    print("-" * 40)


f = Fitter(sizes, distributions="common")
f.fit()
best_fit = f.get_best()
print(best_fit)

# 儲存四張圖
plot_and_save(
    widths,
    "Width Distribution",
    "Width (pixels)",
    OUTPUT_DIR_PATH / "width_distribution.png",
)
plot_and_save(
    heights,
    "Height Distribution",
    "Height (pixels)",
    OUTPUT_DIR_PATH / "height_distribution.png",
)
plot_and_save(
    sizes,
    "Anchor Size (sqrt(w*h)) Distribution",
    "Size",
    OUTPUT_DIR_PATH / "size_distribution.png",
)
plot_and_save(
    aspect_ratios,
    "Aspect Ratio (w/h) Distribution",
    "Aspect Ratio",
    OUTPUT_DIR_PATH / "aspect_ratio_distribution.png",
)


def print_size_histogram(data, bin_width=2):
    data = np.array(data)
    max_value = np.max(data)
    min_value = np.min(data)

    # 設置每 2 單位的區間範圍
    bins = np.arange(min_value, max_value + bin_width, bin_width)

    # 計算每個區間的資料數量
    hist, bin_edges = np.histogram(data, bins=bins)

    # 印出每個區間的資料數量
    for i in range(len(hist)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {hist[i]}")


# 打印 sizes 每 2 單位區間的資料數量
print_size_histogram(sizes, bin_width=2)
