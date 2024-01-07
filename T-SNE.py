import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def plot_tsne(data_file, label_file, output_filename):
    data = np.load(data_file)
    labels = np.load(label_file)

    # 从原始数据中取第0位数字
    new_data = data[:, 0, :]

    # 使用 t-SNE 进行降维到2维，并设置随机种子
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    tsne_proj = tsne.fit_transform(new_data)

    # 归一化处理
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(tsne_proj)

    # 绘制散点图
    plt.figure(figsize=(8, 5))
    for i in range(normalized_data.shape[0]):
        if labels[i] == 0:
            plt.scatter(normalized_data[i, 0], normalized_data[i, 1], color='red')
        elif labels[i] == 1:
            plt.scatter(normalized_data[i, 0], normalized_data[i, 1], color='yellow')

    plt.title(f'Normalized t-SNE Scatter Plot - {output_filename}')
    plt.xlabel('Normalized t-SNE Component 1')
    plt.ylabel('Normalized t-SNE Component 2')
    plt.savefig(output_filename)
    plt.close()

# 6个文件的处理
file_numbers = [0, 1, 2, 3, 4, 5]
for number in file_numbers:
    data_file = fr'results/middleresults/transfomer_{number}output.npy'
    label_file = fr'results/middleresults/labels.npy'
    output_filename = f'results/analysis/tsne_plot_{number}.png'
    plot_tsne(data_file, label_file, output_filename)

# 将六张图合并为一张图
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
for i in range(len(file_numbers)):
    img = plt.imread(f'results/analysis/tsne_plot_{file_numbers[i]}.png')
    axes[i // 2, i % 2].imshow(img)
    axes[i // 2, i % 2].axis('off')

plt.show()
