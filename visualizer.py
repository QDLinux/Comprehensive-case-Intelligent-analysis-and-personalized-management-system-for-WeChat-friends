# visualizer.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import numpy as np
from config import FONT_PATH, OUTPUT_DIR # 确保 FONT_PATH 配置正确
import os

# 设置matplotlib的字体，解决中文乱码问题
# 放在模块级别，这样导入时就会尝试设置
try:
    plt.rcParams['font.sans-serif'] = [os.path.basename(FONT_PATH).split('.')[0]] # 例如 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    # 进一步确保字体被matplotlib识别
    from matplotlib.font_manager import fontManager
    font_paths = [FONT_PATH] # 可以添加更多字体路径
    for font_path in font_paths:
        if os.path.exists(font_path):
            fontManager.addfont(font_path)
        else:
            print(f"警告: visualizer.py - 字体文件 {font_path} 未找到，可能导致中文乱码。")

except Exception as e:
    print(f"设置matplotlib中文字体时出错: {e}。请检查config.py中的FONT_PATH。")


def plot_wordcloud_from_text(text_corpus_string, output_filename="wordcloud.png"):
    """
    根据文本语料（单个长字符串）生成词云。
    text_corpus_string: 所有文本拼接成的单个字符串，用空格分隔词语。
    """
    if not text_corpus_string.strip():
        print("文本内容为空，无法生成词云。")
        return

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        wordcloud = WordCloud(
            background_color='white',
            font_path=FONT_PATH, # 指定中文字体路径
            width=800,
            height=400,
            max_words=150,
            collocations=False #避免词语重复
        ).generate(text_corpus_string)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path)
        plt.close() # 关闭图像，释放内存
        print(f"词云已保存到: {output_path}")
    except RuntimeError as e: # Pillow字体相关错误
         print(f"生成词云失败: {e}. 请确保 FONT_PATH ('{FONT_PATH}') 是一个有效的字体文件路径。")
    except Exception as e:
        print(f"生成词云时发生未知错误: {e}")


def plot_clusters_pca(feature_matrix, labels, output_filename="cluster_plot.png"):
    """
    使用PCA降维到2D并绘制聚类结果散点图。
    feature_matrix: TF-IDF或其他特征矩阵 (稠密或稀疏)
    labels: 聚类标签
    """
    if feature_matrix is None or feature_matrix.shape[0] == 0 or feature_matrix.shape[1] < 2 :
        print("特征矩阵数据不足或维度过低，无法进行PCA降维和绘图。")
        return
    if len(np.unique(labels)) < 1 : # 至少要有一个簇
        print("聚类标签无效，无法绘制散点图。")
        return
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 如果是稀疏矩阵，转换为稠密矩阵
    if hasattr(feature_matrix, "toarray"):
        feature_matrix_dense = feature_matrix.toarray()
    else:
        feature_matrix_dense = feature_matrix

    if feature_matrix_dense.shape[1] < 2:
        print("特征矩阵维度不足2维，无法进行PCA到2D的降维。")
        # 可以考虑1D绘图或其他方式，这里直接返回
        return

    try:
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(feature_matrix_dense)

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        if -1 in unique_labels: # 如果有噪声点（例如DBSCAN）或者聚类失败的标签
            colors = plt.cm.get_cmap('viridis', len(unique_labels) -1 if -1 in unique_labels and len(unique_labels) > 1 else len(unique_labels))
        else:
            colors = plt.cm.get_cmap('viridis', len(unique_labels))


        for i, label in enumerate(unique_labels):
            if label == -1: # 噪声点用灰色表示
                plt.scatter(reduced_features[labels == label, 0],
                            reduced_features[labels == label, 1],
                            color='grey', label=f'噪声/未分配 ({np.sum(labels == label)})', alpha=0.5, s=30)
            else:
                # 确保颜色索引不越界
                color_idx = i if not (-1 in unique_labels and i >= np.where(unique_labels == -1)[0][0]) else i-1
                color_val = colors(color_idx % colors.N) # 使用取模防止索引越界

                plt.scatter(reduced_features[labels == label, 0],
                            reduced_features[labels == label, 1],
                            color=color_val, # 使用cmap获取颜色
                            label=f'簇 {label} ({np.sum(labels == label)})', alpha=0.7, s=50)
        
        plt.title('微信好友聚类结果 (PCA降维)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title="簇标签")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        print(f"聚类散点图已保存到: {output_path}")
    except Exception as e:
        print(f"绘制聚类散点图时出错: {e}")


if __name__ == '__main__':
    # 测试词云 (需要确保FONT_PATH正确)
    sample_text = "自然语言处理 机器 学习 微信 数据 分析 爬虫 案例 Python 课程 设计 计算机 智能 算法 项目"
    sample_text += " 朋友 社交 网络 用户 画像 兴趣 文本挖掘" * 3 # 增加词频
    plot_wordcloud_from_text(sample_text, "test_wordcloud.png")

    # 测试聚类图 (模拟数据)
    # 创建一些模拟的特征和标签
    from sklearn.datasets import make_blobs
    sim_features, sim_labels = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)
    plot_clusters_pca(sim_features, sim_labels, "test_cluster_plot.png")

    sim_features_sparse, _ = make_blobs(n_samples=50, centers=2, n_features=5, random_state=0)
    # 模拟聚类失败的情况
    failed_labels = np.array([-1] * 50)
    plot_clusters_pca(sim_features_sparse, failed_labels, "test_failed_cluster_plot.png")