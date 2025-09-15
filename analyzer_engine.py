# analyzer_engine.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # 用于可视化降维
import numpy as np
from collections import Counter

def build_feature_matrix_for_clustering(texts_list, max_features=500):
    """
    基于文本列表构建TF-IDF特征矩阵。
    texts_list: 字符串列表，每个字符串是空格分隔的词语。
    """
    if not texts_list or all(not text for text in texts_list):
        print("警告: 输入的文本列表为空或所有文本均为空，无法构建特征矩阵。")
        return None, None # 返回 None 表示无法处理

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=max_features, stop_words=None) # 使用预处理过的文本，不再需要stopwords
    try:
        feature_matrix = vectorizer.fit_transform(texts_list)
        print(f"成功构建特征矩阵，形状: {feature_matrix.shape}")
        return feature_matrix, vectorizer.get_feature_names_out()
    except ValueError as e:
        print(f"构建特征矩阵时出错: {e}。可能因为有效词汇过少。")
        # 可以尝试返回一个全零矩阵或更简单的特征，或者直接返回None
        return None, None


def cluster_friends_kmeans(feature_matrix, n_clusters=3, random_state=42):
    """使用K-Means进行聚类。"""
    if feature_matrix is None or feature_matrix.shape[0] < n_clusters:
        print(f"警告: 特征矩阵为空或样本数 ({feature_matrix.shape[0] if feature_matrix is not None else 0}) 小于簇数 ({n_clusters})。无法进行聚类。")
        # 返回一个全为-1的标签数组，表示聚类失败
        return np.full(feature_matrix.shape[0] if feature_matrix is not None else 0, -1, dtype=int)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    try:
        labels = kmeans.fit_predict(feature_matrix)
        print(f"K-Means聚类完成，得到 {len(np.unique(labels))} 个簇。")
        return labels
    except Exception as e:
        print(f"K-Means聚类时出错: {e}")
        return np.full(feature_matrix.shape[0], -1, dtype=int)


def generate_user_profile(friend_data_series, segmented_signature, keywords, sentiment, scraped_info=None):
    """
    生成单个用户画像字典。
    friend_data_series: pandas Series, 一行好友数据
    segmented_signature: list of words
    keywords: list of keywords for this user
    sentiment: float
    scraped_info: dict from web_scraper
    """
    profile = {
        "昵称": friend_data_series.get('Nickname', 'N/A'),
        "备注": friend_data_series.get('RemarkName', 'N/A'),
        "原始签名": friend_data_series.get('Signature', 'N/A'),
        "分词后签名": " ".join(segmented_signature) if segmented_signature else "无有效内容",
        "主要兴趣点(关键词)": ", ".join(keywords) if keywords else "未能提取",
        "签名情感倾向": f"{sentiment:.2f} ({'积极' if sentiment > 0.6 else '消极' if sentiment < 0.4 else '中性'})",
        "地区": friend_data_series.get('Region', 'N/A'),
        "模拟动态/标签": friend_data_series.get('Simulated_Last_Activity_Text', 'N/A') + " | " + friend_data_series.get('Tags', 'N/A')
    }
    if scraped_info:
        profile["相关百科信息"] = scraped_info
    return profile

def get_cluster_top_keywords(feature_matrix, all_feature_names, labels, cluster_id, top_n=10):
    """获取指定簇的TF-IDF最高的关键词"""
    if feature_matrix is None or all_feature_names is None:
        return []
    
    # 获取属于该簇的所有文档的索引
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return []

    # 计算该簇中文档的平均TF-IDF值
    cluster_vector = feature_matrix[indices].mean(axis=0)
    
    # 将稀疏矩阵行向量转换为稠密数组
    if hasattr(cluster_vector, "A"): # A 是 numpy.matrix 的属性，用于转为ndarray
        cluster_vector_dense = cluster_vector.A.flatten()
    else: # 如果已经是ndarray
        cluster_vector_dense = np.array(cluster_vector).flatten()


    # 获取TF-IDF值最高的top_n个词的索引
    top_indices = cluster_vector_dense.argsort()[-top_n:][::-1]
    
    top_keywords = [all_feature_names[i] for i in top_indices if cluster_vector_dense[i] > 0] # 只取有值的
    return top_keywords


def recommend_topics_for_cluster(cluster_keywords, scraped_news_dict):
    """
    根据簇的关键词和爬取的新闻，为簇推荐话题。
    cluster_keywords: list of keywords for the cluster
    scraped_news_dict: dict where keys are topics and values are lists of headlines
    """
    recommendations = []
    if not cluster_keywords or not scraped_news_dict:
        return ["无可用话题推荐"]

    for keyword in cluster_keywords[:3]: # 用簇的前3个关键词尝试匹配
        if keyword in scraped_news_dict:
            recommendations.extend(scraped_news_dict[keyword][:2]) # 最多推荐2条相关新闻
    
    if not recommendations:
        # 如果没有直接匹配，可以从所有新闻中随机选一些，或者根据更广泛的类别
        # 这里简单返回一个通用提示
        all_news = []
        for news_list in scraped_news_dict.values():
            all_news.extend(news_list)
        if all_news:
            return list(set(all_news[:3])) # 取不重复的前3条作为通用推荐
        else:
            return ["暂无相关新闻资讯可推荐"]
            
    return list(set(recommendations)) # 去重

if __name__ == '__main__':
    # 简单测试
    from config import STOPWORDS_PATH
    from nlp_processor import segment_text, load_stopwords
    
    stopwords = load_stopwords(STOPWORDS_PATH)
    
    docs = [
        "我 喜欢 编程 和 算法",
        "他 热爱 篮球 运动",
        "自然语言处理 是 人工智能 的 分支",
        "机器 学习 技术 发展 迅速",
        "每天 坚持 跑步 有益健康",
        "深度 学习 模型 应用 广泛"
    ]
    # 假设 docs 是已经分词并用空格连接的文本
    
    matrix, features = build_feature_matrix_for_clustering(docs, max_features=10)
    if matrix is not None:
        print("特征词:", features)
        labels = cluster_friends_kmeans(matrix, n_clusters=2)
        print("聚类标签:", labels)

        if features is not None and len(features) > 0 :
            cluster_0_keywords = get_cluster_top_keywords(matrix, features, labels, 0, top_n=3)
            print("簇0的关键词:", cluster_0_keywords)
            cluster_1_keywords = get_cluster_top_keywords(matrix, features, labels, 1, top_n=3)
            print("簇1的关键词:", cluster_1_keywords)
        else:
            print("无法获取簇关键词，因为特征名为空。")

    else:
        print("无法进行后续测试，因为特征矩阵构建失败。")

    # 测试用户画像生成 (模拟数据)
    sample_friend_data = pd.Series({
        'Nickname': '测试用户', 'RemarkName': '测试备注', 'Signature': '爱生活爱技术',
        'Region': '北京', 'Simulated_Last_Activity_Text': '刚看完一部科幻电影', 'Tags': '科幻,技术'
    })
    sample_segments = ['爱', '生活', '爱', '技术']
    sample_keywords = ['生活', '技术']
    sample_sentiment = 0.8
    sample_scraped = {"百科_技术": "技术是指..."}

    profile = generate_user_profile(sample_friend_data, sample_segments, sample_keywords, sample_sentiment, sample_scraped)
    print("\n用户画像示例:")
    for key, value in profile.items():
        print(f"  {key}: {value}")