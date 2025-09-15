# nlp_processor.py
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from snownlp import SnowNLP
from collections import Counter

# 加载用户词典（如果需要）
# jieba.load_userdict("user_dict.txt")

def load_stopwords(filepath):
    """加载停用词列表。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f}
        print(f"成功加载停用词: {filepath}, 共 {len(stopwords)} 个。")
        return stopwords
    except FileNotFoundError:
        print(f"错误: 停用词文件未找到 {filepath}")
        return set()
    except Exception as e:
        print(f"加载停用词文件时出错: {e}")
        return set()


def segment_text(text, stopwords_set):
    """使用jieba进行分词并去除停用词和单字。"""
    if not isinstance(text, str) or not text.strip():
        return []
    # seg_list = jieba.cut(text, cut_all=False) # 精确模式
    seg_list = jieba.lcut(text) # 返回列表
    filtered_segments = [
        word for word in seg_list
        if word not in stopwords_set and len(word.strip()) > 1 and not word.isnumeric()
    ]
    return filtered_segments

def extract_keywords_tfidf(corpus_list, top_k=5):
    """
    从整个文档集合中提取TF-IDF关键词，并返回每个文档的关键词。
    corpus_list: 分词后的文档列表，每个文档是一个词的列表 e.g., [["词1", "词2"], ["词3", "词4"]]
    """
    if not corpus_list:
        return []

    # 将分词列表转换回空格分隔的字符串，以适配TfidfVectorizer
    processed_corpus = [" ".join(doc) for doc in corpus_list]

    try:
        vectorizer = TfidfVectorizer(max_features=1000) # 限制特征数量
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError as e: # 如果语料库太小或都是停用词
        print(f"TF-IDF计算错误: {e}. 可能语料库中有效词过少。")
        # 返回空关键词列表或基于简单词频的关键词
        all_keywords_per_doc = []
        for doc_segments in corpus_list:
            if not doc_segments:
                all_keywords_per_doc.append([])
                continue
            word_counts = Counter(doc_segments)
            top_words = [word for word, count in word_counts.most_common(top_k)]
            all_keywords_per_doc.append(top_words)
        return all_keywords_per_doc


    doc_keywords = []
    for i in range(tfidf_matrix.shape[0]):
        doc_vector = tfidf_matrix[i]
        # 获取非零元素的索引和值
        tfidf_scores = [(feature_names[idx], doc_vector[0, idx]) for idx in doc_vector.nonzero()[1]]
        # 按TF-IDF值降序排序
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        doc_keywords.append([keyword for keyword, score in sorted_keywords[:top_k]])

    return doc_keywords


def get_sentiment_snownlp(text):
    """使用SnowNLP进行情感分析（0-1，越接近1越积极）。"""
    if not isinstance(text, str) or not text.strip():
        return 0.5 # 中性
    try:
        s = SnowNLP(text)
        return s.sentiments
    except Exception: # 处理SnowNLP可能出现的错误
        return 0.5

if __name__ == '__main__':
    from config import STOPWORDS_PATH
    stopwords = load_stopwords(STOPWORDS_PATH)

    test_text1 = "我今天很高兴，天气真好！"
    test_text2 = "这个电影太烂了，我不喜欢。"
    test_text3 = "深圳是一个充满机遇的城市，但也很有挑战。"

    print(f"原文1: {test_text1}")
    segments1 = segment_text(test_text1, stopwords)
    print(f"分词1: {segments1}")
    print(f"情感1: {get_sentiment_snownlp(test_text1)}")

    print(f"\n原文2: {test_text2}")
    segments2 = segment_text(test_text2, stopwords)
    print(f"分词2: {segments2}")
    print(f"情感2: {get_sentiment_snownlp(test_text2)}")

    corpus = [segments1, segments2, segment_text(test_text3, stopwords)]
    print(f"\n语料库: {corpus}")
    all_doc_keywords = extract_keywords_tfidf(corpus, top_k=3)
    print(f"TF-IDF 关键词:")
    for i, keywords in enumerate(all_doc_keywords):
        print(f"  文档 {i+1}: {keywords}")