# main_controller.py
import pandas as pd
import time
import os
import argparse
from collections import Counter

import config # 从config导入配置
from data_handler import load_wechat_csv, clean_text
from nlp_processor import load_stopwords, segment_text, get_sentiment_snownlp # extract_keywords_tfidf (用于聚类特征)
from web_scraper import scrape_baidubaike_summary, scrape_generic_news_headlines
from analyzer_engine import (
    build_feature_matrix_for_clustering,
    cluster_friends_kmeans,
    generate_user_profile,
    get_cluster_top_keywords,
    recommend_topics_for_cluster
)
from visualizer import plot_wordcloud_from_text, plot_clusters_pca


def main(csv_filepath):
    print("--- 开始微信好友智能分析与个性化管理系统 ---")

    # 1. 加载和预处理数据
    print("\n--- 1. 数据加载与预处理 ---")
    df_friends = load_wechat_csv(csv_filepath)
    if df_friends is None:
        print("数据加载失败，程序终止。")
        return

    stopwords = load_stopwords(config.STOPWORDS_PATH)

    # 清洗和分词签名，计算情感
    df_friends['Cleaned_Signature'] = df_friends['Signature'].apply(clean_text)
    df_friends['Segmented_Signature'] = df_friends['Cleaned_Signature'].apply(lambda x: segment_text(x, stopwords))
    df_friends['Sentiment'] = df_friends['Cleaned_Signature'].apply(get_sentiment_snownlp)
    
    # 将分词结果转换回字符串，用于后续TF-IDF等处理
    df_friends['Segmented_Signature_Str'] = df_friends['Segmented_Signature'].apply(lambda x: " ".join(x))

    print(f"数据预处理完成，处理了 {len(df_friends)} 条好友数据。")
    print("示例处理后数据:")
    print(df_friends[['Nickname', 'Cleaned_Signature', 'Segmented_Signature_Str', 'Sentiment']].head())

    # 2. 特征工程与聚类
    print("\n--- 2. 好友聚类 ---")
    # 使用签名文本进行聚类
    corpus_for_clustering = df_friends['Segmented_Signature_Str'].tolist()
    
    # 过滤掉空字符串，防止 TfidfVectorizer 出错
    valid_corpus_indices = [i for i, text in enumerate(corpus_for_clustering) if text.strip()]
    valid_corpus = [corpus_for_clustering[i] for i in valid_corpus_indices]

    if not valid_corpus:
        print("没有有效文本进行聚类，跳过聚类步骤。")
        df_friends['Cluster_Label'] = -1 #标记为未聚类
    else:
        feature_matrix, feature_names = build_feature_matrix_for_clustering(valid_corpus, max_features=500)
        
        if feature_matrix is not None and feature_matrix.shape[0] > 0:
            # 确保聚类数量不超过样本数
            num_clusters = min(config.NUM_CLUSTERS, feature_matrix.shape[0])
            if num_clusters < 2 : # 如果样本太少，不聚类
                print(f"有效样本数 ({feature_matrix.shape[0]})过少，无法进行有意义的聚类。跳过聚类。")
                cluster_labels_full = pd.Series([-1] * len(df_friends)) # 全都设为-1
            else:
                print(f"开始对 {feature_matrix.shape[0]} 个有效样本进行 K-Means 聚类 (k={num_clusters})...")
                cluster_labels = cluster_friends_kmeans(feature_matrix, n_clusters=num_clusters)
                
                # 将聚类结果映射回原始DataFrame
                cluster_labels_full = pd.Series([-1] * len(df_friends), index=df_friends.index) # 初始化为-1
                for i, original_idx in enumerate(valid_corpus_indices):
                    cluster_labels_full.iloc[original_idx] = cluster_labels[i]
                
            df_friends['Cluster_Label'] = cluster_labels_full
            print("聚类完成。簇标签已添加到DataFrame。")
            print(df_friends[['Nickname', 'Cluster_Label']].head())

            # 可视化聚类结果 (如果特征矩阵有效)
            if feature_matrix.shape[0] >= num_clusters and feature_matrix.shape[1] >=2: # 确保数据足够降维和绘图
                 plot_clusters_pca(feature_matrix, cluster_labels, "friends_clusters_pca.png")
            else:
                print("特征矩阵数据不足，无法生成PCA聚类图。")

            # 分析每个簇的关键词
            if feature_names is not None and len(feature_names)>0:
                print("\n各簇的Top关键词:")
                for i in range(num_clusters):
                    if i not in cluster_labels: # 如果某个簇因为样本少等原因没有形成
                        continue
                    cluster_keywords = get_cluster_top_keywords(feature_matrix, feature_names, cluster_labels, cluster_id=i, top_n=5)
                    print(f"  簇 {i}: {', '.join(cluster_keywords)}")
            else:
                print("未能提取特征名，无法显示簇关键词。")

        else:
            print("特征矩阵构建失败或为空，跳过聚类。")
            df_friends['Cluster_Label'] = -1 #标记为未聚类
            feature_matrix = None # 确保后面不使用它
            feature_names = None


    # 3. 用户画像构建与爬虫信息增强 (示例几个用户)
    print("\n--- 3. 用户画像与信息增强 (示例前3位好友) ---")
    user_profiles = []
    scraped_news_for_recommendation = {} # 用于后续话题推荐

    # 为每个用户提取个人关键词 (简单方法：从其分词结果中取高频词)
    df_friends['Individual_Keywords'] = df_friends['Segmented_Signature'].apply(
        lambda seg_list: [word for word, count in Counter(seg_list).most_common(3)] if seg_list else []
    )

    for index, friend in df_friends.head(3).iterrows(): # 仅演示前3个用户
        print(f"\n正在处理好友: {friend.get('Nickname', '未知昵称')}")
        
        # 获取个人关键词
        individual_keywords = friend['Individual_Keywords']
        
        # 爬取信息 (基于个人关键词中的第一个)
        scraped_baike_info = {}
        if individual_keywords:
            keyword_to_scrape = individual_keywords[0]
            print(f"  尝试爬取 '{keyword_to_scrape}' 的百科信息...")
            summary = scrape_baidubaike_summary(keyword_to_scrape, num_sentences=1)
            scraped_baike_info[f"百科_{keyword_to_scrape}"] = summary
            time.sleep(0.5) # 友好爬取

            # 爬取与关键词相关的新闻标题，用于后续话题推荐
            if keyword_to_scrape not in scraped_news_for_recommendation:
                print(f"  尝试爬取与 '{keyword_to_scrape}' 相关的新闻标题...")
                news_headlines = scrape_generic_news_headlines(keyword_to_scrape, num_headlines=2)
                scraped_news_for_recommendation[keyword_to_scrape] = news_headlines
                time.sleep(0.5) # 友好爬取
        else:
            print("  无个人关键词可用于爬取。")

        profile = generate_user_profile(
            friend,
            friend['Segmented_Signature'],
            individual_keywords,
            friend['Sentiment'],
            scraped_baike_info
        )
        user_profiles.append(profile)
        
        print("  用户画像:")
        for key, value in profile.items():
            print(f"    {key}: {value}")

    # 4. 话题推荐 (基于聚类和爬取的新闻)
    print("\n--- 4. 话题推荐 (示例) ---")
    if 'Cluster_Label' in df_friends.columns and feature_matrix is not None and feature_names is not None:
        # 假设我们为第一个非-1的簇推荐话题
        # 找到第一个有效簇的ID
        first_valid_cluster_id = -1
        unique_labels = df_friends['Cluster_Label'].unique()
        for lbl in unique_labels:
            if lbl != -1:
                first_valid_cluster_id = lbl
                break
        
        if first_valid_cluster_id != -1:
            # 获取该簇的关键词
            example_cluster_keywords = get_cluster_top_keywords(
                feature_matrix, 
                feature_names, 
                df_friends.loc[valid_corpus_indices, 'Cluster_Label'].values, # 使用有效样本的标签
                cluster_id=first_valid_cluster_id, 
                top_n=5
            )
            if example_cluster_keywords:
                print(f"为 簇 {first_valid_cluster_id} (关键词: {', '.join(example_cluster_keywords)}) 推荐话题:")
                # 补充爬取这些簇关键词的新闻
                for ck in example_cluster_keywords:
                    if ck not in scraped_news_for_recommendation:
                        print(f"  尝试爬取与簇关键词 '{ck}' 相关的新闻标题...")
                        news_headlines = scrape_generic_news_headlines(ck, num_headlines=2)
                        scraped_news_for_recommendation[ck] = news_headlines
                        time.sleep(0.5)

                recommendations = recommend_topics_for_cluster(example_cluster_keywords, scraped_news_for_recommendation)
                for i, topic in enumerate(recommendations):
                    print(f"  推荐 {i+1}: {topic}")
            else:
                print(f"簇 {first_valid_cluster_id} 没有提取到关键词，无法推荐话题。")
        else:
            print("所有用户均未成功聚类，无法进行基于簇的话题推荐。")
    else:
        print("聚类信息不完整，无法进行话题推荐。")


    # 5. 可视化
    print("\n--- 5. 可视化 ---")
    all_signatures_text = " ".join(df_friends['Segmented_Signature_Str'].dropna())
    if all_signatures_text.strip():
        plot_wordcloud_from_text(all_signatures_text, "all_signatures_wordcloud.png")
    else:
        print("无有效签名文本，无法生成全局词云。")
    
    # 聚类图已在聚类部分生成 (如果条件满足)

    # 6. 保存结果
    print("\n--- 6. 保存结果 ---")
    output_csv_path = os.path.join(config.OUTPUT_DIR, "analyzed_friends_data.csv")
    try:
        df_friends.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # utf-8-sig 确保Excel打开中文不乱码
        print(f"分析结果已保存到: {output_csv_path}")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

    print("\n--- 系统运行结束 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="微信好友数据智能分析系统")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=config.FRIENDS_DATA_CSV,
        help="输入的微信好友数据CSV文件路径"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"错误: 输入文件 '{args.input_csv}' 不存在。请检查路径或文件名。")
        print(f"确保在 'data/' 目录下有名为 'simulated_wechat_friends.csv' 的文件，或者通过 --input_csv 指定正确路径。")
    else:
        main(args.input_csv)