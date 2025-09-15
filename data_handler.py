# data_handler.py
import pandas as pd
import re

def load_wechat_csv(file_path):
    """加载并初步清洗微信导出的CSV数据。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # 确保关键列存在，用空字符串填充缺失的签名
        if 'Signature' not in df.columns:
            df['Signature'] = ""
        else:
            df['Signature'] = df['Signature'].fillna('')

        if 'Simulated_Last_Activity_Text' not in df.columns:
            df['Simulated_Last_Activity_Text'] = ""
        else:
            df['Simulated_Last_Activity_Text'] = df['Simulated_Last_Activity_Text'].fillna('')

        print(f"成功加载数据: {file_path}, 共 {len(df)} 条记录。")
        return df
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        return None

def clean_text(text):
    """文本清洗函数：去除非中文字符、字母、数字，转换为小写，去多余空格。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text) # 去除URL
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text) # 保留中文、英文、数字和空格
    text = text.lower() # 转换为小写
    text = re.sub(r'\s+', ' ', text).strip() # 去除多余空格
    return text

if __name__ == '__main__':
    # 测试
    from config import FRIENDS_DATA_CSV
    df_friends = load_wechat_csv(FRIENDS_DATA_CSV)
    if df_friends is not None:
        print("\n原始数据示例:")
        print(df_friends.head())

        # 清洗签名
        df_friends['Cleaned_Signature'] = df_friends['Signature'].apply(clean_text)
        print("\n清洗后签名示例:")
        print(df_friends[['Signature', 'Cleaned_Signature']].head())