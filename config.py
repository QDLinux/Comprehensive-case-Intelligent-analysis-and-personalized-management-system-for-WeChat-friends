# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

FRIENDS_DATA_CSV = os.path.join(DATA_DIR, 'simulated_wechat_friends.csv')
STOPWORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')

# 字体路径，用于matplotlib和wordcloud显示中文
# 请确保你的系统中有这个字体，或者替换为你系统中的中文字体路径
# Windows: C:/Windows/Fonts/simhei.ttf (黑体), simsun.ttc (宋体)
# MacOS: /System/Library/Fonts/PingFang.ttc
# Linux:
#   查找: fc-list :lang=zh
#   例如: /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
FONT_PATH = "simhei.ttf" # 这是一个常见选择，如果找不到请替换
try:
    # 尝试加载字体以进行早期检查，尽管这不会保证matplotlib/wordcloud能找到它
    # 真正的检查发生在visualizer模块
    from matplotlib.font_manager import findfont, FontProperties
    if not findfont(FontProperties(fname=FONT_PATH)):
        print(f"警告: 字体 '{FONT_PATH}' 未在matplotlib的默认搜索路径中找到。")
        print("请确保 FONT_PATH 指向一个有效的中文字体文件(.ttf, .otf, .ttc)。")
        print("如果可视化出现中文乱码，请修改 config.py 中的 FONT_PATH。")
except ImportError:
    print("警告: matplotlib 未安装，无法检查字体。")
except Exception as e:
    print(f"检查字体时出错: {e}")


# 爬虫相关 (如果需要)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# 聚类簇数量
NUM_CLUSTERS = 3 #可以根据实际数据调整