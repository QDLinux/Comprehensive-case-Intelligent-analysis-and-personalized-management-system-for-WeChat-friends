# web_scraper.py
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import quote # 用于URL编码中文字符
from config import USER_AGENT

def scrape_baidubaike_summary(keyword, num_sentences=2, retries=2, delay=1):
    """
    从百度百科爬取关键词的摘要信息。
    注意：百度百科的页面结构可能会变化，导致爬虫失效。
    """
    search_url = f"https://baike.baidu.com/item/{quote(keyword)}"
    headers = {'User-Agent': USER_AGENT}
    summary_texts = []

    for attempt in range(retries):
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status() # 如果请求失败则抛出HTTPError
            soup = BeautifulSoup(response.content, 'html.parser')

            # 百度百科摘要通常在 class="lemma-summary" 或 class="lemmaWgt-lemmaSummary" 的div中
            summary_div = soup.find('div', class_=lambda x: x and ('lemma-summary' in x or 'lemmaWgt-lemmaSummary' in x))

            if summary_div:
                paragraphs = summary_div.find_all('div', class_='para')
                full_summary = ""
                for p in paragraphs:
                    full_summary += p.get_text(strip=True)
                
                if full_summary:
                    # 简单按句号分割，取前 N 句
                    sentences = [s.strip() for s in full_summary.split('。') if s.strip()]
                    summary_texts = [s + "。" for s in sentences[:num_sentences]]
                    print(f"成功爬取到 '{keyword}' 的百度百科摘要。")
                    return " ".join(summary_texts)
                else:
                    print(f"未能在页面中找到 '{keyword}' 的摘要内容。")
                    return f"未找到'{keyword}'的百科摘要。"

            else: # 尝试搜索结果页
                search_url_fallback = f"https://baike.baidu.com/search/word?word={quote(keyword)}"
                response_fallback = requests.get(search_url_fallback, headers=headers, timeout=10)
                response_fallback.raise_for_status()
                soup_fallback = BeautifulSoup(response_fallback.content, 'html.parser')
                
                first_result = soup_fallback.find('a', class_='result-title')
                if first_result and first_result.get('href'):
                    print(f"'{keyword}' 直接页面未找到，尝试第一个搜索结果: {first_result.get('href')}")
                    # 递归调用，但只尝试一次，防止无限循环
                    if attempt == 0: # 只在第一次尝试失败后用这个逻辑
                        time.sleep(delay) # 礼貌等待
                        return scrape_baidubaike_summary(first_result.get_text(strip=True), num_sentences, retries=1) # keyword用title
                    else:
                        return f"未找到'{keyword}'的百科摘要。"

            print(f"未找到 '{keyword}' 的百度百科摘要区域。")
            return f"未找到'{keyword}'的百科摘要。"

        except requests.exceptions.RequestException as e:
            print(f"爬取 '{keyword}' 时发生网络错误 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1)) # 增加等待时间
            else:
                return f"爬取'{keyword}'失败: 网络错误。"
        except Exception as e:
            print(f"爬取 '{keyword}' 时发生未知错误 (尝试 {attempt + 1}/{retries}): {e}")
            return f"爬取'{keyword}'失败: {e}"
    return f"多次尝试后仍未能爬取'{keyword}'的百科摘要。"


def scrape_generic_news_headlines(topic_keyword, num_headlines=3):
    """
    (示例) 从一个通用新闻搜索（如百度新闻）爬取相关新闻标题。
    注意：这只是一个非常基础的示例，实际新闻爬虫会更复杂。
    """
    search_url = f"https://www.baidu.com/s?wd={quote(topic_keyword + ' 最新资讯')}&tn=news" # 百度新闻搜索
    headers = {'User-Agent': USER_AGENT}
    headlines = []

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # 百度新闻标题通常在特定class的<a>标签内 (这个选择器需要根据实际页面调整)
        # 例如: class="news-title_1YtI1 " 或者 result下的h3 > a
        results = soup.find_all('div', class_='result-op', limit=num_headlines + 2) # 多取几个以防有非新闻项
        if not results: # 尝试另一种可能的选择器
             results = soup.find_all('div', class_='result', limit=num_headlines + 2)


        for res in results:
            title_tag = res.find('h3', class_=lambda x: x and 'title' in x.lower()) # 找 h3 且 class含title
            if not title_tag: # 再尝试找class带c-title的a标签
                title_tag = res.find('a', class_=lambda x: x and 'c-title' in x.lower())
            
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                if title_text:
                    headlines.append(title_text)
                if len(headlines) >= num_headlines:
                    break
        
        if headlines:
            print(f"成功爬取到 '{topic_keyword}' 的相关新闻标题: {headlines}")
            return headlines
        else:
            print(f"未能爬取到 '{topic_keyword}' 的相关新闻标题。")
            return [f"未找到与'{topic_keyword}'相关的新闻。"]

    except requests.exceptions.RequestException as e:
        print(f"爬取新闻 '{topic_keyword}' 时发生网络错误: {e}")
        return [f"爬取'{topic_keyword}'新闻失败: 网络错误。"]
    except Exception as e:
        print(f"爬取新闻 '{topic_keyword}' 时发生未知错误: {e}")
        return [f"爬取'{topic_keyword}'新闻失败: {e}。"]


if __name__ == '__main__':
    print("--- 测试百度百科爬虫 ---")
    keyword1 = "Python"
    summary1 = scrape_baidubaike_summary(keyword1, num_sentences=1)
    print(f"\n摘要 for '{keyword1}': {summary1}")

    time.sleep(1) # 友好等待

    keyword2 = "自然语言处理"
    summary2 = scrape_baidubaike_summary(keyword2, num_sentences=1)
    print(f"\n摘要 for '{keyword2}': {summary2}")

    time.sleep(1)

    keyword3 = "一个不存在的词条XYZABC" # 测试找不到的情况
    summary3 = scrape_baidubaike_summary(keyword3)
    print(f"\n摘要 for '{keyword3}': {summary3}")

    print("\n--- 测试新闻标题爬虫 ---")
    topic = "人工智能"
    news = scrape_generic_news_headlines(topic, num_headlines=2)
    print(f"\n新闻 for '{topic}': {news}")