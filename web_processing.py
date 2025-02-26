from typing import List, Dict, Optional
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import requests
import os
from config.constants import TOPICS, CONTENT_DOMAINS
from utils.logger import setup_logger

logger = setup_logger(__name__, 'web_processing.log')

VERSION = '0.3.0'

async def version():
    return VERSION

async def get_links(query: str) -> List[str]:
    try:
        search_url = await _build_search_url(query)
        if not search_url:
            logger.error("Failed to build search URL")
            return []
            
        response = requests.get(search_url)
        response.raise_for_status()  # エラーチェックを追加
        
        data = response.json()
        if 'items' not in data:
            logger.error(f"No search results found: {data.get('error', {}).get('message', 'Unknown error')}")
            return []
            
        return [item['link'] for item in data['items']]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting links: {e}")
        return []

async def extract_data_from_links(links: List[str], query: str) -> str:
    results = []
    retry_count = 0
    max_retries = 20  # 最大再試行回数を増加
    
    # 情報源のカテゴリ分け
    source_categories = {
        'news': [
            'news.yahoo.co.jp',
            'www3.nhk.or.jp',
            'www.asahi.com',
            'mainichi.jp',
            'www.yomiuri.co.jp',
            'www.nikkei.com',
            'jp.reuters.com',
            'www.jiji.com',
            'kyodo.jp'
        ],
        'tech': [
            'qiita.com',
            'zenn.dev',
            'github.com',
            'stackoverflow.com',
            'dev.to'
        ],
        'knowledge': [
            'wikipedia.org',
            'hatena.ne.jp',
            'note.com',
            'medium.com'
        ],
        'qa': [
            'teratail.com',
            'detail.chiebukuro.yahoo.co.jp',
            'oshiete.goo.ne.jp'
        ],
        'social': [
            'twitter.com',
            'x.com',
            'facebook.com',
            'instagram.com',
            'linkedin.com'
        ]
    }

    # 各カテゴリから情報を取得
    for category, domains in source_categories.items():
        category_count = 0
        max_per_category = 2  # 各カテゴリから最大2つまで取得
        
        for link in links:
            if category_count >= max_per_category:
                break
                
            if any(domain in link for domain in domains):
                try:
                    content = await _extract_content(link, category, query)
                    if content and content.get('content'):
                        content['category'] = category
                        results.append(content)
                        category_count += 1
                except Exception as e:
                    logger.error(f"Error processing {category} link {link}: {e}")
                    continue

    # 一般ドメインの処理（カテゴリに含まれないドメイン）
    if len(results) < 10:  # 総数を10つまで増加
        for link in links:
            if retry_count >= max_retries:
                break
                
            if any(result['url'] == link for result in results):
                continue
                
            try:
                content = await _extract_content(link, 'general', query)
                if content and content.get('content'):
                    content['category'] = 'general'
                    results.append(content)
                    if len(results) >= 10:
                        break
                else:
                    retry_count += 1
            except Exception as e:
                logger.error(f"Error processing link {link}: {e}")
                retry_count += 1

    # 結果の組み合わせと要約
    combined_content = await _combine_search_results(results, query)
    return combined_content

async def _combine_search_results(results: List[Dict], query: str) -> str:
    try:
        # カテゴリごとにグループ化
        categorized_results = {}
        for result in results:
            category = result.get('category', 'general')
            if category not in categorized_results:
                categorized_results[category] = []
            categorized_results[category].append(result)
        
        # 結果を組み合わせて整形
        combined_text = f"「{query}」に関する検索結果:\n\n"
        
        # カテゴリごとに結果を表示
        category_names = {
            'news': 'ニュース',
            'tech': '技術情報',
            'knowledge': '一般知識',
            'qa': 'Q&A情報',
            'social': 'SNS情報',
            'general': 'その他の情報'
        }
        
        for category, items in categorized_results.items():
            if items:
                combined_text += f"=== {category_names.get(category, category)} ===\n"
                for idx, result in enumerate(items, 1):
                    title = result.get('title', '不明なタイトル')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    timestamp = result.get('timestamp', '')
                    
                    # 長すぎる内容は適切な長さに切り詰める
                    max_content_length = 800
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    
                    combined_text += f"[情報源 {idx}]\n"
                    combined_text += f"タイトル: {title}\n"
                    if timestamp:
                        combined_text += f"日時: {timestamp}\n"
                    combined_text += f"内容: {content}\n"
                    combined_text += f"URL: {url}\n\n"
        
        return combined_text
        
    except Exception as e:
        logger.error(f"Error combining search results: {e}")
        return "検索結果の処理中にエラーが発生しました。"

async def _extract_content(url: str, content_type: str, query: str) -> Optional[Dict]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # エンコーディングの自動検出を試みる
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # メタデータの抽出を強化
        title = await _extract_title(soup)
        timestamp = await _extract_timestamp(soup)
        
        # コンテンツ抽出の改善
        content = await _extract_main_content(soup)
        
        if content and len(content) > 50:
            return {
                'url': url,
                'title': title,
                'content': content,
                'type': content_type,
                'timestamp': timestamp
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None

async def _extract_title(soup: BeautifulSoup) -> str:
    """ページタイトルを抽出する改善版関数"""
    title = None
    
    # 優先順位付きでタイトルを探す
    selectors = [
        ('h1', {'class': ['article-title', 'entry-title', 'title']}),
        ('meta', {'property': 'og:title'}),
        ('title', {}),
    ]
    
    for tag, attrs in selectors:
        element = soup.find(tag, attrs)
        if element:
            if tag == 'meta':
                title = element.get('content')
            else:
                title = element.get_text(strip=True)
            break
    
    return title or '不明なタイトル'

async def _extract_timestamp(soup: BeautifulSoup) -> str:
    """タイムスタンプを抽出する関数"""
    timestamp = None
    
    # 一般的な日付要素のセレクタ
    selectors = [
        ('time', {}),
        ('meta', {'property': 'article:published_time'}),
        ('span', {'class': ['date', 'time', 'published']}),
    ]
    
    for tag, attrs in selectors:
        element = soup.find(tag, attrs)
        if element:
            if tag == 'meta':
                timestamp = element.get('content')
            else:
                timestamp = element.get_text(strip=True)
            break
    
    return timestamp or datetime.now().isoformat()

async def _extract_main_content(soup: BeautifulSoup) -> str:
    """メインコンテンツを抽出する改善版関数"""
    # 不要な要素を削除
    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # コンテンツ候補の優先順位付きリスト
    content_candidates = []
    
    # 記事本文を探す
    article = soup.find('article')
    if article:
        content_candidates.append(article.get_text(strip=True))
    
    # メインコンテンツエリアを探す
    main_content = soup.find('main') or soup.find('div', {'id': 'main-content'})
    if main_content:
        content_candidates.append(main_content.get_text(strip=True))
    
    # 段落を収集
    paragraphs = soup.find_all('p')
    if paragraphs:
        content_candidates.append(' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20))
    
    # 最も長いコンテンツを選択
    if content_candidates:
        return max(content_candidates, key=len)
    
    return ''

async def _determine_content_type(url: str) -> str:
    for content_type, domains in CONTENT_DOMAINS.items():
        if any(domain in url for domain in domains):
            return content_type
    return 'news'

async def _build_search_url(query: str) -> str:
    try:
        # Google Custom Search APIのURLを構築
        base_url = "https://www.googleapis.com/customsearch/v1"
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        if not api_key or not search_engine_id:
            logger.error("Google API credentials not found in environment variables")
            return ""
        
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query
        }
        
        # URLエンコーディングを行う
        from urllib.parse import urlencode
        return f"{base_url}?{urlencode(params)}"
        
    except Exception as e:
        logger.error(f"Error building search URL: {e}")
        return ""

async def _is_reliable_domain(url: str) -> bool:
    # 信頼できるドメインかどうかをチェック
    trusted_domains = [
        'news.yahoo.co.jp',
        'www3.nhk.or.jp',
        'www.asahi.com',
        'mainichi.jp',
        'www.yomiuri.co.jp',
        'www.nikkei.com'
    ]
    return any(domain in url for domain in trusted_domains)