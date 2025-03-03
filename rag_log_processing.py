
from typing import List, Dict, Optional
from datetime import datetime, timezone
import os
import json
from config.constants import LOG_DIR, CHAT_LOG_FILE, FOR_ANSWER_PATH
from utils.logger import setup_logger
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

VERSION = '0.2.1'

async def version():
    return VERSION

logger = setup_logger(__name__, 'rag_processing.log')

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.index = faiss.IndexFlatL2(384)  # モデルの出力次元数
        self.documents = []
        self.load_index()

    def load_index(self):
        index_path = os.path.join(LOG_DIR, 'document_index.faiss')
        docs_path = os.path.join(LOG_DIR, 'documents.pkl')
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)

    def save_index(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        faiss.write_index(self.index, os.path.join(LOG_DIR, 'document_index.faiss'))
        with open(os.path.join(LOG_DIR, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)

    async def process_file(self, file_content: str, file_name: str, file_type: str) -> None:
        # ファイルの内容をチャンクに分割
        chunks = self._split_into_chunks(file_content)
        
        for chunk in chunks:
            # チャンクのベクトル化
            embedding = self.model.encode(chunk)
            
            # インデックスに追加
            self.index.add(np.array([embedding]))
            
            # ドキュメント情報を保存
            self.documents.append({
                'content': chunk,
                'file_name': file_name,
                'file_type': file_type,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        self.save_index()

    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    async def search_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        # インデックスが空の場合は空のリストを返す
        if self.index.ntotal == 0:
            return []
        
        query_vector = self.model.encode(query)
        D, I = self.index.search(np.array([query_vector]), k)
        
        relevant_docs = []
        # I[0]が存在することを確認
        if len(I) > 0 and len(I[0]) > 0:
            for idx in I[0]:
                if 0 <= idx < len(self.documents):  # インデックスの範囲チェックを追加
                    relevant_docs.append(self.documents[idx])
        
        return relevant_docs

class ChatHistoryManager:
    def __init__(self):
        self.chat_histories: Dict = {}
        self.document_processor = DocumentProcessor()
        self.bot_roles = {'MalChat'}  # ボットのロール名を設定

    async def process_uploaded_files(self, files: List[Dict]) -> None:
        for file in files:
            await self.document_processor.process_file(
                file['content'],
                file['name'],
                file['type']
            )

    async def get_relevant_context(self, query: str, current_files: Optional[List[Dict]] = None) -> Optional[str]:
        # 現在送信されたファイルがない場合はNoneを返す
        if not current_files:
            return None
            
        try:
            context = []
            for file in current_files:
                context.append(f"ファイル '{file['name']}' の内容:\n{file['content']}")
            
            return '\n\n'.join(context)
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return None

    async def add_entry(self, role: str, content: str, channel_id: Optional[str] = None) -> None:
        try:
            entry = {
                'role': role,
                'content': content,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'channel_id': channel_id
            }
            
            log_path = self._get_log_path(channel_id)
            await self._write_to_log(entry, log_path)
            
        except Exception as e:
            logger.error(f"Error adding chat entry: {e}")
            raise

    async def get_relevant_history(self, query: str, channel_id: Optional[str] = None) -> List[Dict]:
        try:
            history = await self._read_history(channel_id)
            return await self._search_related(query, history)
        except Exception as e:
            logger.error(f"Error getting relevant history: {e}")
            return []

    async def write_log_file(self, role: str, content: str, channel_id: Optional[str] = None) -> None:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            timestamp = datetime.now(timezone.utc).isoformat()
            
            log_entry = {
                'timestamp': timestamp,
                'role': role,
                'content': content,
                'channel_id': channel_id
            }
            
            # チャンネル固有のログファイルを使用
            log_file = os.path.join(LOG_DIR, f'chat_log_{channel_id}.json' if channel_id else CHAT_LOG_FILE)
            
            # 既存のログを読み込む
            existing_logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            
            # 新しいエントリを追加
            existing_logs.append(log_entry)
            
            # ログを書き込む
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
            
            # 要約ログも更新
            await self._update_summary_log(existing_logs)
                
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
            raise

    async def _update_summary_log(self, logs: List[Dict]) -> None:
        try:
            summary_file = os.path.join(LOG_DIR, 'summarized_log.txt')
            
            # 最新の会話履歴から要約を生成
            summary = "# 会話履歴の要約\n\n"
            
            # 最新の会話をループ化して要約
            current_topic = None
            topic_messages = []
            
            for log in logs[-10:]:  # 最新の10件を処理
                if log['role'] not in self.bot_roles:  # ボット以外のメッセージを新しいトピックとして扱う
                    if current_topic and topic_messages:
                        summary += f"## {current_topic}\n"
                        for msg in topic_messages:
                            summary += f"- {msg}\n"
                        summary += "\n"
                    current_topic = log['content']
                    topic_messages = []
                else:
                    topic_messages.append(log['content'])
            
            # 最後のトピックを追加
            if current_topic and topic_messages:
                summary += f"## {current_topic}\n"
                for msg in topic_messages:
                    summary += f"- {msg}\n"
            
            # 要約を保存
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
                
        except Exception as e:
            logger.error(f"Error updating summary log: {e}")
            raise

    @staticmethod
    async def read_log_file(filename: str) -> str:
        try:
            file_path = os.path.join(LOG_DIR, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return ""

    def _get_log_path(self, channel_id: Optional[str]) -> str:
        return os.path.join(LOG_DIR, f'chat_log_{channel_id}.json' if channel_id else CHAT_LOG_FILE)

    async def _write_to_log(self, entry: Dict, log_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            existing_logs = []
            
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
                    
            existing_logs.append(entry)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error writing to log: {e}")
            raise

    async def _read_history(self, channel_id: Optional[str]) -> List[Dict]:
        try:
            log_path = self._get_log_path(channel_id)
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error reading history: {e}")
            return []

    async def _calculate_relevance_scores(self, query: str, history: List[Dict]) -> List[tuple[float, Dict]]:
        """
        履歴エントリーの関連性スコアを計算する
        """
        if not history:
            return []

        # クエリのエンベディングを計算
        query_embedding = self.document_processor.model.encode(query)

        # 各履歴エントリーのスコアを計算
        scored_entries = []
        for entry in history:
            if entry.get('content'):
                # コンテンツのエンベディングを計算
                content_embedding = self.document_processor.model.encode(entry['content'])
                
                # コサイン類似度を計算
                similarity = np.dot(query_embedding, content_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                )
                
                scored_entries.append((similarity, entry))

        # スコアの降順でソート
        return sorted(scored_entries, key=lambda x: x[0], reverse=True)

    async def _get_recent_history(self, history: List[Dict], count: int) -> List[Dict]:
        """
        最新の履歴を取得する
        """
        return history[-count:] if history else []

    async def _merge_and_deduplicate(self, related: List[Dict], recent: List[Dict]) -> List[Dict]:
        """
        関連履歴と最新履歴をマージして重複を除去する
        """
        seen = set()
        merged = []
        
        # 関連履歴を追加
        for entry in related:
            content_hash = hashlib.md5(entry['content'].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append(entry)
        
        # 最新履歴を追加（重複を除く）
        for entry in recent:
            content_hash = hashlib.md5(entry['content'].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                merged.append(entry)
        
        return merged

    async def get_combined_history(
        self,
        query: str,
        channel_id: Optional[str] = None,
        related_count: int = 3,
        recent_count: int = 3
    ) -> List[Dict]:
        """
        関連性の高い履歴と最新の履歴を組み合わせて取得する
        """
        try:
            # チャンネルの履歴を取得
            history = await self._read_history(channel_id)
            if not history:
                return []

            # 関連性スコアを計算
            scored_entries = await self._calculate_relevance_scores(query, history)
            related_history = [entry for _, entry in scored_entries[:related_count]]

            # 最新の履歴を取得
            recent_history = await self._get_recent_history(history, recent_count)

            # 結果をマージして重複を除去
            return await self._merge_and_deduplicate(related_history, recent_history)

        except Exception as e:
            logger.error(f"Error getting combined history: {e}")
            return []

    async def _search_related(self, query: str, history: List[Dict]) -> List[Dict]:
        """
        クエリに関連する履歴を検索する
        """
        try:
            # get_combined_historyを使用して関連履歴を取得
            return await self.get_combined_history(query, None, related_count=3, recent_count=3)
        except Exception as e:
            logger.error(f"Error in _search_related: {e}")
            return []

# グローバルなインスタンスを作成
chat_history_manager = ChatHistoryManager()
