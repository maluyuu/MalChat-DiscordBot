import logging
import os
from typing import Optional
from config.constants import LOG_DIR

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.DEBUG) -> logging.Logger:
    """ロガーのセットアップを行う"""
    
    # logsディレクトリが存在しない場合は作成
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ハンドラーが既に設定されている場合は追加しない
    if not logger.handlers:
        # コンソール出力用のハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # ファイル出力用のハンドラー
        if log_file:
            # ログファイルのパスを logs ディレクトリ内に設定
            log_path = os.path.join(LOG_DIR, log_file)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
    return logger