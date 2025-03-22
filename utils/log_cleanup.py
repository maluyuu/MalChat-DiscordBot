import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_logs(logs_dir: str, max_age_days: int = 7):
    """古いログファイルをアーカイブし、不要なファイルを削除します"""
    logs_path = Path(logs_dir)
    archive_path = logs_path / "archive"
    archive_path.mkdir(exist_ok=True)

    current_time = datetime.now()
    
    try:
        for file_path in logs_path.glob("**/*"):
            if not file_path.is_file():
                continue
                
            # アーカイブディレクトリ内のファイルはスキップ
            if "archive" in str(file_path):
                continue

            # ファイルの最終更新時刻を取得
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            age = current_time - modified_time

            # 特定の拡張子のファイルのみを処理
            if file_path.suffix in [".log", ".json"] and age > timedelta(days=max_age_days):
                # アーカイブフォルダに移動
                archive_file = archive_path / f"{file_path.stem}_{modified_time.strftime('%Y%m%d')}{file_path.suffix}"
                shutil.move(str(file_path), str(archive_file))
                logging.info(f"Archived old log file: {file_path.name} -> {archive_file.name}")

        # 空のPycacheディレクトリを削除
        for pycache in logs_path.glob("**/__pycache__"):
            if pycache.is_dir():
                try:
                    shutil.rmtree(pycache)
                    logging.info(f"Removed pycache directory: {pycache}")
                except Exception as e:
                    logging.error(f"Failed to remove pycache directory {pycache}: {e}")

    except Exception as e:
        logging.error(f"Error during log cleanup: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    cleanup_logs(str(logs_dir))
