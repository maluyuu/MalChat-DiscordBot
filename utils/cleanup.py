#!/usr/bin/env python3
import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta

class ProjectCleaner:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
            "**/.pytest_cache",
            "**/.coverage",
            "**/.DS_Store"
        ]
        self.archive_patterns = [
            "**/logs/*.log",
            "**/logs/*.json",
            "**/data/vectors/*.faiss",
            "**/data/vectors/*.pkl"
        ]

    def cleanup_cache_files(self):
        """キャッシュファイルとディレクトリの削除"""
        print("🧹 キャッシュファイルのクリーンアップを開始...")
        for pattern in self.cleanup_patterns:
            for path in self.project_root.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    print(f"✓ 削除完了: {path.relative_to(self.project_root)}")
                except Exception as e:
                    print(f"⚠️  削除失敗: {path.relative_to(self.project_root)} - {e}")

    def archive_old_files(self, days=7):
        """古いログファイルのアーカイブ"""
        print(f"\n📦 {days}日以上前のファイルをアーカイブ中...")
        cutoff = datetime.now() - timedelta(days=days)
        
        for pattern in self.archive_patterns:
            for path in self.project_root.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff.timestamp():
                        archive_dir = path.parent / "archive"
                        archive_dir.mkdir(exist_ok=True)
                        
                        new_name = f"{path.stem}_{datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d')}{path.suffix}"
                        archive_path = archive_dir / new_name
                        
                        shutil.move(str(path), str(archive_path))
                        print(f"✓ アーカイブ完了: {path.relative_to(self.project_root)} → {archive_path.relative_to(self.project_root)}")
                except Exception as e:
                    print(f"⚠️  アーカイブ失敗: {path.relative_to(self.project_root)} - {e}")

    def compress_archives(self):
        """アーカイブディレクトリの圧縮"""
        print("\n🗜️  アーカイブの圧縮を開始...")
        for archive_dir in self.project_root.glob("**/archive"):
            if not any(archive_dir.iterdir()):
                continue

            try:
                archive_name = f"{archive_dir.parent.name}_archive_{datetime.now().strftime('%Y%m%d')}.zip"
                shutil.make_archive(
                    str(archive_dir.parent / archive_name.replace('.zip', '')),
                    'zip',
                    archive_dir
                )
                print(f"✓ 圧縮完了: {archive_name}")
                
                # 圧縮後のファイルを削除
                shutil.rmtree(archive_dir)
                print(f"✓ 元ファイル削除完了: {archive_dir.relative_to(self.project_root)}")
            except Exception as e:
                print(f"⚠️  圧縮失敗: {archive_dir.relative_to(self.project_root)} - {e}")

def main():
    project_root = Path(__file__).parent.parent
    cleaner = ProjectCleaner(project_root)
    
    print("🚀 プロジェクトクリーンアップを開始します")
    print(f"📂 プロジェクトルート: {project_root}\n")
    
    cleaner.cleanup_cache_files()
    cleaner.archive_old_files()
    cleaner.compress_archives()
    
    print("\n✨ クリーンアップが完了しました")

if __name__ == "__main__":
    main()
