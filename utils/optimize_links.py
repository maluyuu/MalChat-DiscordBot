#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime

class SymlinkOptimizer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.settings_file = self.project_root / '.vscode' / 'settings.json'
        self.symlinks = []
        self.common_dirs = set()

    def find_symlinks(self):
        """プロジェクト内のシンボリックリンクを検索"""
        print("🔍 シンボリックリンクを検索中...")
        for path in self.project_root.rglob('*'):
            try:
                if path.is_symlink():
                    target = Path(os.path.realpath(path))
                    self.symlinks.append((path, target))
                    print(f"✓ 発見: {path.relative_to(self.project_root)} → {target}")
                    
                    # 共通ディレクトリの特定
                    if target.is_dir():
                        self.common_dirs.add(str(target))
            except Exception as e:
                print(f"⚠️  エラー: {path} - {e}")

    def update_vscode_settings(self):
        """VSCode設定の更新"""
        print("\n⚙️ VSCode設定を更新中...")
        try:
            if not self.settings_file.exists():
                print("⚠️  settings.jsonが見つかりません")
                return

            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # extraPathsの更新
            settings['python.analysis.extraPaths'] = list(self.common_dirs)
            
            # 除外設定の更新
            if not settings.get('python.analysis.exclude'):
                settings['python.analysis.exclude'] = []
            
            # バックアップの作成
            backup_file = self.settings_file.with_suffix('.json.bak')
            shutil.copy2(self.settings_file, backup_file)
            print(f"✓ 設定バックアップ作成: {backup_file}")

            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            print("✓ settings.jsonを更新しました")

        except Exception as e:
            print(f"⚠️  設定更新エラー: {e}")

    def generate_report(self):
        """最適化レポートの生成"""
        report_file = self.project_root / 'symlinks_report.md'
        print("\n📝 レポートを生成中...")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"# シンボリックリンク最適化レポート\n\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 検出されたシンボリックリンク\n\n")
                for link, target in self.symlinks:
                    f.write(f"- {link.relative_to(self.project_root)} → {target}\n")
                
                f.write("\n## 共通ディレクトリ\n\n")
                for common_dir in self.common_dirs:
                    f.write(f"- {common_dir}\n")
                
                f.write("\n## 推奨事項\n\n")
                f.write("1. 共通ディレクトリは `python.analysis.extraPaths` に追加されました\n")
                f.write("2. シンボリックリンクの使用を最小限に抑えることを推奨します\n")
                f.write("3. 可能な場合は、Pythonのパッケージング機能の使用を検討してください\n")
            
            print(f"✓ レポート生成完了: {report_file}")
        except Exception as e:
            print(f"⚠️  レポート生成エラー: {e}")

def main():
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path(__file__).parent.parent

    print(f"🚀 シンボリックリンク最適化を開始します")
    print(f"📂 プロジェクトルート: {project_root}\n")

    optimizer = SymlinkOptimizer(project_root)
    optimizer.find_symlinks()
    optimizer.update_vscode_settings()
    optimizer.generate_report()

    print("\n✨ 最適化が完了しました")

if __name__ == "__main__":
    main()
