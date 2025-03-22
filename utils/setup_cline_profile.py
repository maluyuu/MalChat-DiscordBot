#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
from pathlib import Path

class ClineProfileSetup:
    def __init__(self):
        self.home = Path.home()
        self.vscode_dir = self.home / "Library" / "Application Support" / "Code"
        self.profile_name = "Cline-Optimized"
        self.project_root = Path(__file__).parent.parent
        self.settings_file = self.project_root / ".vscode" / "cline.settings.json"

    def create_profile(self):
        """Cline用の最適化されたVSCodeプロファイルを作成"""
        print("🚀 Cline用のVSCodeプロファイルを作成中...")
        
        try:
            # プロファイルディレクトリの作成
            profile_dir = self.vscode_dir / "User" / "profiles" / self.profile_name
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # 設定ファイルの読み込みとコピー
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # プロファイル固有の設定を追加
                settings.update({
                    "window.title": "${dirty}${activeEditorShort}${separator}${rootName} [Cline]",
                    "workbench.colorTheme": "Default Dark Modern",
                    "workbench.colorCustomizations": {
                        "titleBar.activeBackground": "#2b3f55",
                        "titleBar.activeForeground": "#ffffff"
                    }
                })
                
                # 設定の保存
                settings_path = profile_dir / "settings.json"
                with open(settings_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=4)
                
                print(f"✓ 設定ファイルを保存: {settings_path}")
            
            # 拡張機能の設定
            extensions = {
                "recommendations": [
                    "saoudrizwan.cline-dev"
                ],
                "unwantedRecommendations": [
                    "ms-python.pylint",
                    "ms-python.isort",
                    "ms-python.black-formatter"
                ]
            }
            
            extensions_file = self.project_root / ".vscode" / "extensions.json"
            with open(extensions_file, 'w', encoding='utf-8') as f:
                json.dump(extensions, f, indent=4)
            
            print("✓ 拡張機能の推奨設定を保存")
            
            # プロファイル情報の作成
            profile_info = {
                "name": self.profile_name,
                "settings": str(settings_path),
                "extensions": str(extensions_file),
                "uiState": {
                    "workbench.activityBar.location": "hidden",
                    "workbench.statusBar.visible": True,
                    "workbench.sideBar.location": "right"
                }
            }
            
            profile_info_path = profile_dir / "profile.json"
            with open(profile_info_path, 'w', encoding='utf-8') as f:
                json.dump(profile_info, f, indent=4)
            
            print("✓ プロファイル情報を保存")
            
            # スタートアップスクリプトの作成
            startup_script = f"""#!/bin/bash
code --profile "{self.profile_name}" --max-memory=4096 "$@"
"""
            
            script_path = self.project_root / "start-cline.sh"
            with open(script_path, 'w') as f:
                f.write(startup_script)
            
            os.chmod(script_path, 0o755)
            print(f"✓ スタートアップスクリプトを作成: {script_path}")
            
            print("\n✨ セットアップが完了しました")
            print(f"\n使用方法:")
            print(f"1. VSCodeを再起動")
            print(f"2. コマンドパレットを開き (Cmd/Ctrl+Shift+P)")
            print(f"3. 'プロファイルの切り替え' を選択")
            print(f"4. '{self.profile_name}' を選択")
            print(f"\nまたは、以下のコマンドでプロジェクトを開く:")
            print(f"./start-cline.sh {self.project_root}")
            
        except Exception as e:
            print(f"⚠️  エラーが発生しました: {e}")

if __name__ == "__main__":
    setup = ClineProfileSetup()
    setup.create_profile()
