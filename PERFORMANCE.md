# VSCode/Cline パフォーマンス最適化ガイド

このガイドでは、VSCodeとCline拡張機能のパフォーマンスを最適化するための手順を説明します。

## クイックスタート

```bash
# 1. Cline専用プロファイルのセットアップ
./utils/setup_cline_profile.py

# 2. VSCodeを再起動し、新しいプロファイルを選択
# Cmd+Shift+P → "プロファイルの切り替え" → "Cline-Optimized"

# または、専用起動スクリプトを使用
./start-cline.sh
```

## 最適化内容

### 1. メモリ使用量の最適化
- メモリ制限の設定（4GB）
- ファイル監視の最適化
- キャッシュの効率化

### 2. パフォーマンス設定
```json
{
    "cline": {
        "experimental": {
            "enableBackgroundProcessing": false,
            "enableCaching": true,
            "cacheTTL": 3600
        },
        "completion": {
            "delay": 500,
            "maxResults": 3
        }
    }
}
```

### 3. ファイルシステムの最適化
- 不要なファイルの監視を除外
- シンボリックリンクの最適化
- キャッシュファイルの自動クリーンアップ

## 定期メンテナンス

以下のタスクを実行することで、環境を最適な状態に保つことができます：

1. **通常のクリーンアップ**
   ```bash
   # VSCodeのコマンドパレットから実行
   Tasks: クリーンアップ実行
   ```

2. **シンボリックリンクの最適化**
   ```bash
   # VSCodeのコマンドパレットから実行
   Tasks: シンボリックリンク最適化
   ```

3. **Cline環境の完全最適化**
   ```bash
   # VSCodeのコマンドパレットから実行
   Tasks: Cline環境最適化
   ```

## トラブルシューティング

### 1. パフォーマンスが低下した場合
- VSCodeを再起動
- `Extension Hosts のリセット` を実行
- キャッシュのクリア

### 2. メモリ使用量が高い場合
- 不要な拡張機能を無効化
- ワークスペースを分割
- プロセスエクスプローラーで原因を特定

### 3. Clineが応答しない場合
- APIリクエストの遅延を調整（`requestDelay`）
- キャッシュをクリア
- プロファイルを再セットアップ

## 設定ファイル

主な設定ファイルの場所：

- `/.vscode/cline.settings.json` - Cline固有の設定
- `/.vscode/settings.json` - VSCode全般の設定
- `/.vscode/extensions.json` - 推奨拡張機能の設定

## 注意事項

1. プロファイルの切り替え後は必ずVSCodeを再起動してください
2. 大規模なプロジェクトでは、ワークスペースの分割を検討してください
3. メモリ制限は環境に応じて調整してください

## 参考リンク

- [VSCode パフォーマンスガイド](https://code.visualstudio.com/docs/supporting/performance)
- [Cline ドキュメント](https://github.com/cline/cline)
- [VSCode プロファイル機能](https://code.visualstudio.com/docs/editor/profiles)
