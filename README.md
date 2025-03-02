# MalChat

## プロジェクト概要
- マルチモーダルな対話が可能なDiscordチャットボット
- バージョン：0.80
- 画像認識、PDF解析、ウェブ検索など多機能な処理に対応
- 複数のAIモデル（Gemini, Ollama）に対応し、用途に応じて切り替え可能
- 現在のデフォルトモデル: gemini-2.0-flash

## 主要機能

### 1. チャット機能 (main.py)
- 通常チャット (!malChat)
- デバッグチャット (!malDebugChat)
- システム情報の表示 (!malSys)
  - Bot情報 (!malSys about)
  - システムリソース情報 (!malSys info)
  - モデル切り替え機能 (!malSys model)
- 特殊キャラクターチャット (!malRoboKasu)
- チャンネルID指定による自動応答
- メンション応答
- ランダム応答（10%の確率）
- 長文応答の自動分割機能
- コードブロック保持機能

### 2. AI対話処理 (chat_processing.py v0.0.1)
- Gemini APIとの連携
  - Gemini 2.0シリーズ対応
    - gemini-2.0-flash：マルチモーダル対応の高速モデル
    - gemini-2.0-pro：最高品質のモデル
    - gemini-2.0-flash-lite：コスト効率の良いモデル
    - gemini-2.0-flash-thinking：思考プロセスを含む推論モデル
- Ollamaモデルとの連携
  - gemma2：Gemmaベースモデル
  - gemma2JP：日本語最適化Gemma
  - llama3.1：Llama3シリーズ
  - llama3.2：Llama3シリーズ最新版
  - deepseek-r1：Deepseekベースモデル
  - deepseek-r1:14b：14Bパラメータ版
  - deepseek-r1JP：日本語最適化版
  - phi4：軽量高性能モデル
  - tinySwallow：超軽量モデル

### 3. マルチモーダル処理

#### 画像処理 (image_processing.py v0.2.0)
- 画像認識と解析
  - YOLOv8による物体検出
  - OpenCV Haar Cascadeによる特定物体検出
    - 顔検出
    - 猫検出
    - 全身検出
  - OCRによるテキスト抽出（pytesseract）
  - 画像の色分析
    - 主要色の特定
    - 色調傾向の分析
- マルチモーダル処理
  - メモリ使用量に基づく処理方法の自動選択
    - llama3.2-visionによる直接処理
    - 特徴抽出による段階的処理

#### PDF処理 (pdf_processing.py v0.0.2)
- PDFテキスト抽出
  - PDFMinerによる高精度テキスト抽出
  - 抽出許可の確認
  - ページ単位の処理
- 安全なファイル管理
  - 一時ファイルの自動管理
  - エラーハンドリング
- 抽出テキストの活用
  - チャットコンテキストへの統合
  - 質問応答への活用

#### Web処理 (web_processing.py v0.3.0)
- Google Custom Search APIによる情報検索
  - キーワードベースの検索
  - 関連性の高い情報の優先取得
- インテリジェント情報収集
  - 情報源のカテゴリ分類
    - ニュース：主要メディアからの情報
    - 技術情報：開発者向けプラットフォーム
    - 一般知識：Wikipedia等
    - Q&A：技術的な質問回答
    - SNS：ソーシャルメディア
  - カテゴリごとの情報取得制限
  - 信頼性評価に基づく情報フィルタリング
- コンテンツ抽出と加工
  - メタデータの抽出
    - タイトル
    - タイムスタンプ
    - コンテンツタイプ
  - 本文の抽出と最適化
    - 不要要素の除去
    - 長文の適切な要約

### 4. データ管理・ログ処理 (rag_log_processing.py v0.2.1)
- チャット履歴の管理
  - チャンネルごとの履歴保存
  - JSONフォーマットでの永続化
- ベクトル検索システム
  - SentenceTransformerによる文書ベクトル化
  - Faissによる高速類似度検索
  - チャンク単位での文書管理
- ファイル管理システム
  - 添付ファイルの自動処理
  - コンテンツのインデックス化
  - 文書の自動チャンク分割
- 履歴要約システム
  - トピックベースの会話要約
  - 定期的な要約更新
  - 最新10件の会話の構造化

### 5. システム管理 (system.py v0.0.1)
- システム情報管理
  - CPUリソース監視
  - メモリ使用状況監視
  - 各モジュールのバージョン管理
- ログ管理
  - 10分ごとの自動ログ要約
  - エラーログの記録
  - 要約ログの生成と保存

## 環境設定

### 必要な環境変数
- DISCORD_TOKEN：Discordボットトークン
- CHAN_ID：自動応答を有効にするチャンネルID（カンマ区切りで複数指定可能）
- GEMINI_API_KEY：Google Gemini APIキー
- GOOGLE_API_KEY：Google Custom Search APIキー
- GOOGLE_SEARCH_ENGINE_ID：カスタム検索エンジンID

### 依存ライブラリ
#### 基本ライブラリ
- discord.py：Discordボット機能
- python-dotenv：環境変数管理
- requests：HTTP通信
- beautifulsoup4：Webスクレイピング

#### AI・機械学習
- google.generativeai：Gemini API連携
- ollama：ローカルAIモデル連携
- sentence-transformers：テキストベクトル化
- faiss-cpu：ベクトル検索

#### 画像処理
- PIL（Pillow）：基本的な画像処理
- opencv-python：高度な画像処理
- ultralytics：YOLOv8物体検出
- pytesseract：OCRテキスト抽出

#### PDF処理
- pdfminer：PDFテキスト抽出

#### システム管理
- psutil：システムリソース監視

#### データ処理
- numpy：数値計算
- pickle：オブジェクトシリアライズ

#### ログ管理
- logging：ログ機能
