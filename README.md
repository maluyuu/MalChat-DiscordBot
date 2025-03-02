# malChatbot

## プロジェクト概要
- マルチモーダルな対話が可能なDiscordチャットボット
- バージョン：0.70
- 画像認識、PDF解析、ウェブ検索など多機能な処理に対応
- 複数のAIモデル（Gemini, Llama）に対応し、用途に応じて切り替え可能

## 主要機能

### 1. チャット機能 (main.py)
- 通常チャット (!malChat)
- デバッグチャット (!malDebugChat)
- システム情報の表示 (!malSys)
  - Bot情報
  - システムリソース情報
  - モデル切り替え機能
- 特殊キャラクターチャット (!malRoboKasu)

### 2. AI対話処理 (chat_processing.py)
- Gemini APIとの連携
  - Gemini 2.0シリーズ対応
    - gemini-2.0-flash：マルチモーダル対応の高速モデル
    - gemini-2.0-pro：最高品質のモデル
    - gemini-2.0-flash-lite：コスト効率の良いモデル
    - gemini-2.0-flash-thinking：思考プロセスを含む推論モデル
  - Gemini 1.5シリーズ対応
    - gemini-1.5-flash：高速なマルチモーダルモデル
    - gemini-1.5-pro：長文理解に優れたモデル
  - Gemini 1.0シリーズ対応（非推奨）
    - gemini-1.0-pro：テキストのみの処理に特化
    - gemini-1.0-pro-vision：画像・動画理解に特化
- Ollamaモデルとの連携
  - gemma2/gemma2JP
  - llama3.1/llama3.2
  - deepseek-r1シリーズ
  - phi4
  - tinySwallow

### 3. マルチモーダル処理

#### 画像処理 (image_processing.py v0.2.0)
- YOLO/OpenCVによる物体検出
  - 人物、物体、動物などの認識
  - 顔検出
  - 物体のカウントと位置特定
- OCRによるテキスト抽出
  - 日本語/英語テキストの認識
- 画像の色分析
  - 主要色の抽出
  - 色調傾向の分析
- 画像に基づいた対話生成
  - 認識結果を考慮した自然な応答
  - マルチモーダルな文脈理解

#### PDF処理 (pdf_processing.py v0.0.2)
- PDFテキスト抽出
  - テキストコンテンツの抽出
  - フォーマット保持
- 抽出テキストに基づいた対話生成
  - 文書内容の理解と応答
  - 文脈を考慮した情報提供

#### Web処理 (web_processing.py v0.3.0)
- Google Custom Search APIによる情報検索
  - キーワードベースの検索
  - 関連性の高い情報の優先取得
- カテゴリ別の情報収集
  - ニュース：主要メディアからの最新情報
  - 技術情報：開発者向けプラットフォームからの情報
  - 一般知識：Wikipedia等の信頼できるソース
  - Q&A：技術的な質問回答サイト
  - SNS：ソーシャルメディアの情報
- コンテンツの自動要約と整形
  - カテゴリごとの情報整理
  - 重要情報の抽出と構造化
- 信頼性の高いドメインからの優先的な情報取得
  - 主要ニュースサイトの優先
  - 信頼性評価に基づく情報フィルタリング

### 4. データ管理・ログ処理 (rag_log_processing.py v0.2.1)
- ベクトル検索による関連情報検索
  - SentenceTransformerによる文書ベクトル化
  - Faissによる高速検索
  - 類似度に基づく関連情報の取得
- チャット履歴の管理
  - チャンネルごとの履歴保存
  - 文脈を考慮した応答生成
  - 過去の対話内容の参照
- ログファイルの自動要約
  - 重要な対話内容の抽出
  - トピックベースの要約生成
- アップロードされたファイルの管理
  - ファイル内容のインデックス化
  - 検索可能なコンテンツ化

### 5. システム管理 (system.py v0.0.1)
- 定期的なログ要約処理
  - 10分ごとの自動要約
  - 効率的なログ管理
- システムバージョン管理
  - 各モジュールのバージョン追跡
  - 更新履歴の管理
- エラー監視
  - エラーログの記録
  - 異常検知と報告

## 環境設定

### 必要な環境変数
- DISCORD_TOKEN：Discordボットトークン
- GEMINI_API_KEY：Google Gemini APIキー
- GOOGLE_API_KEY：Google Custom Search APIキー
- GOOGLE_SEARCH_ENGINE_ID：カスタム検索エンジンID

### 依存ライブラリ
- discord.py：Discordボット機能
- google.generativeai：Gemini API連携
- ollama：ローカルAIモデル連携
- PIL：画像処理
- opencv-python：画像認識
- ultralytics：YOLO物体検出
- pytesseract：OCR処理
- sentence-transformers：テキストベクトル化
- faiss-cpu：ベクトル検索
- pdfminer：PDF解析
- beautifulsoup4：Webスクレイピング
