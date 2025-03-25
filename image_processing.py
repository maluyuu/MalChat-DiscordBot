from PIL import ImageStat, Image
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import ollama
import rag_log_processing
import psutil
import base64
from io import BytesIO
from utils.logger import setup_logger
from chat_processing import chat_with_model
from typing import List, Dict, Optional
import re
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

VERSION = '0.3.0'

# ロガーの設定
logger = setup_logger(__name__, 'image_processing.log')

class ImageGenSettings:
    def __init__(self):
        self.aspect_ratio = "1:1"
        self.number_of_images = 1
        self.safety_filter_level = "BLOCK_MEDIUM_AND_ABOVE"
        self.person_generation = "ALLOW_ADULT"

    def to_dict(self) -> Dict:
        return {
            "aspect_ratio": self.aspect_ratio,
            "number_of_images": self.number_of_images,
            "safety_filter_level": self.safety_filter_level,
            "person_generation": self.person_generation
        }

class ImageGenTrigger:
    def __init__(self):
        self.visual_keywords = [
            "描いて", "見せて", "生成して", "表示して",
            "イメージ", "画像", "絵", "図",
            "ビジュアル", "グラフィック"
        ]
        self.context_patterns = [
            r"どんな(様子|感じ)",
            r"(どう|どのように)見える",
            r"視覚化",
            r"想像して",
            r"例を示して"
        ]

    def should_generate(self, text: str) -> bool:
        # キーワードによる判定
        if any(keyword in text for keyword in self.visual_keywords):
            return True
        
        # パターンによる判定
        if any(re.search(pattern, text) for pattern in self.context_patterns):
            return True
        
        return False

async def version():
    return VERSION

# Haar Cascade Classifier のパス
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
body_cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'

# Haar Cascade Classifier を読み込み
face_cascade = cv2.CascadeClassifier(face_cascade_path)
cat_cascade = cv2.CascadeClassifier(cat_cascade_path)
body_cascade = cv2.CascadeClassifier(body_cascade_path)

# Gemini APIの設定
GEMINI_API_VERSION = "v1"  # APIバージョンを明示的に指定

# 利用可能なモデルの定義
TEXT_MODELS = [
    "gemini-2.0-flash-lite",  # 優先的に使用
    "gemini-2.0-flash"       # フォールバック
]

IMAGE_GEN_MODELS = [
    "gemini-2.0-flash-exp-image-generation"  # 画像生成に使用するモデル
]

# クライアントとモデルの状態
class GeminiState:
    def __init__(self):
        self.client = None
        self.text_model = None
        self.image_model = None
        self.available_models = set()

    def is_initialized(self) -> bool:
        return self.client is not None

    def get_text_model(self) -> str:
        return self.text_model or TEXT_MODELS[-1]  # フォールバックとして最後のモデルを使用

    def get_image_model(self) -> str:
        return self.image_model or IMAGE_GEN_MODELS[-1]  # フォールバックとして最後のモデルを使用

# グローバル状態の初期化
gemini_state = GeminiState()
image_gen_settings = ImageGenSettings()
image_gen_trigger = ImageGenTrigger()

async def init_gemini_client(api_key: str):
    """
    Gemini APIクライアントを初期化し、利用可能なモデルを確認する
    
    Args:
        api_key (str): Gemini APIキー
        
    Raises:
        ValueError: APIキーが無効な場合やAPIが利用できない場合
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEYが設定されていません")
    
    try:
        global gemini_state
        # APIバージョンをHttpOptionsを使用して指定
        gemini_state.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version=GEMINI_API_VERSION)
        )

        # デフォルトのモデルを設定
        gemini_state.text_model = TEXT_MODELS[0]
        gemini_state.image_model = IMAGE_GEN_MODELS[0]

        logger.info(f"テキスト生成に {gemini_state.text_model} を使用します")
        logger.info(f"画像生成に {gemini_state.image_model} を使用します")
        logger.info("Gemini APIクライアントの初期化が完了しました")
        
    except Exception as e:
        raise ValueError(f"APIクライアントの初期化に失敗しました: {e}")


async def optimize_prompt(text: str) -> str:
    """
    会話の文脈から最適な画像生成プロンプトを生成
    """
    # クライアントの初期化確認
    if not gemini_state.is_initialized():
        logger.error("Gemini APIクライアントが初期化されていません")
        raise ValueError("GEMINI_API_KEYが設定されていないか、クライアントが初期化されていません")

    # 日本語テキストかどうかを判定
    is_japanese = any(ord(c) > 127 for c in text)
    
    try:
        if is_japanese:
            prompt_template = f'''
            以下の日本語テキストを、高品質な画像生成が可能な英語のプロンプトに変換してください。
            できるだけ詳細な特徴を含め、写実的な画像が生成できるようにしてください。
            装飾的な説明は不要で、プロンプトのみを出力してください。
            テキスト: {text}
            '''
        else:
            prompt_template = f'''
            Convert the following text into a high-quality image generation prompt.
            Include detailed characteristics to ensure photorealistic image generation.
            Only output the prompt without any decorative explanations.
            Text: {text}
            '''

        # プロンプト最適化に利用可能なモデルで試行
        model = gemini_state.get_text_model()
        response = gemini_state.client.models.generate_content(
            model=model,
            contents=prompt_template
        )

        if response and response.text:
            optimized = response.text.strip()
            logger.info(f"最適化されたプロンプト: {optimized}")
            return optimized
        else:
            logger.warning("プロンプト最適化の応答が空でした")
            if is_japanese:
                return f"Create a photorealistic image of: {text}"
            return text
    except Exception as e:
        logger.warning(f"プロンプト最適化に失敗しました: {e}")
        if is_japanese:
            return f"Generate an image of: {text}"
        return text

async def generate_image_with_gemini(prompt: str, config: Optional[Dict] = None) -> List[BytesIO]:
    """
    Gemini APIを使用して画像を生成
    Args:
        prompt (str): 画像生成のためのプロンプト
        config (Optional[Dict]): 追加の設定パラメータ（オプション）

    Returns:
        List[BytesIO]: 生成された画像のバイトストリームのリスト
    """
    try:
        # クライアントの初期化確認
        if not gemini_state.is_initialized():
            logger.error("Gemini APIクライアントが初期化されていません")
            raise ValueError("GEMINI_API_KEYが設定されていないか、クライアントが初期化されていません")

        # 設定の準備
        gen_config = config or image_gen_settings.to_dict()

        # プロンプトの最適化
        optimized_prompt = await optimize_prompt(prompt)
        logger.info(f"最適化されたプロンプト: {optimized_prompt}")

        # 生成設定（最新のAPI仕様に準拠）
        generate_content_config = GenerateContentConfig(
            response_modalities=["Text", "Image"]
        )

        # 必要な設定をパラメータとして渡す
        generation_params = {
            "temperature": 0.9,  # クリエイティビティのレベル
            "candidate_count": 1,  # 生成する候補の数
        }

        # 画像生成の実行
        model = gemini_state.get_image_model()
        return await _generate_with_model(
            model,
            optimized_prompt,
            generate_content_config,
            gen_config
        )

    except ValueError as ve:
        logger.error(f"バリデーションエラー: {ve}")
        raise
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
        raise

async def _generate_with_model(model: str, prompt: str, content_config: GenerateContentConfig, gen_config: Dict) -> List[BytesIO]:
    """
    指定されたモデルを使用して画像を生成する内部メソッド
    """
    images = []
    try:
        # 生成設定
        response = gemini_state.client.models.generate_content(
            model=model,
            contents={"parts": [{"text": prompt}]},
            safety_settings=[{
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }]
        )

        # レスポンスのバリデーション
        if not response or not hasattr(response, 'candidates') or not response.candidates:
            logger.error(f"{model}での画像生成でレスポンスが不正でした")
            raise ValueError("画像生成のレスポンスが不正です")
        
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    image_data = part.inline_data.data
                    image = Image.open(BytesIO(image_data))
                    image_bytes = BytesIO()
                    image.save(image_bytes, format='PNG')
                    image_bytes.seek(0)
                    images.append(image_bytes)
                    logger.info(f"{model}で画像が正常に生成されました")
                elif part.text:
                    logger.info(f"テキストパート: {part.text}")

        if not images:
            logger.warning(f"{model}での画像生成に失敗しました")
            raise ValueError("画像生成に失敗しました")

    except Exception as e:
        logger.error(f"{model}での画像生成中にエラーが発生: {e}")
        raise

    return images

async def process_img_with_ollama(img, question, bot_model):
    # 画像をリサイズして最適化
    max_size = (800, 800)
    img.thumbnail(max_size, Image.LANCZOS)
    
    # 画像をバイト列に変換
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Geminiモデルの場合
    if bot_model.startswith('gemini'):
        response = await chat_with_model(bot_model, messages=[
            {
                'role': 'user',
                'content': question,
                'images': [{'mime_type': 'image/jpeg', 'data': base64.b64encode(img_bytes.getvalue()).decode('utf-8')}]
            }
        ])
    else:
        # メモリ使用量のしきい値を定数として定義
        MEMORY_THRESHOLD_MB = 10000
        memory_info = psutil.virtual_memory()
        
        if (memory_info.used / 1024 / 1024) < MEMORY_THRESHOLD_MB:
            response = await chat_with_model('llama3.2-vision', messages=[
                {
                    'role': 'user',
                    'content': question,
                    'images': [img_bytes.getvalue()]
                }
            ])
        else:
            response = await chat_with_model(bot_model, messages=[
            {
                'role': 'user',
                'content': (
                    f'提供された画像の特徴は次のとおりです:\n'
                    f'{(await detect_objects_in_imageV2(img))}\n'
                    f'画像には{await detect_color_trend_in_image(img)}の傾向があり、\n'
                    f'画像に含まれるテキスト:\n{await extract_text_from_image(img)}\n'
                    f'このことから得られる画像の印象に基づいて、次の質問に回答してください:\n{question}\n'
                    f'なお、会話ログは下記の通りです:\n{await rag_log_processing.read_log_file("chatbot_log.txt")}\n'
                    f'回答生成時は以下の注意点を踏まえる:\n{await rag_log_processing.read_log_file("forAnswer.txt")}'
                )
            }
        ])
    return response

async def detect_objects_in_image(img): # OpenCVで物体検出
    # 画像をグレースケールに変換
    gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    # 物体 (顔、目など) を検出
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cats = cat_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    bodies = body_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出された物体の種類と数をリストに格納
    detected_objects = []
    for (x, y, w, h) in faces:
        detected_objects.append("顔")
    for (x, y, w, h) in cats:
        detected_objects.append("猫")
    for (x, y, w, h) in bodies:
        detected_objects.append("体")

    # 検出された物体の種類と数を使って、言語化した回答を生成
    num_objects = len(detected_objects)
    if num_objects == 0:
        response = "画像には物体が見つかりませんでした。"
    elif num_objects == 1:
        response = f"画像には 1 つの {detected_objects[0]} が見つかりました。"
    else:
        response = f"画像には {num_objects} つの {', '.join(detected_objects)} が見つかりました。"

    return response

async def detect_objects_in_imageV2(img): # YOLOで物体検出
    response = ''
    # Load a model
    model = YOLO('yolov8n.pt')

    # Predict the model
    results = model.predict(img, save=False, conf=0.5)

    # Print detected objects
    for result in results:
        for detection in result.boxes.data:
            class_name = model.names[int(detection[5])]
            print(f"Detected object: {class_name}")
            response += (f"Detected object: {class_name}")

    # Print the total number of detected objects
    print(response)

    return response

async def detect_color_trend_in_image(img): # OpenCVで画像の色の傾向を分析
    # 画像をRGB形式に変換
    rgb_img = img.convert('RGB')

    # 画像の色の平均を計算
    r_mean, g_mean, b_mean = ImageStat.Stat(rgb_img).mean

    # 色の傾向を特定
    color_trend = ''
    if r_mean > g_mean and r_mean > b_mean:
        color_trend = '赤色が多い'
    elif g_mean > r_mean and g_mean > b_mean:
        color_trend = '緑色が多い'
    elif b_mean > r_mean and b_mean > g_mean:
        color_trend = '青色が多い'
    elif r_mean == g_mean and r_mean > b_mean:
        color_trend = '赤色と緑色が等しいほど多い'
    elif r_mean == b_mean and r_mean > g_mean:
        color_trend = '赤色と青色が等しいほど多い'
    elif g_mean == b_mean and g_mean > r_mean:
        color_trend = '緑色と青色が等しいほど多い'
    else:
        color_trend = '色の傾向がはっきりしない'

    return color_trend

async def extract_text_from_image(img): # 画像からテキストを抽出
    text = pytesseract.image_to_string(img,lang='jpn')
    return text
