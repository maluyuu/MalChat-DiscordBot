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

# Gemini APIのクライアント設定
imagen_client = None
image_gen_settings = ImageGenSettings()
image_gen_trigger = ImageGenTrigger()

def init_imagen_client(api_key: str):
    global imagen_client
    genai.configure(api_key=api_key)
    imagen_client = genai.Client()

async def optimize_prompt(text: str) -> str:
    """
    会話の文脈から最適な画像生成プロンプトを生成
    """
    try:
        # Gemini APIを使用してプロンプトを最適化
        response = await chat_with_model('gemini-2.0-pro', messages=[
            {
                'role': 'user',
                'content': f'''
                以下のテキストから、画像生成に適した英語のプロンプトを生成してください。
                装飾的な説明は不要で、プロンプトのみを出力してください。

                テキスト: {text}
                '''
            }
        ])
        return response.strip()
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        # エラーの場合は元のテキストを英語に翻訳して返す
        response = await chat_with_model('gemini-2.0-pro', messages=[
            {
                'role': 'user',
                'content': f'Translate this to English: {text}'
            }
        ])
        return response.strip()

async def generate_image_with_gemini(prompt: str, config: Optional[Dict] = None) -> List[BytesIO]:
    """
    Gemini APIを使用して画像を生成
    """
    if imagen_client is None:
        raise ValueError("Imagen client is not initialized. Call init_imagen_client first.")

    try:
        # 設定の準備
        gen_config = config or image_gen_settings.to_dict()
        
        # 画像生成
        response = imagen_client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(
                number_of_images=gen_config['number_of_images'],
                aspect_ratio=gen_config['aspect_ratio'],
                safety_filter_level=gen_config['safety_filter_level'],
                person_generation=gen_config['person_generation']
            )
        )

        # 生成された画像をBytesIOオブジェクトのリストとして返す
        images = []
        for generated_image in response.generated_images:
            image_bytes = BytesIO(generated_image.image.image_bytes)
            images.append(image_bytes)
        
        return images

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise

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
