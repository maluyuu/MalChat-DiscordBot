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

VERSION = '0.2.0'

# ロガーの設定
logger = setup_logger(__name__, 'image_processing.log')

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
    #total_objects = sum(len(result.boxes.data) for result in results)
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
