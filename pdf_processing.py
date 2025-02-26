from pdfminer.high_level import extract_text
import os
from utils.logger import setup_logger
from chat_processing import chat_with_model

VERSION = '0.0.1'

logger = setup_logger(__name__, 'pdf_processing.log')

async def version():
    return VERSION

async def process_pdf_with_ollama(pdf_file_path, question, bot_model):
    text = extract_text(pdf_file_path)
    response = await chat_with_model(bot_model, messages=[
        {
            'role': 'user',
            'content': (
                f'提供されたファイルの内容は次のとおりです:\n{text}\n'
                f'これを踏まえて次の質問に回答してください:\n{question}'
            )
        }
    ])
    return response

async def get_pdfText(attachment):
    # 一時ファイル名をユニークにする
    temp_filename = f'temp_{attachment.filename}'
    pdf_data = await attachment.read()
    
    try:
        with open(temp_filename, 'wb') as pdf_file:
            pdf_file.write(pdf_data)
        
        # PDFからテキストを抽出
        text = extract_text(temp_filename)
        
        # 一時ファイルを削除
        os.remove(temp_filename)
        
        return text
        
    except Exception as e:
        # エラー発生時も一時ファイルを確実に削除
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise e