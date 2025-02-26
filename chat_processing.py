import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv

VERSION = '0.0.1'

load_dotenv()

def convert_model_name(model: str) -> str:
    """
    ユーザーフレンドリーなモデル名をAPIで使用する正確なモデル識別子に変換
    """
    model_mapping = {
        # Gemini 2.0 models
        'gemini-2.0-flash': 'gemini-2.0-flash-exp',  # 実験的モデル
        'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking-exp',  # 思考プロセス実験モデル
        'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite-exp',  # 軽量実験モデル
        'gemini-2.0-pro': 'gemini-2.0-pro-exp',  # Pro実験モデル
        # Gemini 1.5 models
        'gemini-1.5-flash': 'gemini-1.5-flash',  # 標準モデル
        'gemini-1.5-pro': 'gemini-1.5-pro',  # Proモデル
        'gemini-1.5-flash-8b': 'gemini-1.5-flash-8b',  # 8Bモデル
        # Gemini 1.0 models (Deprecated)
        'gemini-1.0-pro': 'gemini-1.0-pro',  # 非推奨モデル
        'gemini-1.0-pro-vision': 'gemini-1.0-pro-vision',  # 非推奨Visionモデル
    }
    return model_mapping.get(model, model)

async def chat_with_model(model: str, messages: list) -> str:
    """
    modelが"gemini"で始まる場合はGemini APIを利用し、それ以外は従来のollama.chatを利用する。
    """
    if model.lower().startswith("gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("Gemini APIの認証情報が.envに設定されていません。")
        
        # Gemini APIの設定
        genai.configure(api_key=api_key)
        
        try:
            # モデル名を正確なAPI識別子に変換
            model_name = convert_model_name(model)
            model_instance = genai.GenerativeModel(model_name)
            
            # チャット履歴の変換
            gemini_messages = []
            for msg in messages:
                role = "user" if msg["role"] != "assistant" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
            
            # チャットの実行
            chat = model_instance.start_chat(history=gemini_messages)
            response = await asyncio.to_thread(
                chat.send_message,
                messages[-1]["content"]
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini APIでエラーが発生しました: {str(e)}")
    else:
        # 従来のollamaを利用
        import ollama
        response = await asyncio.to_thread(ollama.chat, model=model, messages=messages)
        return response['message']['content']

async def list_available_models():
    """
    利用可能なGeminiモデルの一覧を取得する
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("Gemini APIの認証情報が.envに設定されていません。")
        
        # Gemini APIの設定
        genai.configure(api_key=api_key)
        
        # 利用可能なモデルを取得
        models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models.append(model.name)
        
        return models
    except Exception as e:
        raise Exception(f"Gemini APIでエラーが発生しました: {str(e)}") 