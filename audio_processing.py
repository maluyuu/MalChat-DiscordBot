import json
import asyncio
import tempfile
import os
from convertEngine import AudioConverter
from chat_processing import chat_with_model

VERSION = '0.0.1'

async def analyze_audio_request(request: str, model: str) -> dict:
    """
    chat_processing.pyを使用して自然言語のリクエストを解析
    """
    messages = [
        {
            'role': 'user',
            'content': f'''
            以下の音声変換リクエストから必要なパラメータを抽出してJSON形式で出力してください。
            JSONのみを出力し、その他の説明は不要です。

            抽出するパラメータ：
            - format: 出力フォーマット（wav/mp3/flac/ogg/m4a-aac/m4a-alac）
            - bit_depth: ビット深度（16bit/24bit/32bit float）
            - sample_rate: サンプルレート（Hz）

            デフォルト値（パラメータが明示されていない場合）：
            - format: パラメータ必須
            - bit_depth: "16bit"
            - sample_rate: 44100

            変換要求の例と対応するJSON：
            「このファイルをwavに変換して」
            {{"format": "wav", "bit_depth": "16bit", "sample_rate": 44100}}

            「320kbpsのmp3にして」
            {{"format": "mp3", "bit_depth": "16bit", "sample_rate": 44100}}

            「48kHz 24bitのflacファイルにして」
            {{"format": "flac", "bit_depth": "24bit", "sample_rate": 48000}}
            
            リクエスト: {request}
            '''
        }
    ]
    response = await chat_with_model(model, messages)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("変換パラメータの解析に失敗しました。")

def validate_conversion_params(params: dict) -> bool:
    """
    変換パラメータのバリデーション
    """
    valid_formats = ['mp3', 'wav', 'flac', 'ogg', 'm4a-aac', 'm4a-alac']
    valid_bit_depths = ['16bit', '24bit', '32bit float']
    valid_sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]

    if not all(key in params for key in ['format', 'bit_depth', 'sample_rate']):
        return False
    
    return (params['format'] in valid_formats and
            params['bit_depth'] in valid_bit_depths and
            params['sample_rate'] in valid_sample_rates)

async def is_audio_conversion_request(content: str, model: str) -> bool:
    """
    メッセージが音声変換要求かどうかを判断
    """
    messages = [
        {
            'role': 'user',
            'content': f'''
            以下のメッセージは音声ファイルの変換要求かどうかを判定してください。

            変換要求の例：
            - このファイルをmp3にして
            - wavファイルに変換して
            - 48kHzに変換
            - 320kbpsのmp3にして
            - m4aに変換
            - この音声を16bitにして
            
            変換要求の場合はtrueのみを、それ以外の場合はfalseのみを出力してください。
            その他の説明は不要です。

            メッセージ: {content}
            '''
        }
    ]
    response = await chat_with_model(model, messages)
    return 'true' in response.lower()

async def convert_audio_file(input_path: str, params: dict, progress_callback=None) -> str:
    """
    音声ファイルを変換
    """
    converter = AudioConverter()
    
    # 一時的な出力ファイルパスを生成
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{params['format']}") as tmp_file:
        output_path = tmp_file.name

    try:
        # 変換実行
        return await asyncio.to_thread(
            converter.convert_audio,
            input_path,
            params['format'],
            params['bit_depth'],
            params['sample_rate'],
            progress_callback=progress_callback,
            output_path=output_path
        )
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise e
