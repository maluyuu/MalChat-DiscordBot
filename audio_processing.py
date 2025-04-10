import json
import asyncio
import tempfile
import os
import logging
from convertEngine import AudioConverter
from chat_processing import chat_with_model
from utils.logger import setup_logger

VERSION = '0.0.1'

# ロギングの設定
logger = setup_logger(__name__, 'audio_processing.log')

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
            - format: 出力フォーマット（wav/mp3/flac/ogg/m4a-aac/m4a-alac）必須
            - bit_depth: ビット深度（16bit/24bit/32bit float）
            - sample_rate: サンプルレート（Hz）
            - bitrate: ビットレート（mp3形式のみ、kbps）

            デフォルト値：
            - bit_depth: "16bit"
            - sample_rate: 44100
            - bitrate: mp3の場合は192（kbps）

            変換要求の例と対応するJSON：
            「このファイルをmp3にして」
            {{"format": "mp3", "bit_depth": "16bit", "sample_rate": 44100, "bitrate": 192}}

            「320kbpsのmp3にして」
            {{"format": "mp3", "bit_depth": "16bit", "sample_rate": 44100, "bitrate": 320}}

            「48kHz 24bitのflacファイルにして」
            {{"format": "flac", "bit_depth": "24bit", "sample_rate": 48000}}

            「wavに変換」
            {{"format": "wav", "bit_depth": "16bit", "sample_rate": 44100}}
            
            リクエスト: {request}
            '''
        }
    ]
    response = await chat_with_model(model, messages)
    logger.debug(f"analyze_audio_request raw response: {response}")
    try:
        parsed = json.loads(response)
        logger.info(f"Parsed parameters: {parsed}")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}\nRaw response: {response}")
        raise ValueError(f"変換パラメータの解析に失敗しました。応答: {response}")

def validate_conversion_params(params: dict) -> bool:
    """
    変換パラメータのバリデーション
    """
    valid_formats = ['mp3', 'wav', 'flac', 'ogg', 'm4a-aac', 'm4a-alac']
    valid_bit_depths = ['16bit', '24bit', '32bit float']
    valid_sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000]
    valid_bitrates = [96, 128, 160, 192, 256, 320]  # MP3のビットレート（kbps）

    # 必須パラメータのチェック
    if 'format' not in params:
        return False

    # デフォルト値の設定
    if 'bit_depth' not in params:
        params['bit_depth'] = '16bit'
    if 'sample_rate' not in params:
        params['sample_rate'] = 44100
    if params['format'] == 'mp3' and 'bitrate' not in params:
        params['bitrate'] = 192

    # パラメータの検証
    if params['format'] not in valid_formats:
        return False
    if params['bit_depth'] not in valid_bit_depths:
        return False
    if params['sample_rate'] not in valid_sample_rates:
        return False
    
    # MP3の場合はビットレートも検証
    if params['format'] == 'mp3' and 'bitrate' in params:
        if params['bitrate'] not in valid_bitrates:
            return False

    return True

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
    logger.debug(f"is_audio_conversion_request response: {response}")
    result = 'true' in response.lower()
    logger.info(f"Audio conversion request detection: {result}")
    return result

async def convert_audio_file(input_path: str, params: dict, progress_callback=None) -> str:
    logger.info(f"Starting audio conversion with parameters: {params}")
    """
    音声ファイルを変換
    params:
        - format: 出力フォーマット（必須）
        - bit_depth: ビット深度（デフォルト: 16bit）
        - sample_rate: サンプルレート（デフォルト: 44100）
        - bitrate: ビットレート（MP3形式のみ、デフォルト: 192）
    """
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
