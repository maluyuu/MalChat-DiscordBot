from pydub import AudioSegment
import os
import sys
import uuid
import time
import threading
from typing import Optional, Callable

class AudioConversionError(Exception):
    """オーディオ変換に関連するエラーを表すカスタム例外"""
    pass

def find_ffmpeg():
    """システムにインストールされているFFmpegを探索"""
    # macOSの一般的なFFmpegの場所をチェック
    common_paths = [
        '/opt/homebrew/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/usr/bin/ffmpeg',
        os.path.dirname(sys.executable), # 実行ファイル自身のパスを基準にdistフォルダを探索
        'C:\\\\FFmpeg\\\\bin'  # Windowsの一般的なFFmpegの場所
    ]
    
    print(f"DEBUG: Searching FFmpeg in common paths: {common_paths}") # デバッグ出力

    for path in common_paths:
        ffmpeg_executable = os.path.join(path, 'ffmpeg' if sys.platform != 'win32' else 'ffmpeg.exe')
        if os.path.isfile(ffmpeg_executable): # パスとffmpeg実行可能ファイルを結合して存在確認
            print(f"DEBUG: FFmpeg found in common path: {path}") # デバッグ出力
            return path

    # PATHから検索
    for path in os.environ.get('PATH', '').split(os.pathsep):
        ffmpeg_path = os.path.join(path, 'ffmpeg' if sys.platform != 'win32' else 'ffmpeg.exe')
        if os.path.isfile(ffmpeg_path):
            print(f"DEBUG: FFmpeg found in PATH: {path}") # デバッグ出力
            return path

    print("DEBUG: FFmpeg not found in common paths or PATH") # デバッグ出力
    return None

class AudioConverter:
    def __init__(self):
        self.supported_formats = ['mp3', 'wav', 'flac', 'ogg', 'm4a-aac', 'm4a-alac', 'aiff']
        
        # FFmpegのパスを探索
        ffmpeg_dir = find_ffmpeg()
        if not ffmpeg_dir:
            raise AudioConversionError("FFmpegが見つかりません。システムにFFmpegをインストールしてください。")
        
        # FFmpegのパスを設定
        os.environ['PATH'] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    def validate_input_file(self, input_path: str) -> None:
        """入力ファイルの妥当性を検証"""
        if not os.path.exists(input_path):
            raise AudioConversionError(f"入力ファイルが見つかりません: {input_path}")
        
        if not os.path.isfile(input_path):
            raise AudioConversionError(f"指定されたパスはファイルではありません: {input_path}")
        
        if os.path.getsize(input_path) == 0:
            raise AudioConversionError(f"ファイルが空です: {input_path}")

    def convert_audio(self, 
                     input_path: str, 
                     output_format: str, 
                     bit_depth: Optional[str] = None, 
                     sample_rate: Optional[int] = None, 
                     bitrate: Optional[str] = None, 
                     progress_callback: Optional[Callable[[int], None]] = None, 
                     output_path: Optional[str] = None) -> str:
        """
        オーディオファイルを指定された形式に変換する
        
        Args:
            input_path (str): 入力ファイルのパス
            output_format (str): 出力ファイルの形式
            bit_depth (str, optional): ビット深度
            sample_rate (int, optional): サンプルレート（Hz）
            bitrate (str, optional): ビットレート
            progress_callback (function, optional): 進捗を報告するコールバック関数
            output_path (str, optional): 出力ファイルのパス
        
        Returns:
            str: 出力ファイルのパス
        
        Raises:
            AudioConversionError: 変換処理中にエラーが発生した場合
        """
        try:
            # 入力ファイルの検証
            self.validate_input_file(input_path)
            
            # フォーマットの検証
            if not self.is_format_supported(output_format):
                raise AudioConversionError(f"未サポートの出力形式です: {output_format}")

            # 入力ファイルの形式を取得（拡張子が大文字の場合も考慮）
            input_format = os.path.splitext(input_path)[1].lower().lstrip('.')
            
            # output_pathの生成と検証
            if not output_path:
                if output_format.lower().startswith('m4a-'):
                    output_path = os.path.splitext(input_path)[0] + '.m4a'
                else:
                    output_path = os.path.splitext(input_path)[0] + '.' + output_format.lower()

            # 出力ディレクトリの存在確認
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 入力ファイルを読み込み（m4aの場合は両方のコーデックを試行）
            # 入力ファイルを読み込み
            try:
                # 拡張子に基づいてフォーマットを決定
                if input_format == 'm4a':
                    try:
                        audio = AudioSegment.from_file(input_path, format='m4a', codec='aac')  # まずAACとして試行
                    except:
                        audio = AudioSegment.from_file(input_path, format='m4a', codec='alac')  # 次にALACとして試行
                elif input_format == 'aif':
                    audio = AudioSegment.from_file(input_path, format='aiff') # aifの場合もaiffとして処理
                else:
                    audio = AudioSegment.from_file(input_path, format=input_format)
            except Exception as e:
                raise AudioConversionError(f"ファイルの読み込みに失敗しました: {str(e)}")
            
            # 基本的なexport_argsの設定
            if output_format.lower() == 'm4a-aac':
                # AACエンコーダを使用（lossy圧縮）
                export_args = {
                    "format": "ipod",   # ipodフォーマットを使用
                    "codec": "aac"      # AACコーデックを指定
                }
            elif output_format.lower() == 'm4a-alac':
                # ALACエンコーダを使用（lossless圧縮）
                export_args = {
                    "format": "ipod",   # ipodフォーマットを使用
                    "codec": "alac"     # ALACコーデックを指定
                }
            else:
                export_args = {"format": output_format.lower()}

            # サンプルレートとビット深度の設定
            if sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            if bit_depth:
                if bit_depth == "16bit":
                    audio = audio.set_sample_width(2)
                elif bit_depth == "24bit":
                    audio = audio.set_sample_width(3)
                elif bit_depth == "32bit float":
                    audio = audio.set_sample_width(4)

            # ビットレートの設定（AACの場合のみ）
            if bitrate:
                if output_format.lower() == 'm4a-aac':
                    export_args["audio_bitrate"] = bitrate
                elif output_format.lower() != 'm4a-alac':  # ALACではビットレート指定不要
                    export_args["bitrate"] = bitrate

            if progress_callback:
                # バックグラウンドでexport処理を実行
                finished = [False]
                def export_func():
                    try:
                        self.export_func(audio, output_path, export_args, progress_callback)
                        finished[0] = True
                    except Exception as e:
                        raise e

                thread = threading.Thread(target=export_func)
                thread.start()

                # export処理が終わるまで、0～95%を徐々に更新（5%ずつ約0.1秒間隔で進捗更新）
                progress = 0
                while not finished[0] and progress < 95:
                    progress += 5
                    progress_callback(progress)
                    time.sleep(0.1)  # 進捗更新の間隔を長くする

                thread.join()
                progress_callback(100)
            else:
                audio.export(output_path, **export_args)
            
            return output_path
            
        except Exception as e:
            raise AudioConversionError(f"変換エラー: {str(e)}")

    def export_func(self, audio, output_path, export_args, progress_callback):
        try:
            # 修正: 'audio_bitrate' になっている場合、正しい 'bitrate' キーに変換する
            if 'audio_bitrate' in export_args:
                export_args['bitrate'] = export_args.pop('audio_bitrate')
            audio.export(output_path, **export_args)
            progress_callback(100)
        except Exception as e:
            # エラー発生時でも進捗を完了として更新しておく
            progress_callback(100)
            raise e

    def is_format_supported(self, format: str) -> bool:
        """指定された形式がサポートされているかチェック"""
        return format.lower() in self.supported_formats

    def get_supported_formats(self) -> list:
        """サポートされている形式のリストを返す"""
        return self.supported_formats.copy()

# 使用例
if __name__ == "__main__":
    converter = AudioConverter()
    
    try:
        # WAVファイルをMP3に変換する例
        input_file = "example.wav"
        output_format = "mp3"
        
        if converter.is_format_supported(output_format):
            output_path = converter.convert_audio(input_file, output_format, bitrate="320k")
            print(f"変換成功: {output_path}")
        else:
            print(f"未サポートの形式です: {output_format}")
            
    except Exception as e:
        print(f"エラー: {str(e)}")
