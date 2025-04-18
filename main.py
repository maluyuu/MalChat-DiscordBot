import discord
from discord.ext import commands
from pdfminer.high_level import extract_text
import ollama
import psutil
import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import datetime
from datetime import timezone, timedelta
import image_processing
import pdf_processing
import web_processing
import rag_log_processing
import system
import audio_processing
import logging
from logging.handlers import RotatingFileHandler
from rag_log_processing import chat_history_manager
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict
from utils.logger import setup_logger
from chat_processing import chat_with_model
import asyncio
import random  # 確率応答のために追加
import json

# 環境変数の読み込み
load_dotenv() #不要なコミットを追加

# ロギングの設定
logger = setup_logger(__name__, 'bot.log')

chanID_env = os.getenv('CHAN_ID')
chanID = [int(x) for x in chanID_env.split(',')] if chanID_env else []

# グローバル変数の定義
VERSION = '1.10'
BOT_NAME = 'MalChat'
BOT_MODEL = 'gemini-2.0-flash-lite'
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

async def download_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # RGBAモードの場合、RGBに変換
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    return img

async def generate_search_keywords(question: str) -> str:
    try:
        response = await chat_with_model(model=get_bot_model(), messages=[
            {
                'role': 'user',
                'content': f'''
                以下の質問から、ウェブ検索に適したキーワードを1-3個抽出してください。
                キーワードのみを出力し、説明は不要です。
                キーワードは半角スペースで区切ってください。

                質問: {question}
                '''
            }
        ])
        # responseそのものがテキスト文字列として返ってくるため、
        # 直接bot_responseに代入
        bot_response = response.strip()
        bot_response = re.sub(r'<think>.*?</think>', '', bot_response, flags=re.DOTALL)
        return bot_response
    except Exception as e:
        logger.error(f"Error generating search keywords: {e}")
        return question  # エラーの場合は元の質問をそのまま返す

async def determine_search_need(question):
    # 検索が必要かどうかを判断する関数を修正
    needs_search = False
    if any(keyword in question.lower() for keyword in [
        '調べ', 'ネットで', '検索', '最新', '最近', '現在', '今',
        '事例', '具体例', '情報', 'ニュース', '詳細','噂','真偽','真実'
    ]):
        needs_search = True

    # URLが含まれている場合の処理
    if 'http' in question:
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', question)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            question = question.replace(url, '').strip()
            question = f'{text}\n{question}'
            needs_search = True
        except Exception as e:
            pass
    return needs_search

def get_bot_model():
    global BOT_MODEL
    return BOT_MODEL

def set_bot_model(new_model):
    global BOT_MODEL
    BOT_MODEL = new_model

async def process_attachments(message, question: str) -> tuple[Optional[str], Optional[List[Dict]]]:
    context = []
    current_files = []

    for attachment in message.attachments:
        if attachment.filename.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            await message.reply(f'音声ファイルがアップロードされました: {attachment.filename}\n音声ファイルの処理を開始します。', mention_author=False)
            temp_file_path = f'temp_{attachment.filename}'
            await attachment.save(temp_file_path)

            if question:
                try:
                    # 音声変換のリクエストかどうかを確認
                    is_conversion = await audio_processing.is_audio_conversion_request(question, get_bot_model())
                    logger.debug(f"Audio conversion request check: {is_conversion} for message: {question}")
                    
                    if is_conversion:
                        # 変換パラメータを解析
                        params = await audio_processing.analyze_audio_request(question, get_bot_model())
                        logger.debug(f"Parsed audio parameters: {params}")

                        if audio_processing.validate_conversion_params(params):
                            logger.info(f"Starting audio conversion for {attachment.filename} with params: {params}")
                            # 変換の進捗を通知する関数
                            async def progress_callback(progress):
                                if progress % 25 == 0:  # 25%ごとに進捗を通知
                                    await message.channel.send(f'変換進捗: {progress}%')

                            # 音声ファイルを変換
                            converted_file_path = None
                            try:
                                converted_file_path = await audio_processing.convert_audio_file(
                                    temp_file_path,
                                    params,
                                    progress_callback=progress_callback
                                )
                                # 変換情報の生成
                                conversion_context = f"""
音声変換処理の詳細:
- 入力ファイル: {attachment.filename}
- 変換パラメータ:
  - 出力形式: {params['format']}
  - ビット深度: {params['bit_depth']}
  - サンプルレート: {params['sample_rate']}Hz
{f"  - ビットレート: {params['bitrate']}kbps" if 'bitrate' in params else ''}
- 変換結果: 正常に完了
- 出力ファイル: {os.path.basename(converted_file_path)}
"""
                                # 文脈に変換情報を追加
                                context.append(conversion_context)

                                # チャット履歴に変換情報を追加
                                await chat_history_manager.add_entry(
                                    role='system',
                                    content=f'音声変換機能: {conversion_context}',
                                    channel_id=message.channel.id
                                )

                                # 変換成功のメッセージ生成
                                response = await chat_with_model(get_bot_model(), messages=[
                                    {
                                        'role': 'system',
                                        'content': '音声変換処理を実行中です。このファイルに対する変換操作の結果を報告してください。'
                                    },
                                    {
                                        'role': 'user',
                                        'content': f'''
現在の音声変換処理が完了しました:
{conversion_context}

変換の結果を報告してください。技術的なパラメータについても言及してください。
'''
                                    }
                                ])
                                bot_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                                
                                # 変換後のファイルを送信
                                await message.reply(bot_response, mention_author=False)
                                await message.reply(file=discord.File(converted_file_path), mention_author=False)
                            except Exception as e:
                                error_msg = f'変換処理中にエラーが発生しました: {str(e)}'
                                logger.error(f"Conversion error for {attachment.filename}: {e}", exc_info=True)
                                await message.reply(error_msg, mention_author=False)
                                raise
                            finally:
                                # 一時ファイルを削除
                                if os.path.exists(temp_file_path):
                                    os.remove(temp_file_path)
                                if converted_file_path and os.path.exists(converted_file_path):
                                    os.remove(converted_file_path)
                        else:
                            await message.reply('指定された変換パラメータが無効です。サポートされている形式：wav, mp3, flac, ogg, m4a', mention_author=False)
                    else:
                        await message.reply('音声ファイルの変換方法を指定してください。例：「mp3に変換」「48kHz wavに変換」「320kbpsのmp3に変換」', mention_author=False)
                except json.JSONDecodeError as e:
                    error_msg = f'音声変換パラメータの解析に失敗しました。詳細: {str(e)}'
                    logger.error(f"JSON parse error for {attachment.filename}: {e}")
                    await message.reply(error_msg, mention_author=False)
                except ValueError as e:
                    error_msg = f'パラメータが無効です: {str(e)}'
                    logger.error(f"Invalid parameters for {attachment.filename}: {e}")
                    await message.reply(error_msg, mention_author=False)
                except Exception as e:
                    error_msg = f'音声ファイルの処理中にエラーが発生しました: {str(e)}'
                    logger.error(f"Error processing audio file {attachment.filename}: {e}", exc_info=True)
                    await message.reply(error_msg, mention_author=False)
            else:
                await message.reply('音声ファイルをアップロードしました。変換方法を指定してください。例：「mp3に変換」「48kHz wavに変換」「320kbpsのmp3に変換」', mention_author=False)
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        elif attachment.filename.endswith('.pdf'):
            await message.reply(f'PDFファイルがアップロードされました: {attachment.filename}\nPDFファイルの解析を開始します。', mention_author=False)
            try:
                pdf_text = await pdf_processing.get_pdfText(attachment)
                file_info = {
                    'content': pdf_text,
                    'name': attachment.filename,
                    'type': 'pdf'
                }
                current_files.append(file_info)
                await chat_history_manager.process_uploaded_files([file_info])
                context.append(f"PDFファイル '{attachment.filename}' の内容:\n{pdf_text}")
            except Exception as e:
                await message.reply(f"PDFファイル '{attachment.filename}' の処理中にエラーが発生しました: {str(e)}", mention_author=False)
                logger.error(f"Error processing PDF file {attachment.filename}: {e}")

        elif attachment.filename.endswith(('.jpg', '.png', '.jpeg')):
            await message.reply(f'画像がアップロードされました: {attachment.filename}\n画像の取得を開始します。', mention_author=False)
            img = await download_image_from_url(attachment.url)
            img_description = await image_processing.process_img_with_ollama(img, question, get_bot_model())
            file_info = {
                'content': img_description,
                'name': attachment.filename,
                'type': 'image'
            }
            current_files.append(file_info)
            await chat_history_manager.process_uploaded_files([file_info])
            context.append(f"画像 '{attachment.filename}' の内容:\n{img_description}")

        elif attachment.filename.endswith(('.txt', '.py', '.json', '.md')):
            file_content = await attachment.read()
            text_content = file_content.decode('utf-8')
            file_info = {
                'content': text_content,
                'name': attachment.filename,
                'type': 'text'
            }
            current_files.append(file_info)
            await chat_history_manager.process_uploaded_files([file_info])
            context.append(f"ファイル '{attachment.filename}' の内容:\n{text_content}")

    # 複数のファイルからのテキストを結合して文脈を形成
    combined_context = '\n\n'.join(context) if context else None
    return combined_context, current_files

@bot.event
async def on_message(message):
    try:
        # !malSysコマンドの処理を最初に行う
        if message.content.startswith('!malSys'):
            commands = message.content.replace('!malSys', '').strip()
            await message.channel.send(commands)
            if commands == 'info':
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                gpu_info = '取得できません'
                system_info = f'```\nCPU使用率: {cpu_percent}%\nメモリ使用量: {memory_info.used / 1024 / 1024:.2f} MB / {memory_info.total / 1024 / 1024:.2f} MB\nGPU情報: {gpu_info}\n```'
                await message.reply(system_info, mention_author=False)
                return
            elif commands == 'about':
                system_info = f'```\nName : {BOT_NAME}\nVersionInfo\nBot : {VERSION}\nSystem : {await system.version()}\nLogging Module : {await rag_log_processing.version()}\nImage Module: {await image_processing.version()}\nPDF Module : {await pdf_processing.version()}\nWeb Module : {await web_processing.version()}\nAudio Module: {audio_processing.VERSION}\nModel : {get_bot_model()}\n```'
                await message.reply(system_info, mention_author=False)
            elif commands.startswith('model'):
                model_name = commands.replace('model', '').strip()
                if model_name in [
                    # Gemini 2.0 models
                    'gemini-2.0-flash',
                    'gemini-2.0-pro',
                    'gemini-2.0-flash-lite',
                    'gemini-2.0-flash-thinking',
                    # Gemini 2.5 models
                    'gemini-2.5-pro',
                    # Ollama models
                    'gemma2',
                    'gemma2JP',
                    'gemma3',
                    'llama3.1',
                    'llama3.2',
                    'deepseek-r1',
                    'deepseek-r1:14b',
                    'deepseek-r1JP',
                    'phi4',
                    'tinySwallow'
                ]:
                    set_bot_model(model_name)
                    await message.reply(f'現在のモデルは{get_bot_model()}です。', mention_author=False)
                else:
                    model_list = [
                        '=== Gemini 2.0 ===',
                        'gemini-2.0-flash (マルチモーダル対応の高速モデル)',
                        'gemini-2.0-pro (最高品質のモデル)',
                        'gemini-2.0-flash-lite (コスト効率の良いモデル)',
                        'gemini-2.0-flash-thinking (思考プロセスを含む推論モデル)',
                        '=== Gemini 2.5 ===',
                        'gemini-2.5-pro (実験的なモデル)'
                    ]
                    model_list_str = '\n'.join(model_list)
                    await message.reply(f'利用可能なモデル:\n```{model_list_str}```', mention_author=False)
            return

        # botからのメッセージは無視
        if message.author.id == bot.user.id:
            return

        # 他のコマンド処理を続行
        dt_now = datetime.datetime.now(timezone(timedelta(hours=9)))

        if message.content.startswith('!malRoboKasu'):
            channel_id = message.channel.id
            await chat_history_manager.write_log_file(message.authoa.namee.author.name, message.content, channel_id=channel_id)
            question = message.content.replace('!malRoboKasu', '').strip()
            await message.channel.typing()
            response = await chat_with_model(get_bot_model(), messages=[
                {
                    'role': 'user',
                    'content': (
                        'あなたの名前は「ロボカス」です。あなたは時代背景や宗教観を無視した「事後諸葛亮」的な発言を連発するロボットです。'
                        '例として、江戸時代のキリシタン弾圧時の踏み絵の写真に対し「踏めば助かるのに…。」などの酷薄な発言があります。'
                        '回答は端的にお願いします。'
                        f'\n次の質問に答えてください: {question}'
                    )
                }
            ])
            bot_response = response
            bot_response = re.sub(r'<think>.*?</think>', '', bot_response, flags=re.DOTALL)
            await message.reply(bot_response, mention_author=False)
            await chat_history_manager.write_log_file('{BOT_NAME}', bot_response, channel_id=channel_id)
            logger.info(f"Channel ID: {channel_id}, Bot Response: {bot_response}")
        elif message.content.startswith('!malChat') or message.channel.id in chanID or message.content.startswith('!malDebugChat') or message.channel.type == discord.ChannelType.private or random.random() < 1.00:
            try:
                # チャンネルIDを取得
                channel_id = message.channel.id


                if message.content.startswith('!malDebugChat'):
                    debugChat = True
                else:
                    debugChat = False

                question = message.content.replace('!malChat', '').strip()

                # チャット履歴にユーザーの質問を追加
                await chat_history_manager.add_entry(
                    role=message.author.name,
                    content=message.content, channel_id=channel_id
                )

                # 応答が必要かどうかを判定
                is_mention_or_command = False # 応答必須条件フラグ
                needs_response = False
                is_random_response = False

                # 応答必須条件のチェック
                if message.channel.id in chanID or \
                   message.content.startswith('!malChat') or \
                   message.content.startswith('!malDebugChat') or \
                   message.channel.type == discord.ChannelType.private:
                    is_mention_or_command = True
                elif message.reference:
                    try: # fetch_message が失敗する可能性を考慮
                        referenced_message = await message.channel.fetch_message(message.reference.message_id)
                        if referenced_message.author.id == bot.user.id:
                            is_mention_or_command = True
                    except discord.NotFound:
                        logger.warning(f"Referenced message not found: {message.reference.message_id}")
                    except discord.HTTPException as e:
                        logger.error(f"Failed to fetch referenced message: {e}")
                elif bot.user.mentioned_in(message):
                    is_mention_or_command = True

                # 応答全体の要否判定
                if is_mention_or_command:
                    needs_response = True
                elif random.random() < 0.00: # ランダム応答確率
                    needs_response = True
                    is_random_response = True

                if needs_response:
                    await message.channel.typing()

                # 応答必須条件が満たされている場合のみ、添付ファイルと履歴の処理を行う
                context = None
                current_files = None
                relevant_history = None
                relevant_context = None
                if is_mention_or_command: # 応答必須条件が満たされている場合
                    if message.attachments:
                        # 添付ファイルの処理
                        context, current_files = await process_attachments(message, question)

                    # 関連する履歴と文脈を取得
                    relevant_history = await chat_history_manager.get_combined_history(
                        query=question,
                        channel_id=channel_id,
                        similarity_threshold=0.5,
                        recent_count=10
                    )
                    relevant_context = await chat_history_manager.get_relevant_context(question, current_files)

                    if relevant_context:
                        question = f"{relevant_context}\nこれを考慮して次の質問に回答してください:\n{question}"

                # 応答生成ロジック
                botAnswer = None # 送信する応答メッセージを初期化
                if needs_response:
                    # システムプロンプトの構築 (応答が必要な場合のみ)
                    system_prompt = (
                        f'あなたの名前は{BOT_NAME}で、maluyuuによって開発されたDiscord botです。\n'
                        f'現在の時刻は: {dt_now}\n'
                        f'あなたは以下の機能を持っています：\n'
                        f'- 音声ファイルの形式変換（MP3, WAV, FLAC, OGG, M4A対応）\n'
                        f'- ビットレート、サンプルレート、ビット深度の調整\n'
                        f'以下の情報を考慮して回答してください:\n'
                    )
                    if context: # context は is_mention_or_command が True の場合のみ生成される
                        system_prompt += f'\n現在のメッセージの添付ファイル:\n{context}\n'
                    if relevant_context: # relevant_context も同様
                        system_prompt += f'\n関連する過去のファイル:\n{relevant_context}\n'

                    # メッセージの作成
                    messages = []
                    if relevant_history: # relevant_history も同様
                        # 最新の履歴エントリを確認
                        latest_entry = relevant_history[-1] if relevant_history else None
                        if latest_entry and 'role' in latest_entry and latest_entry['role'] == 'system' and '音声変換機能' in latest_entry.get('content', ''):
                            # 音声変換が進行中の場合、変換情報を優先
                            messages.append({
                                'role': 'system',
                                'content': '音声変換処理の結果について回答を生成します。'
                            })
                        else:
                            # 通常の履歴処理
                            for entry in relevant_history:
                                if entry and isinstance(entry, dict):
                                    messages.append({
                                        'role': entry.get('role', 'user'),
                                        'content': entry.get('content', '')
                                    })

                    # 現在の質問を追加
                    if context and '音声変換処理の詳細' in context:
                        messages.append({
                            'role': 'user',
                            'content': f'音声変換の結果について報告してください:\n\n{context}'
                        })
                    else:
                        messages.append({
                            'role': 'user',
                            'content': f'{system_prompt}\nそれを踏まえて次の質問に回答してください : {question}'
                        })
                    print(messages) # デバッグ用

                    # 検索処理 (応答必須条件が満たされている場合のみ)
                    needs_search = False
                    if is_mention_or_command and any(keyword in question.lower() for keyword in [
                        '調べ', 'ネットで', '検索', '最新', '最近', '現在', '今',
                        '事例', '具体例', '情報', 'ニュース'
                    ]):
                        needs_search = True

                    if needs_search: # is_mention_or_command は既に True
                        await message.channel.typing() # 検索中に再度 typing 表示

                        search_query = await generate_search_keywords(question)
                        logger.debug(f"Generated search keywords: {search_query}")
                        await message.reply(f"次の検索ワードでウェブ検索を開始します : {search_query}", mention_author=False)
                        links = await web_processing.get_links(search_query)
                        if links:
                            search_results = await web_processing.extract_data_from_links(links, search_query)
                            await message.reply('検索が完了しました。回答を生成します...', mention_author=False)

                            response = await chat_with_model(get_bot_model(), messages=[
                                {
                                    'role': 'user',
                                    'content': (
                                        f'以下の検索結果を基に、質問に詳しく回答してください。\n\n{search_results}\n\n質問:\n{question}'
                                    )
                                }
                            ])

                            botAnswer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                            logger.info(f"Channel ID: {channel_id}, Bot Response (search): {botAnswer}")
                        else:
                            botAnswer = '申し訳ありません。関連する情報が見つかりませんでした。'
                            logger.info(f"Channel ID: {channel_id}, Bot Response (search failed)")
                    else: # 検索不要の場合 (needs_response は True)
                        # 通常の質問応答を実行 (ファイル情報も messages に含まれているはず)
                        response = await chat_with_model(get_bot_model(), messages=messages)
                        botAnswer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                        logger.info(f"Channel ID: {channel_id}, Bot Response (normal/file): {botAnswer}")

                # 応答メッセージの送信処理
                if botAnswer: # botAnswer が生成された場合のみ送信
                    # メッセージが2000文字を超える場合は分割して送信
                    if len(botAnswer) > 2000:
                        # メッセージを分割 (コードブロック対応改善)
                        chunks = []
                        current_chunk = ""
                        in_code_block = False
                        for line in botAnswer.splitlines(keepends=True):
                            if len(current_chunk) + len(line) > 1900:
                                if in_code_block:
                                    current_chunk += '```' # コードブロックを閉じる
                                chunks.append(current_chunk)
                                current_chunk = ""
                                if in_code_block:
                                    current_chunk += '```\n' # 次のチャンクでコードブロックを開始

                            current_chunk += line
                            # 行末の ``` で判定するのではなく、行全体が ``` かどうかで判定
                            if line.strip() == '```':
                                in_code_block = not in_code_block

                        if current_chunk: # 最後のチャンクを追加
                            # 最後のチャンクがコードブロックの途中で終わっている場合、閉じる
                            if in_code_block and current_chunk.count('```') % 2 == 1:
                                 current_chunk += '\n```'
                            chunks.append(current_chunk)

                        # 分割したメッセージを順番に送信
                        for i, chunk in enumerate(chunks, 1):
                            header = f"(分割メッセージ {i}/{len(chunks)})\n" if len(chunks) > 1 else ""
                            await message.reply(f"{header}{chunk}", mention_author=False)
                            await asyncio.sleep(1)  # レート制限を避けるため少し待機
                    else:
                        await message.reply(botAnswer, mention_author=False)
            except Exception as e:
                await message.reply(f"メッセージ処理中にエラーが発生: {str(e)}", mention_author=False)
                raise
    except Exception as e:
        error_message = f"Error: {str(e)}\nError type: {type(e).__name__}\nError location: {__file__}, line={e.__traceback__.tb_lineno}"
        print(error_message)

    # コマンド処理を確実に行う
    await bot.process_commands(message)

@bot.event
async def on_ready():
    print(f'ログインしました: {bot.user}')

bot.run(TOKEN)
