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

# 環境変数の読み込み
load_dotenv() #不要なコミットを追加

# ロギングの設定
logger = setup_logger(__name__, 'bot.log')

chanID_env = os.getenv('CHAN_ID')
chanID = [int(x) for x in chanID_env.split(',')] if chanID_env else []

# グローバル変数の定義
VERSION = '1.00'
BOT_NAME = 'MalChat'
BOT_MODEL = 'gemini-2.0-flash'
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
        if attachment.filename.endswith('.pdf'):
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
                system_info = f'```\nName : {BOT_NAME}\nVersionInfo\nBot : {VERSION}\nSystem : {await system.version()}\nLogging Module : {await rag_log_processing.version()}\nImage Module: {await image_processing.version()}\nPDF Module : {await pdf_processing.version()}\nWeb Module : {await web_processing.version()}\nModel : {get_bot_model()}\n```'
                await message.reply(system_info, mention_author=False)
            elif commands.startswith('model'):
                model_name = commands.replace('model', '').strip()
                if model_name in [
                    # Gemini 2.0 models
                    'gemini-2.0-flash',
                    'gemini-2.0-pro',
                    'gemini-2.0-flash-lite',
                    'gemini-2.0-flash-thinking',
                    # Ollama models
                    'gemma2',
                    'gemma2JP',
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
                        '=== Ollama Models ===',
                        'gemma2',
                        'gemma2JP',
                        'llama3.1',
                        'llama3.2',
                        'deepseek-r1',
                        'deepseek-r1:14b',
                        'deepseek-r1JP',
                        'phi4',
                        'tinySwallow'
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
            await chat_history_manager.write_log_file(message.author.name, message.content, channel_id=channel_id)
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

                # Gemini APIのキーを設定
                if not hasattr(bot, 'imagen_initialized'):
                    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
                    if GEMINI_API_KEY:
                        image_processing.init_imagen_client(GEMINI_API_KEY)
                        bot.imagen_initialized = True

                # チャット履歴にユーザーの質問を追加
                await chat_history_manager.add_entry(
                    role=message.author.name,
                    content=message.content, channel_id=channel_id
                )

                # 画像生成が必要かどうかを判断
                if hasattr(bot, 'imagen_initialized') and image_processing.image_gen_trigger.should_generate(question):
                    try:
                        await message.channel.typing()
                        # プロンプトを最適化
                        optimized_prompt = await image_processing.optimize_prompt(question)
                        # 画像を生成
                        images = await image_processing.generate_image_with_gemini(optimized_prompt)
                        
                        # 画像を送信
                        for i, image_bytes in enumerate(images, 1):
                            image_bytes.seek(0)
                            await message.channel.send(
                                f"生成された画像 {i}/{len(images)}",
                                file=discord.File(fp=image_bytes, filename=f'generated_image_{i}.png')
                            )
                    except Exception as e:
                        logger.error(f"Error generating image: {e}")
                        await message.channel.send("画像の生成中にエラーが発生しました。")
                
                # 応答が必要かどうかを判定
                needs_response = False
                is_random_response = False

                if message.channel.id in chanID or message.content.startswith('!malChat') or message.content.startswith('!malDebugChat') or message.channel.type == discord.ChannelType.private:
                    needs_response = True
                elif message.reference:  # メッセージがリプライである場合
                    referenced_message = await message.channel.fetch_message(message.reference.message_id)
                    if referenced_message.author.id == bot.user.id:  # ボットへのリプライの場合
                        needs_response = True
                elif bot.user.mentioned_in(message): # botにメンションがあった場合
                    needs_response = True
                elif random.random() < 0.05:
                    needs_response = True
                    is_random_response = True

                if needs_response:
                    await message.channel.typing()

                # ランダム応答でない場合のみ、添付ファイルと履歴の処理を行う
                if not is_random_response:
                    # 添付ファイルの処理
                    context, current_files = await process_attachments(message, question) if message.attachments else (None, None)

                    # 関連する履歴と文脈を取得（類似度0.5以上の関連履歴と最新10件）
                    relevant_history = await chat_history_manager.get_combined_history(
                        query=question,
                        channel_id=channel_id,
                        similarity_threshold=0.5,
                        recent_count=10
                    )
                    relevant_context = await chat_history_manager.get_relevant_context(question, current_files)

                    if relevant_context:
                        question = f"{relevant_context}\nこれを考慮して次の質問に回答してください:\n{question}"
                else:
                    context = None
                    current_files = None
                    relevant_history = None
                    relevant_context = None

                # システムプロンプトの構築
                system_prompt = (
                    f'あなたの名前は{BOT_NAME}で、maluyuuによって開発されたDiscord botです。\n'
                    f'現在の時刻は: {dt_now}\n'
                    f'以下の情報を考慮して回答してください:\n'
                )

                if context:
                    system_prompt += f'\n現在のメッセージの添付ファイル:\n{context}\n'

                if relevant_context:
                    system_prompt += f'\n関連する過去のファイル:\n{relevant_context}\n'

                # メッセージの作成
                messages = []

                # 結合した履歴をメッセージリストに追加
                if relevant_history:  # Noneでない場合のみ処理
                    for entry in relevant_history:
                        if entry and isinstance(entry, dict):  # エントリーが有効な辞書の場合のみ追加
                            messages.append({
                                'role': entry.get('role', 'user'),
                                'content': entry.get('content', '')
                            })

                # 現在の質問を追加
                messages.append({
                    'role': 'user',
                    'content': f'{system_prompt}\nそれを踏まえて次の質問に回答してください : {question}'
                })
                print(messages)

                # ランダム応答でない場合のみ、検索処理を行う
                needs_search = False
                if not is_random_response and any(keyword in question.lower() for keyword in [
                    '調べ', 'ネットで', '検索', '最新', '最近', '現在', '今',
                    '事例', '具体例', '情報', 'ニュース'
                ]):
                    needs_search = True

                if needs_search and not is_random_response:
                    await message.channel.typing()
                    

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
                        await message.reply(botAnswer, mention_author=False)
                    else:
                        await message.reply('申し訳ありません。関連する情報が見つかりませんでした。', mention_author=False)
                elif needs_response:
                    # 検索が不要な場合は、ファイル情報のみを使用
                    if context:
                        response = await chat_with_model(get_bot_model(), messages=[
                            {
                                'role': 'user',
                                'content': (
                                    f'以下のファイル情報を基に、質問に回答してください。\n\n{context}\n\n質問:\n{question}'
                                )
                            }
                        ])
                        botAnswer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                        logger.info(f"Channel ID: {channel_id}, Bot Response (file): {botAnswer}")
                        await message.reply(botAnswer, mention_author=False)
                    else:
                        # 通常の質問応答を実行
                        response = await chat_with_model(get_bot_model(), messages=messages)
                        botAnswer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                        logger.info(f"Channel ID: {channel_id}, Bot Response (normal): {botAnswer}")

                        # メッセージが2000文字を超える場合は分割して送信
                        if len(botAnswer) > 2000:
                            # メッセージを分割
                            chunks = []
                            for i in range(0, len(botAnswer), 1900):  # 余裕を持って1900文字ずつ
                                chunk = botAnswer[i:i + 1900]
                                # コードブロックの途中で分割される場合の処理
                                if '```' in botAnswer:
                                    # コードブロックの開始位置を確認
                                    code_start = chunk.count('```') % 2 == 1
                                    if code_start:
                                        chunk += '\n```'  # コードブロックを閉じる
                                        if i + 1900 < len(botAnswer):  # 次のチャンクがある場合
                                            next_chunk = '```\n'  # 次のチャンクの先頭にコードブロックを開始
                                chunks.append(chunk)

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
