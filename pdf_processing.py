from io import StringIO
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
import os
import tempfile
from contextlib import contextmanager
from utils.logger import setup_logger
from chat_processing import chat_with_model

VERSION = '0.0.2'

logger = setup_logger(__name__, 'pdf_processing.log')

@contextmanager
def temp_pdf_file(pdf_data, prefix='temp_'):
    """一時PDFファイルを安全に管理するコンテキストマネージャ"""
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix='.pdf', delete=False) as temp_file:
        try:
            temp_file.write(pdf_data)
            temp_file.flush()
            yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

def extract_text_safely(pdf_path):
    """PDFからテキストを安全に抽出する"""
    rsrcmgr = PDFResourceManager()
    retstr = []
    
    try:
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed(
                    "このPDFからテキストを抽出することはできません。")
            
            laparams = LAParams()
            
            for page in PDFPage.create_pages(document):
                output_string = StringIO()
                converter = TextConverter(
                    rsrcmgr, output_string, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, converter)
                
                try:
                    interpreter.process_page(page)
                    page_text = output_string.getvalue()
                    retstr.append(page_text)
                finally:
                    converter.close()
                    output_string.close()
        
        return '\n'.join(retstr)
        
    except Exception as e:
        logger.error(f"PDFテキスト抽出エラー: {str(e)}")
        raise

async def process_pdf_with_ollama(pdf_file_path, question, bot_model):
    try:
        text = extract_text_safely(pdf_file_path)
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
    except Exception as e:
        logger.error(f"PDF処理エラー: {str(e)}")
        raise

async def get_pdfText(attachment):
    """PDFファイルからテキストを抽出する"""
    try:
        pdf_data = await attachment.read()
        
        with temp_pdf_file(pdf_data) as temp_path:
            return extract_text_safely(temp_path)
            
    except PDFTextExtractionNotAllowed:
        logger.error("PDFからテキストを抽出できません")
        raise ValueError("このPDFファイルはテキスト抽出が許可されていません")
    except Exception as e:
        logger.error(f"PDF処理エラー: {str(e)}")
        raise
