import asyncio
import os
from dotenv import load_dotenv
from image_processing import generate_image_with_gemini, init_gemini_client
from utils.logger import setup_logger

# ロガーの設定
logger = setup_logger(__name__, 'test_image_gen.log')

async def test_image_generation():
    """画像生成のテストを実行"""
    test_cases = [
        "写実的な猫の肖像画",
        "未来都市の夜景",
        "緑豊かな森の中の小さな家",
    ]
    
    print("\n画像生成テストを開始します...")
    success = 0
    failed = 0
    
    for prompt in test_cases:
        try:
            print(f"\nテストケース: {prompt}")
            print(f"プロンプト: {prompt}")
            images = await generate_image_with_gemini(prompt)
            
            if images and len(images) > 0:
                print(f"✅ 成功: {len(images)}枚の画像が生成されました")
                success += 1
            else:
                print("❌ 失敗: 画像が生成されませんでした")
                failed += 1
                
        except ValueError as ve:
            print(f"❌ バリデーションエラー: {str(ve)}")
            logger.error(f"バリデーションエラー: {ve}")
            failed += 1
        except Exception as e:
            print(f"❌ エラー: {str(e)}")
            logger.error(f"予期せぬエラー: {e}")
            failed += 1
    
    print(f"\nテスト結果: 成功 {success}, 失敗 {failed}")
    print(f"成功率: {(success / len(test_cases)) * 100:.1f}%")
    logger.info(f"テスト完了 - 成功: {success}, 失敗: {failed}")

async def main():
    """メイン処理"""
    # 環境変数の読み込み
    load_dotenv()
    
    # Gemini APIの初期化
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        error_msg = "GEMINI_API_KEYが環境変数に設定されていません"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # クライアントの初期化
        print("Gemini APIクライアントを初期化中...")
        await init_gemini_client(api_key)
        print("APIクライアントの初期化が完了しました")
        logger.info("APIクライアントの初期化が完了しました")
    except ValueError as e:
        error_msg = f"APIの初期化に失敗しました: {e}"
        print(error_msg)
        logger.error(error_msg)
        return

    try:
        # テストの実行
        await test_image_generation()
    except Exception as e:
        error_msg = f"テスト実行中にエラーが発生しました: {e}"
        print(error_msg)
        logger.error(error_msg)

if __name__ == "__main__":
    asyncio.run(main())
