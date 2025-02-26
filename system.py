import asyncio
import rag_log_processing
import os

VERSION = '0.0.1'

async def version():
    return VERSION

async def summarize_logs():
    print("Starting log summarization...")
    input_log = os.path.join(rag_log_processing.LOG_DIR, 'chatbot_log.json')
    output_log = os.path.join(rag_log_processing.LOG_DIR, 'summarized_log.txt')
    await rag_log_processing.log_summarize(input_log, output_log)
    print("Log summarization completed.")

async def main():
    while True:
        await summarize_logs()
        await asyncio.sleep(600)

if __name__ == '__main__':
    asyncio.run(main())