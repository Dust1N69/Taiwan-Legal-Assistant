"""
此程序需要使用llama-cpp-python 0.3.0或更高版本來支持Llama 3.2模型。
如果遇到模型載入錯誤，請確保已安裝正確版本：
pip install llama-cpp-python==0.3.0 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
"""

import os
import sys
import gradio as gr
from huggingface_hub import hf_hub_download, scan_cache_dir
from llama_cpp import Llama
import logging
from rag_processor import RAGProcessor

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型配置
MODEL_ID = "QuantFactory/Llama-3.2-Taiwan-Legal-3B-Instruct-GGUF"
MODEL_FILENAME = "Llama-3.2-Taiwan-Legal-3B-Instruct.Q4_K_M.gguf"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

# 初始化RAG處理器
try:
    rag_processor = RAGProcessor("processed_documents")
    USE_RAG = True
    logger.info("RAG處理器初始化成功")
except Exception as e:
    logger.warning(f"RAG處理器初始化失敗: {e}")
    USE_RAG = False

# 檢查模型是否已在 Hugging Face 緩存中
def check_model_in_cache():
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == MODEL_ID:
                for file in repo.revisions[0].files:
                    if file.file_name == MODEL_FILENAME:
                        return os.path.join(file.storage_path)
        return None
    except Exception as e:
        logger.error(f"檢查緩存時發生錯誤: {e}")
        return None

# 下載模型（如果尚未下載）
def download_model():
    if os.path.exists(MODEL_PATH):
        logger.info(f"模型已存在於本地路徑: {MODEL_PATH}")
        return MODEL_PATH
    
    cached_path = check_model_in_cache()
    if cached_path:
        logger.info(f"模型已存在於緩存: {cached_path}")
        return cached_path
    
    try:
        logger.info(f"正在下載模型 {MODEL_FILENAME}...")
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            local_dir=os.getcwd(),
            local_dir_use_symlinks=False
        )
        logger.info(f"模型下載完成！路徑: {path}")
        return path
    except Exception as e:
        logger.error(f"下載模型時發生錯誤: {e}")
        raise

# 初始化模型
def init_model(model_path):
    try:
        logger.info("正在載入模型...")
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=512
        )
        logger.info("模型載入完成！")
        return model
    except Exception as e:
        logger.error(f"載入模型時發生錯誤: {e}")
        raise

# 生成回應
def generate_response(model, prompt, system_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    try:
        # 如果啟用了RAG，使用RAG處理器增強提示
        if USE_RAG:
            try:
                enhanced_prompt = rag_processor.process_query(prompt)
                prompt = enhanced_prompt
            except Exception as e:
                logger.warning(f"RAG處理失敗: {e}")
        
        # 構建完整的提示
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        
        logger.info(f"生成回應，提示長度: {len(full_prompt)}")
        
        # 生成回應
        response = model.create_completion(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=["<|user|>", "<|system|>"],
        )
        
        return response["choices"][0]["text"]
    except Exception as e:
        logger.error(f"生成回應時發生錯誤: {e}")
        return f"生成回應時發生錯誤: {str(e)}"

# 主函數
def main():
    # 檢查llama-cpp-python版本
    try:
        import llama_cpp
        version = llama_cpp.__version__
        version_parts = version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major == 0 and minor < 3:
            logger.warning(f"當前llama-cpp-python版本為{version}，但Llama 3.2模型需要0.3.0或更高版本")
            logger.warning("請使用以下命令安裝兼容版本：")
            logger.warning("pip install llama-cpp-python==0.3.0 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
            print(f"當前llama-cpp-python版本為{version}，但Llama 3.2模型需要0.3.0或更高版本")
            print("請使用以下命令安裝兼容版本：")
            print("pip install llama-cpp-python==0.3.0 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
            sys.exit(1)
    except (ImportError, AttributeError):
        logger.warning("無法檢查llama-cpp-python版本")
    
    # 下載並初始化模型
    try:
        model_path = download_model()
        model = init_model(model_path)
        
        # 創建 Gradio 界面
        with gr.Blocks(title="Llama-3.2-Taiwan-Legal-3B 對話") as demo:
            gr.Markdown("# Llama-3.2-Taiwan-Legal-3B 對話助手")
            gr.Markdown("這是一個基於 Llama-3.2 的台灣法律領域模型，專注於台灣法律相關問題。" + 
                       (" (已啟用RAG增強)" if USE_RAG else ""))
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(label="輸入您的問題", placeholder="請輸入您的問題...", lines=3)
                    with gr.Row():
                        clear = gr.Button("清除對話")
                        submit = gr.Button("送出", variant="primary")
                    
                with gr.Column(scale=1):
                    system_prompt = gr.Textbox(
                        label="系統提示詞 (Prompt Engineering)",
                        placeholder="設定系統提示詞以引導模型的行為...",
                        lines=3,
                        value="你是一個專業的台灣法律助手，基於台灣法律知識回答問題。請提供準確、有幫助的回答。"
                    )
                    
                    with gr.Accordion("進階設定", open=False):
                        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="溫度 (Temperature)")
                        max_tokens = gr.Slider(minimum=64, maximum=4096, value=2048, step=64, label="最大生成長度 (Max Tokens)")
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")
                        frequency_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="頻率懲罰 (Frequency Penalty)")
                        presence_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="存在懲罰 (Presence Penalty)")
                    
                    gr.Markdown("### 提示詞範例")
                    with gr.Accordion("法律顧問", open=False):
                        gr.Markdown("你是一個專業的台灣法律顧問，請根據台灣法律提供專業建議。")
                    with gr.Accordion("法條解釋", open=False):
                        gr.Markdown("你是一個台灣法條解釋專家，請解釋相關法條的含義和適用範圍。")
                    with gr.Accordion("案例分析", open=False):
                        gr.Markdown("你是一個法律案例分析專家，請分析這個案例的法律要點和可能的判決結果。")
            
            # 處理對話
            def respond(message, chat_history, system_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
                if not message.strip():
                    return "", chat_history
                
                # 生成回應
                bot_message = generate_response(
                    model, 
                    message, 
                    system_prompt, 
                    temperature, 
                    max_tokens, 
                    top_p, 
                    frequency_penalty, 
                    presence_penalty
                )
                
                # 更新對話歷史
                chat_history.append((message, bot_message))
                return "", chat_history
            
            # 設置事件處理
            msg.submit(
                respond, 
                [msg, chatbot, system_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty], 
                [msg, chatbot]
            )
            
            submit.click(
                respond, 
                [msg, chatbot, system_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty], 
                [msg, chatbot]
            )
            
            clear.click(lambda: None, None, chatbot, queue=False)
        
        # 啟動 Gradio 界面
        demo.launch(share=True)
        
    except Exception as e:
        logger.error(f"發生錯誤: {e}")
        print(f"發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 