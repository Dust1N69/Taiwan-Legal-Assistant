import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
from llama_cpp import Llama
from rag_processor import RAGProcessor

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatMemory:
    def __init__(self, memory_file: str = "chat_memory.json"):
        self.memory_file = Path(memory_file)
        self.conversations: Dict[str, List[Dict]] = {}
        self.load_memory()
    
    def load_memory(self):
        """載入對話記憶"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
    
    def save_memory(self):
        """保存對話記憶"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
    
    def add_conversation(self, conversation_id: str, role: str, content: str):
        """添加對話記錄"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()
    
    def get_conversation_history(self, conversation_id: str, max_turns: int = 5) -> List[Dict]:
        """獲取對話歷史"""
        if conversation_id not in self.conversations:
            return []
        return self.conversations[conversation_id][-max_turns:]
    
    def list_conversations(self) -> List[str]:
        """列出所有對話ID"""
        return list(self.conversations.keys())

class ChatInterface:
    def __init__(self, model_path: str, use_rag: bool = True):
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=512
        )
        self.memory = ChatMemory()
        self.rag = RAGProcessor("processed_documents") if use_rag else None
    
    def format_conversation_history(self, history: List[Dict]) -> str:
        """格式化對話歷史"""
        formatted = ""
        for msg in history:
            if msg["role"] == "user":
                formatted += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"<|assistant|>\n{msg['content']}\n"
            elif msg["role"] == "system":
                formatted += f"<|system|>\n{msg['content']}\n"
        return formatted
    
    def generate_response(self, 
                         conversation_id: str,
                         user_input: str,
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: int = 2048) -> str:
        """生成回應"""
        # 獲取對話歷史
        history = self.memory.get_conversation_history(conversation_id)
        
        # 如果是新對話且有系統提示，添加系統提示
        if not history and system_prompt:
            self.memory.add_conversation(conversation_id, "system", system_prompt)
            history = self.memory.get_conversation_history(conversation_id)
        
        # 如果啟用了RAG，使用RAG處理器增強提示
        if self.rag:
            try:
                enhanced_prompt = self.rag.process_query(user_input)
                user_input = enhanced_prompt
            except Exception as e:
                logger.warning(f"RAG處理失敗: {e}")
        
        # 添加用戶輸入到記憶
        self.memory.add_conversation(conversation_id, "user", user_input)
        
        # 構建完整提示
        prompt = self.format_conversation_history(history) + "<|assistant|>\n"
        
        # 生成回應
        response = self.model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|user|>", "<|system|>"]
        )
        
        response_text = response["choices"][0]["text"]
        
        # 添加助手回應到記憶
        self.memory.add_conversation(conversation_id, "assistant", response_text)
        
        return response_text

def main():
    # 模型配置
    MODEL_PATH = "Llama-3.2-Taiwan-Legal-3B-Instruct.Q4_K_M.gguf"
    
    # 系統提示詞範例
    SYSTEM_PROMPTS = {
        "法律顧問": "你是一個專業的台灣法律顧問，請根據台灣法律提供專業建議。在回答時，請先說明相關法律依據，再提供具體建議。",
        "法條解釋": "你是一個台灣法條解釋專家。請詳細解釋法條的含義，包括立法目的、適用範圍和實務見解。",
        "案例分析": "你是一個法律案例分析專家。分析時請考慮：1)案件事實 2)法律爭點 3)適用法條 4)可能的判決結果。"
    }
    
    # 初始化聊天界面
    chat = ChatInterface(MODEL_PATH)
    
    while True:
        print("\n=== 台灣法律助手 ===")
        print("1. 開始新對話")
        print("2. 繼續既有對話")
        print("3. 查看對話歷史")
        print("4. 退出")
        
        choice = input("請選擇操作 (1-4): ")
        
        if choice == "1":
            print("\n=== 選擇系統提示詞 ===")
            for i, (role, prompt) in enumerate(SYSTEM_PROMPTS.items(), 1):
                print(f"{i}. {role}")
            prompt_choice = input("請選擇提示詞類型 (1-3): ")
            
            system_prompt = list(SYSTEM_PROMPTS.values())[int(prompt_choice)-1]
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print("\n開始對話 (輸入 'quit' 結束當前對話)")
            while True:
                user_input = input("\n您的問題: ")
                if user_input.lower() == 'quit':
                    break
                
                response = chat.generate_response(
                    conversation_id=conversation_id,
                    user_input=user_input,
                    system_prompt=system_prompt
                )
                print("\n助手回應:", response)
        
        elif choice == "2":
            conversations = chat.memory.list_conversations()
            if not conversations:
                print("沒有既有對話！")
                continue
            
            print("\n=== 既有對話 ===")
            for i, conv_id in enumerate(conversations, 1):
                print(f"{i}. {conv_id}")
            
            conv_choice = input("請選擇對話 (輸入編號): ")
            conversation_id = conversations[int(conv_choice)-1]
            
            print("\n繼續對話 (輸入 'quit' 結束當前對話)")
            while True:
                user_input = input("\n您的問題: ")
                if user_input.lower() == 'quit':
                    break
                
                response = chat.generate_response(
                    conversation_id=conversation_id,
                    user_input=user_input
                )
                print("\n助手回應:", response)
        
        elif choice == "3":
            conversations = chat.memory.list_conversations()
            if not conversations:
                print("沒有對話歷史！")
                continue
            
            print("\n=== 對話歷史 ===")
            for i, conv_id in enumerate(conversations, 1):
                print(f"\n對話 {i} ({conv_id}):")
                history = chat.memory.get_conversation_history(conv_id)
                for msg in history:
                    role = "系統" if msg["role"] == "system" else "用戶" if msg["role"] == "user" else "助手"
                    print(f"{role}: {msg['content'][:100]}...")
        
        elif choice == "4":
            print("感謝使用！再見！")
            break

if __name__ == "__main__":
    main() 