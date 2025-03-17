import os
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import faiss
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, processed_docs_dir: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.docs_dir = Path(processed_docs_dir)
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.load_documents()
        self.build_index()
    
    def load_documents(self):
        """載入處理過的文件"""
        logger.info("載入文件...")
        summary_path = self.docs_dir / "processing_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"找不到處理摘要文件: {summary_path}")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            self.documents = summary['documents']
        
        logger.info(f"已載入 {len(self.documents)} 份文件")
    
    def build_index(self):
        """建立FAISS索引"""
        logger.info("建立FAISS索引...")
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 初始化FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        logger.info("索引建立完成")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """檢索相關文件"""
        # 編碼查詢
        query_embedding = self.model.encode([query])[0]
        
        # 搜索最相關的文件
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        
        # 返回檢索結果
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.documents[idx]
            results.append({
                "document": doc,
                "score": float(1 / (1 + distance))  # 將距離轉換為相似度分數
            })
        
        return results

    def process_query(self, query: str, k: int = 5) -> str:
        """處理查詢並返回增強的上下文"""
        # 檢索相關文件
        retrieved_docs = self.retrieve(query, k)
        
        # 構建增強的上下文
        context = "以下是相關的法律文件內容：\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"文件 {i} ({doc['document']['category_zh']}):\n"
            context += f"{doc['document']['content'][:500]}...\n\n"
        
        # 構建最終提示
        final_prompt = f"""基於以下相關法律文件的內容，請回答問題：

相關文件：
{context}

問題：{query}

請根據上述文件內容提供專業的法律建議。"""
        
        return final_prompt

def main():
    # 初始化RAG處理器
    rag = RAGProcessor("processed_documents")
    
    # 測試查詢
    test_query = "請說明房屋租賃契約中的押金規定"
    enhanced_prompt = rag.process_query(test_query)
    print(enhanced_prompt)

if __name__ == "__main__":
    main() 
