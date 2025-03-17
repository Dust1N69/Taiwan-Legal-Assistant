import os
import fitz  # PyMuPDF
import json
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCUMENT_CATEGORIES = {
    "house_contract": "房屋契約",
    "vehicle": "車輛買賣/租賃",
    "labor": "勞動契約",
    "loan": "借貸契約",
    "service": "服務契約",
    "real_estate": "不動產相關"
}

class DocumentProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processed_docs = []
        
        # 創建輸出目錄
        for category in DOCUMENT_CATEGORIES.keys():
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """從PDF文件中提取文字"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"處理PDF文件時發生錯誤 {pdf_path}: {e}")
            return ""
        finally:
            if 'doc' in locals():
                doc.close()
    
    def process_document(self, file_path: Path, category: str) -> Dict:
        """處理單個文件並返回結構化數據"""
        text = self.extract_text_from_pdf(file_path)

        doc_data = {
            "file_name": file_path.name,
            "category": category,
            "category_zh": DOCUMENT_CATEGORIES[category],
            "content": text,
            "path": str(file_path)
        }
        
        # 保存處理後的文本
        output_path = self.output_dir / category / f"{file_path.stem}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        return doc_data
    
    def process_all_documents(self):
        """處理所有文件"""
        for category in DOCUMENT_CATEGORIES.keys():
            category_path = self.input_dir / category
            if not category_path.exists():
                logger.warning(f"目錄不存在: {category_path}")
                continue
            
            logger.info(f"處理類別 {DOCUMENT_CATEGORIES[category]}")
            for pdf_file in category_path.glob("*.pdf"):
                logger.info(f"處理文件: {pdf_file}")
                doc_data = self.process_document(pdf_file, category)
                self.processed_docs.append(doc_data)
        
        # 保存處理摘要
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_documents": len(self.processed_docs),
                "documents": self.processed_docs
            }, f, ensure_ascii=False, indent=2)

def main():
    # 設置輸入和輸出目錄
    input_dir = Path("D:/Datasets")
    output_dir = Path("processed_documents")
    
    # 創建處理器實例
    processor = DocumentProcessor(input_dir, output_dir)
    
    # 處理所有文件
    processor.process_all_documents()
    logger.info("文件處理完成！")

if __name__ == "__main__":
    main() 
