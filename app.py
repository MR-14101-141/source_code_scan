import os
from flask import Flask, request, render_template
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
import torch
from charset_normalizer import detect
import re

app = Flask(__name__)

# Tải mô hình .gguf
model_path = "./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=8192,
    max_tokens=500,
    temperature=0.7,
    n_gpu_layers=20 if torch.cuda.is_available() else 0,
    verbose=False
)

# Danh sách phần mở rộng nhị phân/hình ảnh
binary_image_extensions = {".exe", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

# Hàm quét source code
def scan_source_code(directory="./source_code"):
    all_code = []
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Đã tạo thư mục {directory}. Vui lòng thêm source code vào đây.")
        return all_code

    for root, dirs, files in os.walk(directory):
        if "node_modules" in dirs:
            dirs.remove("node_modules")
        
        for file in files:
            filepath = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            try:
                if file_ext in binary_image_extensions:
                    all_code.append(f"File: {filepath}\n```\n[Binary/Image file - Content not readable]\n```")
                    continue
                
                with open(filepath, "rb") as f:
                    raw_data = f.read()
                
                detected = detect(raw_data)
                # Kiểm tra nếu detected không hợp lệ
                if not detected or detected["confidence"] is None or detected["confidence"] < 0.8 or not detected["encoding"]:
                    print(f"Bỏ qua đọc nội dung file không phải văn bản thuần: {filepath}")
                    all_code.append(f"File: {filepath}\n```\n[Non-text file - Content not readable]\n```")
                    continue
                
                encoding = detected["encoding"]
                try:
                    code_content = raw_data.decode(encoding)
                except (UnicodeDecodeError, TypeError):
                    try:
                        code_content = raw_data.decode("utf-8")
                    except UnicodeDecodeError:
                        print(f"Không thể giải mã file {filepath}, chỉ lưu đường dẫn.")
                        all_code.append(f"File: {filepath}\n```\n[Undecodable file - Content not readable]\n```")
                        continue

                print(f"Đã thêm data vào database {filepath}")
                all_code.append(f"File: {filepath}\n```\n{code_content}\n```")
            except (PermissionError, IOError) as e:
                print(f"Không thể đọc file {filepath}: {e}")
                continue
    
    return all_code

# Hàm lọc file theo tên trong truy vấn
def filter_relevant_documents(query, documents):
    file_pattern = re.compile(r"(\w+\.(?:js|py|cpp|java|txt))", re.IGNORECASE)
    match = file_pattern.search(query)
    if match:
        target_file = match.group(1)
        # Lọc tất cả tài liệu chứa tên file
        filtered_docs = [doc for doc in documents if target_file.lower() in doc.page_content.lower()]
        if filtered_docs:
            return filtered_docs  # Trả về tất cả tài liệu khớp
    return documents  # Nếu không tìm thấy, trả về tất cả tài liệu từ retriever

# Chuẩn bị embedding và vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
source_code_dir = "./source_code"
code_documents = scan_source_code(source_code_dir)

if not code_documents:
    code_documents = ["Không tìm thấy source code. Vui lòng thêm file vào thư mục 'source_code'."]

vector_store = FAISS.from_texts(code_documents, embedding_model)

# Lưu FAISS index ra file
faiss_save_path = "./faiss_index"
vector_store.save_local(faiss_save_path)
print(f"FAISS index đã được lưu tại: {faiss_save_path}")

# Tạo RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
    return_source_documents=True
)

# Route cho giao diện web
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        result = qa_chain({"query": query})
        analysis = result["result"]
        # Lọc tài liệu liên quan dựa trên tên file trong truy vấn
        related_code = filter_relevant_documents(query, result["source_documents"])
        related_code = [doc.page_content for doc in related_code]
        return render_template("index.html", analysis=analysis, related_code=related_code, query=query)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)