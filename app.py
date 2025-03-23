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

# Kiểm tra GPU
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

# Tải mô hình .gguf
model_path = "./models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=8192,
    max_tokens=7000,
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
                if not detected or detected["confidence"] is None or detected["confidence"] < 0.8 or not detected["encoding"]:
                    print(f"Bỏ qua đọc nội dung file không phải văn bản thuần: {filepath}")
                    all_code.append(f"File: {filepath}\n```\n[Non-text file - Content not readable]\n```")
                    continue
                encoding = detected["encoding"]
                try:
                    code_content = raw_data.decode(encoding)
                except (UnicodeDecodeError, TypeError):
                    try:
                        code_content = raw_data.decode("utf-8", errors="replace")
                        print(f"File {filepath} có ký tự không giải mã được, dùng UTF-8 với thay thế.")
                    except Exception as e:
                        print(f"Không thể giải mã file {filepath}: {e}")
                        all_code.append(f"File: {filepath}\n```\n[Undecodable file - Content not readable]\n```")
                        continue
                
                line_count = code_content.count('\n') + 1
                print(f"File {filepath} có {line_count} dòng.")
                all_code.append(f"File: {filepath}\n```\n{code_content}\n```")
            except (PermissionError, IOError) as e:
                print(f"Không thể đọc file {filepath}: {e}")
                continue
    return all_code

# Chuẩn bị embedding và FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
source_code_dir = "./source_code"
faiss_save_path = "./faiss_index"

# Khởi tạo vector_store
vector_store = None
try:
    if os.path.exists(faiss_save_path):
        vector_store = FAISS.load_local(faiss_save_path, embedding_model, allow_dangerous_deserialization=True)
        print(f"Đã tải FAISS DB từ: {faiss_save_path}")
    else:
        code_documents = scan_source_code(source_code_dir)
        if not code_documents:
            code_documents = ["Không tìm thấy source code. Vui lòng thêm file vào thư mục 'source_code'."]
        
        # Chia nhỏ các tài liệu bằng text_splitter
        split_documents = []
        for doc in code_documents:
            chunks = text_splitter.split_text(doc)
            split_documents.extend(chunks)  # Dùng extend thay vì append để có list phẳng
        print(f"Tổng số chunk sau khi chia: {len(split_documents)}")
        
        vector_store = FAISS.from_texts(split_documents, embedding_model)
        vector_store.save_local(faiss_save_path)
        print(f"FAISS DB đã được tạo và lưu tại: {faiss_save_path}")
except Exception as e:
    print(f"Lỗi khi tạo/tải FAISS DB: {e}")
    vector_store = FAISS.from_texts(["Lỗi hệ thống: Không thể tạo vector store."], embedding_model)

# Tạo RAG chain với FAISS retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 50}),
    return_source_documents=True
)

# Hàm lọc file theo tên trong truy vấn
def filter_relevant_documents(query, documents):
    file_pattern = re.compile(r"(\w+\.(?:js|py|cpp|java|txt))", re.IGNORECASE)
    match = file_pattern.search(query)
    if match:
        target_file = match.group(1)
        filtered_docs = [doc for doc in documents if target_file.lower() in doc.page_content.lower()]
        if filtered_docs:
            return filtered_docs
    return documents

# Route cho giao diện web
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        try:
            result = qa_chain({"query": query})
            analysis = result["result"]
            related_code = filter_relevant_documents(query, result["source_documents"])
            related_code = [doc.page_content for doc in related_code]
            return render_template("index.html", analysis=analysis, related_code=related_code, query=query)
        except Exception as e:
            return render_template("index.html", analysis=f"Lỗi khi xử lý truy vấn: {e}", related_code=[], query=query)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)