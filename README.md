# 🩺 Diabetes Diagnosis + RAG Treatment System

> Hệ thống ML chẩn đoán bệnh tiểu đường tích hợp RAG để gợi ý phác đồ điều trị.

---

## 📁 Cấu Trúc Dự Án

```
KB_RAG/
├── notebooks/
│   └── Diabetes_Diagnosis_Full_Pipeline.ipynb   # Notebook chính
├── models/
│   ├── best_model.pkl          # Mô hình tốt nhất
│   ├── scaler.pkl              # StandardScaler đã fit
│   └── model_metadata.json     # Thông số mô hình
├── rag/
│   ├── vector_store/           # ChromaDB index
│   ├── pdf_processor.py        # Xử lý PDF
│   └── rag_pipeline.py         # Pipeline RAG
├── web/
│   ├── backend/
│   │   ├── app.py              # FastAPI server
│   │   └── requirements.txt
│   └── frontend/
│       ├── index.html          # SPA responsive
│       └── static/
│           ├── css/style.css
│           ├── js/app.js
│           ├── manifest.json   # PWA
│           └── service-worker.js
├── data/                       # Đặt CSV datasets vào đây
├── reports/figures/            # Biểu đồ xuất ra
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Chuẩn Bị Dữ Liệu

Tải datasets từ Kaggle và đặt vào thư mục `data/`:
- [`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- `diabetes_binary_health_indicators_BRFSS2015.csv`

Đặt PDF hướng dẫn điều trị vào thư mục gốc hoặc `data/`:
- `HD-chan-doan-dieu-tri-DTD-2017_07_19-Approved.pdf`

### Bước 2: Chạy Notebook

Mở Jupyter với Anaconda kernel:
```powershell
& "E:\anaconda3\python.exe" -m jupyter notebook notebooks/
```

Chạy toàn bộ notebook `Diabetes_Diagnosis_Full_Pipeline.ipynb` theo thứ tự 6 sections.

### Bước 3: Khởi Động Web App

```powershell
cd web/backend
& "E:\anaconda3\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập: [http://localhost:8000](http://localhost:8000)  
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Bước 4 (Optional): Docker

```bash
docker-compose up --build
```

---

## 🛠️ Stack Công Nghệ

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn, XGBoost, LightGBM |
| **RAG** | sentence-transformers, ChromaDB, PyMuPDF |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, Vanilla JS, Chart.js |
| **PWA** | Web App Manifest, Service Worker |

---

## 📊 Models

| # | Model | Mô tả |
|---|-------|-------|
| 1 | Logistic Regression | Baseline |
| 2 | Decision Tree | Dễ interpretable |
| 3 | Random Forest | Ensemble mạnh |
| 4 | Gradient Boosting | GBM chuẩn |
| 5 | **XGBoost** | State-of-the-art |
| 6 | LightGBM | Nhanh hơn XGB |
| 7 | SVM | Kernel RBF |
| 8 | KNN | Instance-based |
| 9 | Naive Bayes | Probabilistic |

---

## ⚠️ Disclaimer

> **Đây là công cụ hỗ trợ nghiên cứu và học tập, KHÔNG thay thế chẩn đoán của bác sĩ.**  
> Luôn tham khảo ý kiến chuyên gia y tế cho các quyết định sức khỏe.
