"""
app.py — FastAPI Backend cho hệ thống chẩn đoán tiểu đường + RAG
Endpoints:
  POST /api/predict   → ML diagnosis
  POST /api/recommend → RAG treatment plan
  GET  /api/health    → health check
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Thêm thư mục rag vào sys.path
RAG_DIR = Path(__file__).parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load ML Models ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
_model = None
_scaler = None
_metadata = {}


def load_models():
    global _model, _scaler, _metadata
    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"

    if model_path.exists() and scaler_path.exists():
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        logger.info(f"✅ Đã load model: {model_path}")
    else:
        logger.warning("⚠️ Chưa có model file! Hãy chạy notebook để train và save model.")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


load_models()

# ─── Load RAG Pipeline ───────────────────────────────────────────────────────
_rag = None
try:
    from rag_pipeline import get_rag_pipeline
    _rag = get_rag_pipeline()
    if _rag.is_ready():
        logger.info(f"✅ RAG pipeline ready. Index có {_rag.collection.count()} docs")
    else:
        pdf_path = Path(__file__).parent.parent.parent / "data" / "HD-chan-doan-dieu-tri-DTD-2017_07_19-Approved.pdf"
        if pdf_path.exists():
            logger.info(f"🔨 Đang tự động build RAG index từ: {pdf_path.name}...")
            _rag.build_index(str(pdf_path))
            logger.info(f"✅ RAG pipeline ready. Index có {_rag.collection.count()} docs")
        else:
            logger.warning(f"⚠️ RAG index đang trống và không tìm thấy file PDF tại: {pdf_path}")
except Exception as e:
    logger.warning(f"⚠️ RAG pipeline không khởi động được: {e}")


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="🩺 Hệ thống Chẩn đoán Tiểu đường",
    description="API chẩn đoán tiểu đường bằng ML + gợi ý phác đồ điều trị bằng RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"❌ Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ─── Pydantic Models ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# Các thuộc tính số liên tục được chuẩn hóa bằng StandardScaler trong pipeline huấn luyện
SCALED_FEATURES = ["BMI", "MentHlth", "PhysHlth", "Age"]
SCALED_INDICES = [FEATURE_NAMES.index(f) for f in SCALED_FEATURES]



class PredictRequest(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="Huyết áp cao (0/1)")
    HighChol: int = Field(..., ge=0, le=1, description="Cholesterol cao (0/1)")
    CholCheck: int = Field(1, ge=0, le=1, description="Kiểm tra cholesterol trong 5 năm (0/1)")
    BMI: float = Field(..., ge=10, le=60, description="Chỉ số BMI")
    Smoker: int = Field(..., ge=0, le=1, description="Hút thuốc (0/1)")
    Stroke: int = Field(0, ge=0, le=1, description="Đột quỵ (0/1)")
    HeartDiseaseorAttack: int = Field(0, ge=0, le=1, description="Bệnh tim (0/1)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Hoạt động thể chất (0/1)")
    Fruits: int = Field(..., ge=0, le=1, description="Ăn trái cây hàng ngày (0/1)")
    Veggies: int = Field(..., ge=0, le=1, description="Ăn rau xanh hàng ngày (0/1)")
    HvyAlcoholConsump: int = Field(0, ge=0, le=1, description="Uống rượu nhiều (0/1)")
    AnyHealthcare: int = Field(1, ge=0, le=1, description="Có bảo hiểm y tế (0/1)")
    NoDocbcCost: int = Field(0, ge=0, le=1, description="Không thể đi khám vì chi phí (0/1)")
    GenHlth: int = Field(..., ge=1, le=5, description="Sức khỏe tổng quát (1-5: Excellent→Poor)")
    MentHlth: int = Field(0, ge=0, le=30, description="Ngày sức khỏe tâm thần kém/30 ngày")
    PhysHlth: int = Field(0, ge=0, le=30, description="Ngày sức khỏe thể chất kém/30 ngày")
    DiffWalk: int = Field(0, ge=0, le=1, description="Khó đi lại (0/1)")
    Sex: int = Field(..., ge=0, le=1, description="Giới tính (0=Nữ, 1=Nam)")
    Age: int = Field(..., ge=1, le=13, description="Nhóm tuổi (1=18-24 đến 13=80+)")
    Education: int = Field(..., ge=1, le=6, description="Trình độ học vấn (1-6)")
    Income: int = Field(..., ge=1, le=8, description="Thu nhập (1-8)")


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    message: str


class RecommendRequest(BaseModel):
    prediction: int
    probability: float
    patient_features: dict


class RecommendResponse(BaseModel):
    treatment_plan: str
    lifestyle_tips: str
    sources: list
    rag_available: bool


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Serve frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "🩺 Diabetes Diagnosis API", "docs": "/docs"}


@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health():
    """Health check endpoint."""
    model_name = _metadata.get("best_model", {}).get("name") or _metadata.get("model_name", "Gradient Boosting")
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "rag_ready": _rag is not None and _rag.is_ready(),
        "model_info": model_name,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(data: PredictRequest):
    """Chẩn đoán tiểu đường từ dữ liệu bệnh nhân."""
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa sẵn sàng. Vui lòng chạy notebook để train và save model trước."
        )

    # Chuẩn bị features theo đúng thứ tự
    features = [getattr(data, name) for name in FEATURE_NAMES]
    X = np.array(features, dtype=float).reshape(1, -1)

    # Scale các đặc trưng số theo đúng pipeline lúc huấn luyện (BMI, MentHlth, PhysHlth, Age)
    X_scaled = X.copy()
    if getattr(_scaler, "n_features_in_", None) == len(SCALED_INDICES):
        X_scaled[:, SCALED_INDICES] = _scaler.transform(X[:, SCALED_INDICES])
    elif getattr(_scaler, "n_features_in_", None) == len(FEATURE_NAMES):
        X_scaled = _scaler.transform(X)
    else:
        X_scaled[:, SCALED_INDICES] = _scaler.transform(X[:, SCALED_INDICES])

    # Predict
    prediction = int(_model.predict(X_scaled)[0])
    probability = float(_model.predict_proba(X_scaled)[0][1])

    # Phân mức nguy cơ
    if probability < 0.3:
        risk_level = "Low"
        message = "Nguy cơ thấp. Duy trì lối sống lành mạnh."
    elif probability < 0.6:
        risk_level = "Medium"
        message = "Nguy cơ trung bình. Nên kiểm tra sức khỏe định kỳ."
    else:
        risk_level = "High"
        message = "Nguy cơ cao. Hãy đến gặp bác sĩ sớm để được tư vấn."

    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
        message=message,
    )



@app.post("/api/build-index")
async def build_index():
    """Build RAG vector index từ PDF (chạy 1 lần sau khi deploy)."""
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline chưa khởi động được.")

    PDF_PATH = Path(__file__).parent.parent.parent / "data" / "HD-chan-doan-dieu-tri-DTD-2017_07_19-Approved.pdf"
    if not PDF_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file PDF: {PDF_PATH}")

    try:
        logger.info(f"🔨 Bắt đầu build RAG index từ: {PDF_PATH}")
        _rag.build_index(str(PDF_PATH))
        count = _rag.collection.count()
        logger.info(f"✅ Build index xong: {count} chunks")
        return {"status": "ok", "message": f"Build index thành công! {count} chunks.", "count": count}
    except Exception as e:
        logger.error(f"❌ Lỗi build index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/build-index")
async def build_index_status():
    """Kiểm tra trạng thái RAG index."""
    if _rag is None:
        return {"rag_ready": False, "count": 0, "message": "RAG pipeline chưa khởi động"}
    count = _rag.collection.count() if _rag.collection else 0
    return {"rag_ready": _rag.is_ready(), "count": count, "message": f"Index có {count} documents"}


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(data: RecommendRequest):
    """Gợi ý phác đồ điều trị dựa trên kết quả chẩn đoán và RAG."""

    # Tạo query từ thông tin bệnh nhân
    bmi = data.patient_features.get("BMI", "N/A")
    age_group = data.patient_features.get("Age", "N/A")
    risk = "cao" if data.probability >= 0.6 else "trung bình" if data.probability >= 0.3 else "thấp"

    # Lifestyle tips cơ bản
    lifestyle_tips = _get_lifestyle_tips(data.prediction, data.probability, data.patient_features)

    # RAG query
    treatment_plan = ""
    sources = []
    rag_available = False

    if _rag is not None and _rag.is_ready():
        query = f"Phác đồ điều trị và kiểm soát đái tháo đường type 2 cho bệnh nhân nguy cơ {risk}"
        result = _rag.query(query)
        treatment_plan = result["answer"]
        sources = result["sources"]
        rag_available = True
    else:
        treatment_plan = _get_template_treatment(data.prediction, data.probability)
        rag_available = False

    return RecommendResponse(
        treatment_plan=treatment_plan,
        lifestyle_tips=lifestyle_tips,
        sources=sources,
        rag_available=rag_available,
    )


def _get_lifestyle_tips(prediction: int, probability: float, features: dict) -> str:
    tips = []

    if features.get("BMI", 25) > 25:
        tips.append("• 🏋️ Giảm cân về mức BMI bình thường (18.5-24.9) bằng chế độ ăn và tập luyện")
    if not features.get("PhysActivity", 1):
        tips.append("• 🚶 Tăng cường hoạt động thể chất ít nhất 150 phút/tuần (đi bộ, bơi lội, đạp xe)")
    if not features.get("Fruits", 1) or not features.get("Veggies", 1):
        tips.append("• 🥗 Ăn nhiều rau xanh và trái cây, hạn chế đường và tinh bột tinh chế")
    if features.get("Smoker", 0):
        tips.append("• 🚭 Bỏ thuốc lá - yếu tố nguy cơ quan trọng của tiểu đường và bệnh tim mạch")
    if features.get("HvyAlcoholConsump", 0):
        tips.append("• 🍷 Hạn chế rượu bia")
    if features.get("HighBP", 0):
        tips.append("• 💊 Kiểm soát huyết áp theo chỉ định bác sĩ, mục tiêu <130/80 mmHg")
    if features.get("HighChol", 0):
        tips.append("• 🧪 Kiểm tra và kiểm soát mỡ máu định kỳ")

    if not tips:
        tips.append("• ✅ Duy trì lối sống lành mạnh hiện tại")
        tips.append("• 🏥 Kiểm tra sức khỏe định kỳ 6 tháng/lần")

    tips.append("• 😴 Ngủ đủ 7-8 giờ/đêm, giảm stress")
    tips.append("• 💧 Uống đủ nước (1.5-2 lít/ngày)")

    return "\n".join(tips)


def _get_template_treatment(prediction: int, probability: float) -> str:
    if prediction == 0 and probability < 0.3:
        return (
            "📋 **Phác đồ dự phòng (nguy cơ thấp):**\n"
            "• Duy trì cân nặng lý tưởng\n"
            "• Kiểm tra đường huyết định kỳ 1 năm/lần\n"
            "• Chế độ ăn cân bằng, tập thể dục thường xuyên\n\n"
            "⚠️ *Lưu ý: Đây là thông tin tham khảo, vui lòng gặp bác sĩ để được tư vấn cụ thể.*"
        )
    elif probability < 0.6:
        return (
            "📋 **Phác đồ tiền tiểu đường (nguy cơ trung bình):**\n"
            "• Kiểm tra HbA1c và glucose máu mỗi 3-6 tháng\n"
            "• Giảm 5-7% cân nặng nếu thừa cân\n"
            "• Chương trình can thiệp lối sống có cấu trúc\n"
            "• Cân nhắc Metformin nếu có chỉ định của bác sĩ\n\n"
            "⚠️ *Lưu ý: Đây là thông tin tham khảo, vui lòng gặp bác sĩ để được tư vấn cụ thể.*"
        )
    else:
        return (
            "📋 **Phác đồ điều trị tiểu đường (nguy cơ cao):**\n"
            "• **Điều trị không dùng thuốc**: Chế độ ăn kiêng, tập luyện 150 phút/tuần\n"
            "• **Thuốc hạ đường huyết đường uống**: Metformin (bước đầu tiên theo khuyến cáo)\n"
            "• **Mục tiêu HbA1c**: < 7% (có thể điều chỉnh theo từng bệnh nhân)\n"
            "• **Mục tiêu glucose**: Lúc đói 4.4-7.2 mmol/L; Sau ăn 2h < 10 mmol/L\n"
            "• Theo dõi đường huyết tại nhà, tái khám định kỳ\n\n"
            "⚠️ *Lưu ý: Đây là thông tin tham khảo, vui lòng gặp bác sĩ để được tư vấn cụ thể.*"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
