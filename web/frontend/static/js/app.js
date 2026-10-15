/**
 * app.js — Diabetes Diagnosis SPA Logic
 * Vanilla JS, no frameworks
 */

const API_BASE = '';  // Same origin (FastAPI serves frontend)
// const API_BASE = 'http://localhost:8000'; // Uncomment nếu dev riêng

// ── State ────────────────────────────────────────────────
let currentResult = null;
let radarChart = null;
let gaugeChart = null;

// ── Age labels map ───────────────────────────────────────
const AGE_LABELS = {
    1:'18–24', 2:'25–29', 3:'30–34', 4:'35–39', 5:'40–44',
    6:'45–49', 7:'50–54', 8:'55–59', 9:'60–64', 10:'65–69',
    11:'70–74', 12:'75–79', 13:'80+'
};

const GENHLT_LABELS = {
    1:'Tuyệt vời', 2:'Rất tốt', 3:'Tốt', 4:'Khá', 5:'Kém'
};

const EDU_LABELS = {
    1:'Tiểu học', 2:'THCS', 3:'THPT', 4:'Trung cấp/CĐ', 5:'Đại học', 6:'Sau ĐH'
};

const INCOME_LABELS = {
    1:'<\$10K', 2:'\$10–15K', 3:'\$15–20K', 4:'\$20–25K',
    5:'\$25–35K', 6:'\$35–50K', 7:'\$50–75K', 8:'≥\$75K'
};

// ── DOM Ready ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    initTabs();
    initForm();
    initAccordion();
    initHistory();
    checkServerHealth();
});

// ── Health Check ─────────────────────────────────────────
async function checkServerHealth() {
    const statusEl = document.getElementById('serverStatus');
    try {
        const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();

        if (data.status === 'ok') {
            const modelOk = data.model_loaded;
            const ragOk = data.rag_ready;
            statusEl.innerHTML = `
                <span class="dot ${modelOk ? 'dot-green' : 'dot-red'}"></span>
                ${modelOk ? `Model: ${data.model_info || 'OK'}` : 'Model chưa sẵn sàng'}
                &nbsp;|&nbsp;
                RAG: ${ragOk ? '✅' : '⏳'}
            `;
        }
    } catch (e) {
        statusEl.innerHTML = '<span class="dot dot-red"></span> Server offline';
    }
}

// ── Slider Init ──────────────────────────────────────────
function initSliders() {
    const sliderConfigs = [
        { id: 'BMI', badgeId: 'bmiValue', format: v => parseFloat(v).toFixed(1) },
        { id: 'GenHlth', badgeId: 'genHlthLabel', format: v => GENHLT_LABELS[v] || v },
        { id: 'Age', badgeId: 'ageLabel', format: v => AGE_LABELS[v] || v },
        { id: 'MentHlth', badgeId: 'mentHlthValue', format: v => v },
        { id: 'PhysHlth', badgeId: 'physHlthValue', format: v => v },
        { id: 'Income', badgeId: 'incomeLabel', format: v => INCOME_LABELS[v] || `Nhóm ${v}` },
        { id: 'Education', badgeId: 'eduLabel', format: v => EDU_LABELS[v] || v },
    ];

    sliderConfigs.forEach(({ id, badgeId, format }) => {
        const slider = document.getElementById(id);
        const badge = document.getElementById(badgeId);
        if (!slider || !badge) return;

        const update = () => { badge.textContent = format(slider.value); };
        update(); // init
        slider.addEventListener('input', update);
    });
}

// ── Tabs ─────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`)?.classList.add('active');
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-panel').forEach(p => {
        p.classList.toggle('active', p.id === `tab-${tabId}`);
    });
}

// ── Form Handling ────────────────────────────────────────
function initForm() {
    document.getElementById('diagnosisForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await runDiagnosis();
    });

    document.getElementById('resetBtn')?.addEventListener('click', () => {
        document.getElementById('diagnosisForm').reset();
        initSliders(); // re-init badges
    });

    document.getElementById('getRecommendBtn')?.addEventListener('click', getRecommendation);

    document.getElementById('saveResultBtn')?.addEventListener('click', () => {
        if (currentResult) {
            saveToHistory(currentResult);
            showToast('✅ Đã lưu kết quả vào lịch sử!');
        }
    });
}

function collectFormData() {
    const form = document.getElementById('diagnosisForm');
    const data = {};

    // Sliders
    ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income'].forEach(name => {
        const el = document.getElementById(name);
        if (el) data[name] = parseFloat(el.value);
    });

    // Checkboxes (binary)
    ['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack',
     'PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare',
     'NoDocbcCost','DiffWalk','Sex'].forEach(name => {
        const el = document.getElementById(name);
        data[name] = el && el.checked ? 1 : 0;
    });

    return data;
}

async function runDiagnosis() {
    showLoading('Đang phân tích dữ liệu...');

    const formData = collectFormData();

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        const rawText = await res.text();
        let data;
        try {
            data = JSON.parse(rawText);
        } catch {
            data = null;
        }

        if (!res.ok) {
            const errMsg = (data && data.detail) ? data.detail : (rawText.substring(0, 200) || `HTTP ${res.status}`);
            throw new Error(errMsg);
        }

        const result = data || {};
        currentResult = { ...result, patient_features: formData, timestamp: new Date().toISOString() };


        hideLoading();
        renderResult(currentResult);
        switchTab('result');
        showToast('✅ Chẩn đoán hoàn thành!');

    } catch (err) {
        hideLoading();
        showToast(`❌ Lỗi: ${err.message}`);
        console.error(err);

        // Demo mode nếu server offline
        if (err.message.includes('fetch') || err.message.includes('offline') || err.name === 'TimeoutError') {
            runDemoMode(formData);
        }
    }
}

function runDemoMode(formData) {
    // Tính probability giản lược (demo)
    let score = 0;
    if (formData.HighBP) score += 0.12;
    if (formData.HighChol) score += 0.10;
    if (formData.BMI > 30) score += 0.15;
    if (formData.BMI > 35) score += 0.08;
    if (formData.Smoker) score += 0.06;
    if (formData.PhysActivity === 0) score += 0.08;
    if (formData.GenHlth >= 4) score += 0.10;
    if (formData.Age >= 9) score += 0.12;
    if (formData.HeartDiseaseorAttack) score += 0.08;
    if (formData.Stroke) score += 0.05;

    const probability = Math.min(0.95, Math.max(0.05, score));
    const prediction = probability >= 0.5 ? 1 : 0;
    const risk_level = probability >= 0.6 ? 'High' : probability >= 0.3 ? 'Medium' : 'Low';
    const message = risk_level === 'High'
        ? 'Nguy cơ cao. Hãy đến gặp bác sĩ sớm.'
        : risk_level === 'Medium'
            ? 'Nguy cơ trung bình. Nên kiểm tra định kỳ.'
            : 'Nguy cơ thấp. Duy trì lối sống lành mạnh.';

    currentResult = { prediction, probability, risk_level, message, patient_features: formData, timestamp: new Date().toISOString(), demo: true };
    renderResult(currentResult);
    switchTab('result');
    showToast('⚡ Chế độ demo (server offline)');
}

// ── Render Result ─────────────────────────────────────────
function renderResult(result) {
    document.getElementById('resultEmpty').classList.add('hidden');
    document.getElementById('resultContainer').classList.remove('hidden');

    const { prediction, probability, risk_level, message } = result;
    const pct = Math.round(probability * 100);

    // Risk card
    const riskCard = document.getElementById('riskCard');
    riskCard.className = `risk-card ${risk_level.toLowerCase()}`;

    const icons = { Low: '✅', Medium: '⚠️', High: '🚨' };
    document.getElementById('riskIcon').textContent = icons[risk_level] || '❓';
    document.getElementById('riskTitle').textContent = prediction === 1
        ? `Có nguy cơ mắc tiểu đường (${pct}%)`
        : `Ít có nguy cơ mắc tiểu đường (${pct}%)`;
    document.getElementById('riskMessage').textContent = message + (result.demo ? ' [DEMO MODE]' : '');

    const badge = document.getElementById('riskBadge');
    badge.className = `risk-badge ${risk_level}`;
    badge.textContent = { Low: 'Thấp', Medium: 'Trung bình', High: 'Cao' }[risk_level];

    // Gauge
    document.getElementById('gaugeValue').textContent = `${pct}%`;
    renderGauge(probability);

    // Radar
    renderRadar(result.patient_features || {});
}

// ── Gauge Chart ───────────────────────────────────────────
function renderGauge(probability) {
    const ctx = document.getElementById('gaugeChart').getContext('2d');
    if (gaugeChart) { gaugeChart.destroy(); }

    const pct = probability * 100;
    const remaining = 100 - pct;

    const color = probability >= 0.6 ? '#dc2626' : probability >= 0.3 ? '#d97706' : '#16a34a';

    gaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [pct, remaining],
                backgroundColor: [color, '#e2e8f0'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
            }]
        },
        options: {
            responsive: true,
            cutout: '75%',
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            animation: { duration: 800 },
        }
    });
}

// ── Radar Chart ───────────────────────────────────────────
function renderRadar(features) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    if (radarChart) { radarChart.destroy(); }

    // Chuẩn hóa các feature về 0-1 để hiển thị
    const bmiNorm = Math.min((features.BMI - 10) / 50, 1);
    const ageNorm = Math.min((features.Age - 1) / 12, 1);
    const genHlthNorm = Math.min((features.GenHlth - 1) / 4, 1);
    const mentNorm = Math.min(features.MentHlth / 30, 1);
    const physHlthNorm = Math.min(features.PhysHlth / 30, 1);

    // Average population (trung bình tham khảo)
    const avgData = [0.38, 0.35, 0.4, 0.15, 0.15];
    const patientData = [bmiNorm, ageNorm, genHlthNorm, mentNorm, physHlthNorm];

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['BMI', 'Tuổi', 'Sức khỏe tổng quát', 'TT Tâm thần', 'TT Thể chất'],
            datasets: [
                {
                    label: 'Bệnh nhân',
                    data: patientData,
                    backgroundColor: 'rgba(37,99,235,0.15)',
                    borderColor: '#2563eb',
                    borderWidth: 2,
                    pointBackgroundColor: '#2563eb',
                },
                {
                    label: 'Trung bình',
                    data: avgData,
                    backgroundColor: 'rgba(100,116,139,0.1)',
                    borderColor: '#94a3b8',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    pointBackgroundColor: '#94a3b8',
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    min: 0, max: 1,
                    ticks: { display: false },
                    grid: { color: 'rgba(0,0,0,0.05)' },
                    pointLabels: { font: { size: 11 } },
                }
            },
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 12 } } }
            },
            animation: { duration: 800 },
        }
    });
}

// ── RAG Recommendation ────────────────────────────────────
async function getRecommendation() {
    if (!currentResult) {
        showToast('⚠️ Chưa có kết quả chẩn đoán!');
        return;
    }

    showLoading('Đang tìm kiếm phác đồ điều trị...');

    try {
        const payload = {
            prediction: currentResult.prediction,
            probability: currentResult.probability,
            patient_features: currentResult.patient_features || {},
        };

        const res = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const rawText = await res.text();
        let data;
        try { data = JSON.parse(rawText); } catch { data = null; }

        if (!res.ok) {
            const errMsg = (data && data.detail) ? data.detail : (rawText.substring(0, 200) || `HTTP ${res.status}`);
            throw new Error(errMsg);
        }

        hideLoading();
        renderTreatment(data || {});
        switchTab('treatment');

    } catch (err) {
        hideLoading();
        // Demo fallback
        renderTreatmentDemo();
        switchTab('treatment');
        showToast('⚡ Hiển thị phác đồ mẫu (server offline)');
    }
}

function renderTreatment(data) {
    document.getElementById('treatmentEmpty').classList.add('hidden');
    document.getElementById('treatmentContainer').classList.remove('hidden');

    document.getElementById('treatmentPlan').textContent = data.treatment_plan || 'Không có dữ liệu.';
    document.getElementById('lifestyleTips').textContent = data.lifestyle_tips || 'Không có dữ liệu.';

    const sourcesEl = document.getElementById('sources');
    if (data.sources && data.sources.length > 0) {
        sourcesEl.innerHTML = data.sources.map((s, i) =>
            `<div style="margin-bottom:0.75rem; padding:0.75rem; background:#f8fafc; border-radius:8px; border-left:3px solid #2563eb;">
                <div style="font-size:0.78rem; color:#64748b; margin-bottom:0.25rem;">
                    [${i+1}] Độ liên quan: ${(s.relevance_score * 100).toFixed(0)}% | Nguồn: ${s.source}
                </div>
                <div style="font-size:0.85rem;">${s.text}</div>
            </div>`
        ).join('');
        document.getElementById('sourcesItem').style.display = '';
    } else {
        document.getElementById('sourcesItem').style.display = 'none';
    }

    const ragBadge = document.getElementById('ragBadge');
    ragBadge.className = `rag-badge ${data.rag_available ? 'rag-on' : 'rag-off'}`;
    ragBadge.textContent = data.rag_available
        ? '📚 RAG: Truy xuất từ hướng dẫn điều trị'
        : '📋 Template response (RAG chưa build index)';
}

function renderTreatmentDemo() {
    const p = currentResult?.probability || 0.5;
    const plan = p >= 0.6
        ? '📋 Phác đồ điều trị tiểu đường:\n• Khám bác sĩ ngay để xét nghiệm HbA1c, glucose huyết tương\n• Điều trị không dùng thuốc: chế độ ăn, tập luyện\n• Cân nhắc Metformin theo chỉ định bác sĩ\n• Mục tiêu HbA1c < 7%'
        : p >= 0.3
            ? '📋 Dự phòng tiểu đường:\n• Kiểm tra đường huyết định kỳ 6 tháng/lần\n• Giảm cân nếu thừa cân\n• Tăng hoạt động thể chất'
            : '📋 Duy trì sức khỏe:\n• Kiểm tra sức khỏe định kỳ 1 năm/lần\n• Chế độ ăn cân bằng\n• Tập thể dục thường xuyên';

    const lifestyle = '• 🏃 Vận động ≥150 phút/tuần\n• 🥗 Ăn nhiều rau xanh, hạn chế đường\n• 💧 Uống đủ nước\n• 😴 Ngủ 7-8 tiếng/đêm\n• 🚭 Không hút thuốc';

    renderTreatment({ treatment_plan: plan, lifestyle_tips: lifestyle, sources: [], rag_available: false });
}

// ── Accordion ─────────────────────────────────────────────
function initAccordion() {
    document.querySelectorAll('.accordion-header').forEach(header => {
        header.addEventListener('click', () => {
            const targetId = header.dataset.target;
            const body = document.getElementById(targetId);
            const item = header.closest('.accordion-item');

            const isOpen = item.classList.contains('open');
            item.classList.toggle('open', !isOpen);
            body.classList.toggle('collapsed', isOpen);
            header.querySelector('.accordion-arrow').textContent = isOpen ? '▶' : '▼';
        });
    });
}

// ── History ───────────────────────────────────────────────
function initHistory() {
    document.getElementById('clearHistoryBtn')?.addEventListener('click', () => {
        if (confirm('Xóa toàn bộ lịch sử chẩn đoán?')) {
            localStorage.removeItem('diagnosisHistory');
            renderHistory();
            showToast('🗑️ Đã xóa lịch sử');
        }
    });
    renderHistory();
}

function saveToHistory(result) {
    const history = JSON.parse(localStorage.getItem('diagnosisHistory') || '[]');
    history.unshift({
        timestamp: result.timestamp || new Date().toISOString(),
        prediction: result.prediction,
        probability: result.probability,
        risk_level: result.risk_level,
        message: result.message,
    });
    // Giữ tối đa 20 records
    if (history.length > 20) history.splice(20);
    localStorage.setItem('diagnosisHistory', JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('diagnosisHistory') || '[]');
    const listEl = document.getElementById('historyList');
    const emptyEl = document.getElementById('historyEmpty');

    if (history.length === 0) {
        listEl.innerHTML = '';
        emptyEl.classList.remove('hidden');
        return;
    }

    emptyEl.classList.add('hidden');
    listEl.innerHTML = history.map((item, i) => {
        const time = new Date(item.timestamp).toLocaleString('vi-VN');
        const riskLabel = { Low: '🟢 Thấp', Medium: '🟡 Trung bình', High: '🔴 Cao' }[item.risk_level] || '--';
        const pct = Math.round(item.probability * 100);
        return `
            <div class="history-item ${item.risk_level}">
                <div class="history-info">
                    <div class="history-time">🕐 ${time}</div>
                    <div class="history-result">
                        ${item.prediction === 1 ? '⚠️ Có nguy cơ' : '✅ Ít nguy cơ'} — ${riskLabel}
                    </div>
                    <div class="history-prob">Xác suất: ${pct}%</div>
                </div>
            </div>
        `;
    }).join('');
}

// ── Loading & Toast Utilities ─────────────────────────────
function showLoading(text = 'Đang xử lý...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

function showToast(message, duration = 3000) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}
