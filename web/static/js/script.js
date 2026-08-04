/* ============================================================
   DermaSense AI — Script (Upload + Webcam + Render Results)
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {

    // ── Elements ──
    const uploadZone     = document.getElementById('uploadZone');
    const imageInput     = document.getElementById('imageInput');
    const uploadContent  = document.getElementById('uploadContent');
    const imagePreview   = document.getElementById('imagePreview');
    const removeBtn      = document.getElementById('removeImageBtn');
    const ageSlider      = document.getElementById('age');
    const ageValue       = document.getElementById('ageValue');
    const form           = document.getElementById('diagnosisForm');
    const analyzeBtn     = document.getElementById('analyzeBtn');
    const image_b64Field = document.getElementById('image_b64');

    const emptyState    = document.getElementById('emptyState');
    const loadingState  = document.getElementById('loadingState');
    const resultContent = document.getElementById('resultContent');

    let selectedFile   = null;
    let currentMode    = 'upload';
    let webcamStream   = null;
    // Lưu b64 của ảnh webcam để hiển thị ảnh gốc sau khi submit
    let lastWebcamB64  = '';

    // ── Age Slider ──
    ageSlider.addEventListener('input', e => { ageValue.textContent = e.target.value; });

    // ================================================================
    //  MODE SWITCHING
    // ================================================================
    window.switchMode = function(mode) {
        currentMode = mode;
        document.getElementById('uploadMode').classList.toggle('hidden', mode !== 'upload');
        document.getElementById('cameraMode').classList.toggle('hidden', mode !== 'camera');
        document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
        document.getElementById('tabCamera').classList.toggle('active', mode === 'camera');

        if (mode === 'upload') {
            stopCamera();
            analyzeBtn.disabled = !selectedFile;
        } else {
            // camera mode: chỉ bật nút sau khi camera bật
            analyzeBtn.disabled = true;
            image_b64Field.value = '';
        }
    };

    // ================================================================
    //  UPLOAD LOGIC
    // ================================================================
    uploadZone.addEventListener('click', () => { if (!selectedFile) imageInput.click(); });

    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.style.background = 'rgba(99,102,241,0.12)';
    });
    uploadZone.addEventListener('dragleave', () => { uploadZone.style.background = ''; });
    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.style.background = '';
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    imageInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

    removeBtn.addEventListener('click', e => {
        e.stopPropagation();
        selectedFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        removeBtn.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        uploadZone.style.borderStyle = 'dashed';
        analyzeBtn.disabled = true;
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) { alert('Vui long chon file hinh anh!'); return; }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            removeBtn.classList.remove('hidden');
            uploadContent.classList.add('hidden');
            uploadZone.style.borderStyle = 'solid';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ================================================================
    //  WEBCAM LOGIC
    // ================================================================
    window.startCamera = async function() {
        try {
            // Thử dùng camera sau trước, nếu lỗi dùng camera trước
            let constraints = { video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (_) {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            }

            const video = document.getElementById('webcamVideo');
            video.srcObject = webcamStream;

            document.getElementById('webcamOverlay').classList.add('hidden');
            document.getElementById('startCameraBtn').classList.add('hidden');
            document.getElementById('stopCameraBtn').classList.remove('hidden');
            document.getElementById('captureBtn').classList.remove('hidden');
        } catch (err) {
            alert('Khong the truy cap Camera: ' + err.message + '. Vui long cap quyen trong trinh duyet.');
        }
    };

    window.stopCamera = function() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }
        const video = document.getElementById('webcamVideo');
        if (video) { video.srcObject = null; }
        ['webcamOverlay', 'startCameraBtn'].forEach(id => document.getElementById(id)?.classList.remove('hidden'));
        ['stopCameraBtn', 'captureBtn'].forEach(id => document.getElementById(id)?.classList.add('hidden'));
        if (currentMode === 'camera') analyzeBtn.disabled = true;
    };

    window.captureAndAnalyze = async function() {
        const video = document.getElementById('webcamVideo');
        if (!video || !video.srcObject || !video.videoWidth) {
            alert('Camera chua san sang! Hay batt camera truoc.');
            return;
        }

        // Chụp frame từ webcam vào canvas
        const canvas = document.getElementById('webcamCanvas');
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const b64 = canvas.toDataURL('image/jpeg', 0.92);

        // Lưu lại để hiển thị ảnh gốc sau
        lastWebcamB64 = b64;

        // Đặt giá trị vào hidden field
        image_b64Field.value = b64;

        // Gọi submit (không qua nút để tránh bị disabled)
        await doSubmit();
    };

    // ================================================================
    //  FORM SUBMIT
    // ================================================================
    form.addEventListener('submit', async e => {
        e.preventDefault();
        if (currentMode === 'camera') {
            // Nếu bấm nút Phân Tích trong chế độ camera → chụp và phân tích
            await window.captureAndAnalyze();
        } else {
            await doSubmit();
        }
    });

    async function doSubmit() {
        // Kiểm tra có dữ liệu không
        if (currentMode === 'upload' && !selectedFile) {
            alert('Vui long chon anh truoc khi phan tich!');
            return;
        }
        if (currentMode === 'camera' && !image_b64Field.value) {
            alert('Vui long chup anh truoc khi phan tich!');
            return;
        }

        showState('loading');
        analyzeBtn.disabled = true;

        // Xây dựng FormData thủ công để kiểm soát chính xác
        const formData = new FormData();

        // Clinical data
        formData.append('age',            document.getElementById('age').value);
        formData.append('sex',            document.querySelector('input[name="sex"]:checked')?.value || 'male');
        formData.append('anatom_site',    document.getElementById('anatom_site').value);
        formData.append('duration',       document.getElementById('duration').value);
        formData.append('symptoms',       document.getElementById('symptoms').value);
        formData.append('skin_type',      document.querySelector('input[name="skin_type"]:checked')?.value || 'III');
        formData.append('family_history', document.querySelector('input[name="family_history"]:checked')?.value || 'no');

        // Ảnh
        if (currentMode === 'upload') {
            formData.append('image', selectedFile);
        } else {
            // Camera mode: gửi base64
            formData.append('image_b64', image_b64Field.value);
        }

        try {
            const response = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            renderResults(data);
        } catch (err) {
            console.error('Fetch error:', err);
            alert('Loi phan tich: ' + err.message);
            showState('empty');
        } finally {
            analyzeBtn.disabled = false;
            // Không xóa image_b64Field ngay để renderResults có thể dùng
        }
    }

    // ================================================================
    //  RENDER RESULTS
    // ================================================================
    function renderResults(data) {
        // 1. Diagnosis
        document.getElementById('primaryDiagnosis').textContent = data.primary_diagnosis;

        // 2. Risk Badge
        const badge = document.getElementById('riskBadge');
        badge.className = 'risk-badge';
        const risk = data.risk_level;
        const icon  = data.risk_icon || (risk === 'High' ? '🔴' : risk === 'Medium' ? '🟡' : '🟢');
        const lvVn  = data.risk_level_vn || (risk === 'High' ? 'Rui ro Cao' : risk === 'Medium' ? 'Rui ro Trung binh' : 'Rui ro Thap');
        const msg   = risk === 'High' ? '— Can kham bac si ngay!' : risk === 'Medium' ? '— Can theo doi / dieu tri' : '— Co the tu cham soc';
        badge.classList.add(risk === 'High' ? 'risk-high' : risk === 'Medium' ? 'risk-medium' : 'risk-low');
        badge.innerHTML = `${icon} <strong>${lvVn}</strong> ${msg}`;

        // 3. Confidence Circle
        const pct = Math.round(data.confidence * 100);
        document.getElementById('confidenceText').textContent = `${pct}%`;
        const circle = document.getElementById('confidenceCircle');
        circle.style.stroke = pct > 79 ? 'var(--risk-low)' : pct > 49 ? 'var(--risk-med)' : 'var(--risk-high)';
        setTimeout(() => { circle.style.strokeDasharray = `${pct}, 100`; }, 80);

        // 4. Ảnh gốc + Ảnh tiền xử lý + Heatmap
        const origEl = document.getElementById('origImgResult');
        if (currentMode === 'upload' && selectedFile) {
            origEl.src = URL.createObjectURL(selectedFile);
        } else {
            // Camera: dùng bản lưu trước khi submit
            origEl.src = lastWebcamB64;
        }
        document.getElementById('processedImgResult').src = `data:image/jpeg;base64,${data.processed_base64}`;
        document.getElementById('heatmapResult').src = `data:image/png;base64,${data.gradcam_base64}`;

        // 5. Lời khuyên
        document.getElementById('adviceText').textContent    = data.advice     || 'Theo doi vung da va tham khao bac si da lieu.';
        document.getElementById('seeDoctorText').textContent = data.see_doctor || 'Den kham bac si de co huong dieu tri chinh xac.';

        // 6. Top 5 Bar Chart
        const chart = document.getElementById('top5Chart');
        chart.innerHTML = '';
        (data.top5 || []).forEach((item, idx) => {
            const p = Math.round(item.prob * 100);
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-label" title="${item.class}">${item.class}</div>
                <div class="bar-track"><div class="bar-fill" id="bf${idx}"></div></div>
                <div class="bar-value">${p}%</div>`;
            chart.appendChild(row);
            setTimeout(() => {
                const fill = document.getElementById(`bf${idx}`);
                if (fill) {
                    fill.style.width = `${p}%`;
                    fill.style.background = idx === 0 ? 'linear-gradient(90deg,#6366F1,#8B5CF6)' : '#CBD5E1';
                }
            }, 60 + idx * 40);
        });

        // Dọn dẹp sau khi render xong
        image_b64Field.value = '';
        showState('result');
    }

    // ================================================================
    //  UTILITY
    // ================================================================
    function showState(state) {
        emptyState.classList.toggle('hidden',    state !== 'empty');
        loadingState.classList.toggle('hidden',  state !== 'loading');
        resultContent.classList.toggle('hidden', state !== 'result');
    }
});
