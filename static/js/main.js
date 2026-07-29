document.addEventListener('DOMContentLoaded', () => {

    // --- TAB NAVIGATION ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            navBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'dataset-tab') {
                loadDatasetStats();
                loadDatasetGallery(currentFilter);
            }
        });
    });

    // --- PREDICT SECTION (TAB 1) ---
    const imageInput = document.getElementById('imageInput');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const previewBox = document.getElementById('previewBox');
    const imagePreview = document.getElementById('imagePreview');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const predictBtn = document.getElementById('predictBtn');

    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');
    const resultsStatus = document.getElementById('resultsStatus');

    let selectedFile = null;

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetImageUpload();
    });

    function handleFileSelect(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneContent.classList.add('hidden');
            previewBox.classList.remove('hidden');
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetImageUpload() {
        selectedFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        dropzoneContent.classList.remove('hidden');
        previewBox.classList.add('hidden');
        predictBtn.disabled = true;

        emptyState.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        resultsStatus.textContent = 'Đang chờ ảnh đầu vào';
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        predictBtn.disabled = true;
        predictBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang Phân Tích...`;
        resultsStatus.textContent = 'Đang xử lý mô hình...';

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                renderResults(data);
            } else {
                alert(data.error || 'Lỗi xử lý hình ảnh');
            }
        } catch (err) {
            console.error(err);
            alert('Lỗi kết nối máy chủ API');
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Phân Loại Ngay`;
        }
    });

    function renderResults(data) {
        emptyState.classList.add('hidden');
        resultsContent.classList.remove('hidden');
        resultsStatus.textContent = 'Hoàn tất phân tích';

        // Top prediction title & confidence
        document.getElementById('predTitle').textContent = data.predicted_class;
        document.getElementById('predConf').textContent = data.confidence;

        // Probabilities Progress Bars
        const probs = data.probabilities;
        
        updateBar('hong_trung_quoc', probs['Hồng Trung Quốc'] || 0);
        updateBar('hong_lang_son', probs['Hồng Lạng Sơn'] || 0);
        updateBar('hong_da_lat', probs['Hồng Đà Lạt'] || 0);

        // Visual analysis
        const ana = data.visual_analysis;
        document.getElementById('anaRatio').textContent = ana.aspect_ratio;
        document.getElementById('anaShape').textContent = ana.shape_type;
        document.getElementById('anaColor').textContent = ana.color_intensity;
        document.getElementById('anaEngine').textContent = `${data.model_architecture} (${data.is_custom_trained ? 'Custom Fine-tuned' : 'Pretrained Feature'})`;
    }

    function updateBar(key, percentage) {
        const valElem = document.getElementById(`val-${key}`);
        const barElem = document.getElementById(`bar-${key}`);
        if (valElem && barElem) {
            valElem.textContent = `${percentage}%`;
            barElem.style.width = `${percentage}%`;
        }
    }

    // --- DATASET STUDIO (TAB 2) ---
    let currentFilter = 'all';

    async function loadDatasetStats() {
        try {
            const res = await fetch('/api/dataset_stats');
            const data = await res.json();
            if (res.ok) {
                const cls = data.classes;
                document.getElementById('statClassTQ').textContent = cls['hong_trung_quoc'].total_files;
                document.getElementById('statClassLS').textContent = cls['hong_lang_son'].total_files;
                document.getElementById('statClassDL').textContent = cls['hong_da_lat'].total_files;
                document.getElementById('statTotalAll').textContent = data.total_images + data.total_videos;

                document.getElementById('modelArchName').textContent = data.current_architecture;
            }
        } catch (err) {
            console.error('Error fetching dataset stats:', err);
        }
    }

    async function loadDatasetGallery(classFilter = 'all') {
        const galleryGrid = document.getElementById('datasetGalleryGrid');
        if (!galleryGrid) return;

        galleryGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-muted);"><i class="fa-solid fa-spinner fa-spin"></i> Đang tải danh sách tệp...</div>`;

        try {
            const res = await fetch(`/api/dataset_items?class_name=${classFilter}`);
            const data = await res.json();

            if (res.ok && data.items && data.items.length > 0) {
                galleryGrid.innerHTML = '';
                data.items.forEach(item => {
                    const card = createGalleryItemCard(item);
                    galleryGrid.appendChild(card);
                });
            } else {
                galleryGrid.innerHTML = `
                    <div style="grid-column: 1/-1; text-align: center; padding: 3rem 1rem; color: var(--text-muted);">
                        <i class="fa-solid fa-folder-open" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;"></i>
                        <p>Chưa có tệp dữ liệu nào trong thư mục này.</p>
                    </div>
                `;
            }
        } catch (err) {
            console.error('Error loading gallery:', err);
            galleryGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: #ff4b2b;">Lỗi khi tải dữ liệu thư viện.</div>`;
        }
    }

    function createGalleryItemCard(item) {
        const card = document.createElement('div');
        card.className = 'gallery-item-card';

        const classOptions = [
            { val: 'hong_trung_quoc', label: 'Hồng Trung Quốc' },
            { val: 'hong_lang_son', label: 'Hồng Lạng Sơn' },
            { val: 'hong_da_lat', label: 'Hồng Đà Lạt' }
        ];

        let selectOptionsHtml = classOptions.map(opt => 
            `<option value="${opt.val}" ${opt.val === item.class_name ? 'selected' : ''}>Chuyển sang: ${opt.label}</option>`
        ).join('');

        card.innerHTML = `
            <div class="gallery-img-container">
                <span class="gallery-class-tag">${item.class_display}</span>
                <span class="gallery-size-tag">${item.size_kb} KB</span>
                ${item.is_video 
                    ? `<video src="${item.file_url}" style="width:100%; height:100%; object-fit:cover;"></video>` 
                    : `<img src="${item.file_url}" alt="${item.filename}" loading="lazy">`}
            </div>
            <div class="gallery-item-body">
                <div class="gallery-filename" title="${item.filename}">${item.filename}</div>
                
                <!-- UPDATE: Move/Reclassify -->
                <select class="move-select" data-filename="${item.filename}" data-src="${item.class_name}">
                    ${selectOptionsHtml}
                </select>

                <!-- Actions: Rename & Delete -->
                <div class="gallery-item-actions">
                    <button class="btn-item-action btn-rename" data-filename="${item.filename}" data-class="${item.class_name}">
                        <i class="fa-solid fa-pen-to-square"></i> Đổi tên
                    </button>
                    <button class="btn-item-action btn-danger-action btn-delete" data-filename="${item.filename}" data-class="${item.class_name}">
                        <i class="fa-solid fa-trash-can"></i> Xóa
                    </button>
                </div>
            </div>
        `;

        // Attach Reclassify/Move event
        const moveSelect = card.querySelector('.move-select');
        moveSelect.addEventListener('change', async (e) => {
            const newClass = e.target.value;
            const srcClass = moveSelect.getAttribute('data-src');
            const filename = moveSelect.getAttribute('data-filename');

            if (newClass === srcClass) return;

            try {
                const res = await fetch('/api/move_dataset_item', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source_class: srcClass, target_class: newClass, filename: filename })
                });

                const resData = await res.json();
                if (res.ok) {
                    loadDatasetStats();
                    loadDatasetGallery(currentFilter);
                } else {
                    alert(resData.error || 'Lỗi khi chuyển tệp');
                }
            } catch (err) {
                alert('Lỗi kết nối máy chủ');
            }
        });

        // Attach Rename event
        const renameBtn = card.querySelector('.btn-rename');
        renameBtn.addEventListener('click', async () => {
            const oldName = renameBtn.getAttribute('data-filename');
            const clsName = renameBtn.getAttribute('data-class');
            const newName = prompt(`Đổi tên tệp "${oldName}" thành:`, oldName);

            if (!newName || newName === oldName) return;

            try {
                const res = await fetch('/api/rename_dataset_item', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ class_name: clsName, old_filename: oldName, new_filename: newName })
                });

                const resData = await res.json();
                if (res.ok) {
                    loadDatasetGallery(currentFilter);
                } else {
                    alert(resData.error || 'Lỗi đổi tên');
                }
            } catch (err) {
                alert('Lỗi kết nối máy chủ');
            }
        });

        // Attach Delete event
        const deleteBtn = card.querySelector('.btn-delete');
        deleteBtn.addEventListener('click', async () => {
            const filename = deleteBtn.getAttribute('data-filename');
            const clsName = deleteBtn.getAttribute('data-class');

            if (!confirm(`Bạn có chắc chắn muốn xóa tệp "${filename}" khỏi bộ dữ liệu?`)) return;

            try {
                const res = await fetch('/api/delete_dataset_item', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ class_name: clsName, filename: filename })
                });

                const resData = await res.json();
                if (res.ok) {
                    loadDatasetStats();
                    loadDatasetGallery(currentFilter);
                } else {
                    alert(resData.error || 'Lỗi xóa tệp');
                }
            } catch (err) {
                alert('Lỗi kết nối máy chủ');
            }
        });

        return card;
    }

    // Filter Buttons logic
    const filterBtns = document.querySelectorAll('.gallery-filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            currentFilter = btn.getAttribute('data-filter');
            loadDatasetGallery(currentFilter);
        });
    });

    const datasetUploadForm = document.getElementById('datasetUploadForm');
    const uploadStatusMsg = document.getElementById('uploadStatusMsg');

    datasetUploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('datasetFile');
        const uploadBtn = document.getElementById('uploadDatasetBtn');

        if (!fileInput.files[0]) return;

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang Tải Lên & Xử Lý...`;
        uploadStatusMsg.classList.add('hidden');

        const formData = new FormData(datasetUploadForm);

        try {
            const res = await fetch('/api/upload_dataset', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                uploadStatusMsg.textContent = data.message + (data.extracted_frames ? ` (Đã trích xuất ${data.extracted_frames} khung hình từ video)` : '');
                uploadStatusMsg.classList.remove('hidden');
                datasetUploadForm.reset();
                loadDatasetStats();
                loadDatasetGallery(currentFilter);
            } else {
                alert(data.error || 'Lỗi tải lên');
            }
        } catch (err) {
            alert('Lỗi kết nối máy chủ');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `<i class="fa-solid fa-upload"></i> Tải Lên & Xử Lý`;
        }
    });


    // --- TRAINER STUDIO (TAB 3) ---
    const trainForm = document.getElementById('trainForm');
    const startTrainBtn = document.getElementById('startTrainBtn');
    const logConsole = document.getElementById('logConsole');
    const trainStatusBadge = document.getElementById('trainStatusBadge');

    trainForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const epochs = document.getElementById('epochInput').value;
        const lr = document.getElementById('lrInput').value;

        startTrainBtn.disabled = true;
        startTrainBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Đang Huấn Luyện...`;
        trainStatusBadge.textContent = 'Đang huấn luyện';
        trainStatusBadge.className = 'badge';

        appendConsoleLog(`[Trainer] Khởi chạy quá trình Fine-Tuning với Epochs=${epochs}, LR=${lr}...`);

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs: parseInt(epochs), learning_rate: parseFloat(lr) })
            });

            const data = await res.json();
            if (res.ok) {
                appendConsoleLog(`[Success] ${data.message}`);
                data.history.forEach(h => {
                    const valInfo = h.val_loss !== undefined ? ` | Val Loss: ${h.val_loss} | Val Acc: ${h.val_accuracy}%` : '';
                    appendConsoleLog(`--> Epoch ${h.epoch}/${epochs} | Train Loss: ${h.loss} | Train Acc: ${h.accuracy}%${valInfo}`);
                });
                appendConsoleLog(`[ModelEngine] Đã kiểm định chống Overfitting & lưu mô hình tối ưu (Best Model Checkpoint Active).`);
                trainStatusBadge.textContent = 'Hoàn tất';
                trainStatusBadge.className = 'badge badge-success';
                loadDatasetStats();
            } else {
                appendConsoleLog(`[Error] ${data.error}`);
                alert(data.error);
                trainStatusBadge.textContent = 'Thất bại';
            }
        } catch (err) {
            appendConsoleLog(`[Error] Lỗi kết nối API huấn luyện.`);
        } finally {
            startTrainBtn.disabled = false;
            startTrainBtn.innerHTML = `<i class="fa-solid fa-play"></i> Bắt Đầu Huấn Luyện (Start Training)`;
        }
    });

    function appendConsoleLog(msg) {
        const p = document.createElement('p');
        p.className = 'console-line';
        p.textContent = msg;
        logConsole.appendChild(p);
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    // --- ARCHITECTURE COMPARISON (TAB 4) ---
    const selectArchBtns = document.querySelectorAll('.select-arch-btn');

    selectArchBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            const arch = btn.getAttribute('data-arch');

            try {
                const res = await fetch('/api/switch_architecture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ architecture: arch })
                });

                const data = await res.json();
                if (res.ok) {
                    document.querySelectorAll('.arch-card').forEach(c => c.classList.remove('active-arch'));
                    document.getElementById(`archCard-${arch}`).classList.add('active-arch');

                    document.getElementById('modelArchName').textContent = data.current_architecture;
                    document.getElementById('trainingArchDisplay').textContent = data.current_architecture;

                    alert(`Đã chuyển mô hình sang: ${data.current_architecture}`);
                }
            } catch (err) {
                alert('Lỗi chuyển đổi kiến trúc mô hình');
            }
        });
    });

    // Initial Load
    loadDatasetStats();
});
