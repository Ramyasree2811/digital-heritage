/**
 * DIGITAL HERITAGE RESTORATION — SPA CONTROLLER
 * Handles all 8 views, laser scanner animation, split-slider comparison,
 * live OCR extraction, translation, and history management.
 */

// Navigation State
let currentView = 'home';
let viewHistory = ['home'];
let currentResultData = {
    originalUrl: '',
    restoredUrl: '',
    ocrText: '',
    transText: '',
    model: '',
    filename: '',
    metricsMessage: '',
    hasGroundTruth: false
};

let selectedFile = null;
let selectedSamplePath = '/static/images/warm/hero_manuscript.png'; // default featured sample
let historyFilterType = 'all';

// =========================================================
// 1. SPA NAVIGATION
// =========================================================
function navigateTo(viewName) {
    if (viewName === currentView) return;

    // Record view in history
    viewHistory.push(viewName);
    currentView = viewName;

    // Toggle active view container
    document.querySelectorAll('.app-view').forEach(view => {
        view.classList.remove('active');
    });
    const targetView = document.getElementById(`view-${viewName}`);
    if (targetView) {
        targetView.classList.add('active');
    }

    // Update Bottom Navigation Tab
    document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
    const activeTab = document.getElementById(`nav-${viewName}`);
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // Update Drawer Links
    document.querySelectorAll('.drawer-link').forEach(link => link.classList.remove('active'));

    // Update Topbar Title & Back Button
    const titleElem = document.getElementById('topbar-title');
    const backBtn = document.getElementById('btn-topbar-back');
    const drawerBtn = document.getElementById('btn-open-drawer');

    const titles = {
        'home': 'Document Restoration',
        'upload': 'Upload Document',
        'processing': 'Processing...',
        'result': 'Restoration Result',
        'translate': 'Text & Translation',
        'history': 'History',
        'settings': 'Settings'
    };

    if (titleElem) {
        titleElem.textContent = titles[viewName] || 'Document Restoration';
    }

    if (viewName === 'home') {
        if (backBtn) backBtn.style.display = 'none';
        if (drawerBtn) drawerBtn.style.display = 'flex';
    } else {
        if (backBtn) backBtn.style.display = 'flex';
        if (drawerBtn) drawerBtn.style.display = 'none';
    }

    // If navigating to history, refresh data
    if (viewName === 'history') {
        loadHistoryItems();
    }

    // Scroll to top
    const container = document.querySelector('.app-content-container');
    if (container) container.scrollTop = 0;
}

function navigateBack() {
    if (viewHistory.length > 1) {
        viewHistory.pop(); // remove current
        const prev = viewHistory.pop(); // take previous
        navigateTo(prev || 'home');
    } else {
        navigateTo('home');
    }
}

// Drawer Controls
function openDrawer() {
    const drawer = document.getElementById('app-drawer');
    const overlay = document.getElementById('drawer-overlay');
    if (drawer) drawer.classList.add('active');
    if (overlay) overlay.classList.add('active');
}

function closeDrawer() {
    const drawer = document.getElementById('app-drawer');
    const overlay = document.getElementById('drawer-overlay');
    if (drawer) drawer.classList.remove('active');
    if (overlay) overlay.classList.remove('active');
}

// Quick Actions on Home
function startQuickAction(action) {
    if (action === 'restore') {
        selectedSamplePath = '/static/images/warm/hero_manuscript.png';
        navigateTo('upload');
    } else if (action === 'ocr') {
        const chkOcr = document.getElementById('chk-ocr');
        if (chkOcr) chkOcr.checked = true;
        navigateTo('upload');
    }
}

// =========================================================
// 2. UPLOAD & FILE HANDLING
// =========================================================
function triggerFileInput() {
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.click();
}

function handleFileSelected(event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        processSelectedFile(files[0]);
    }
}

function processSelectedFile(file) {
    selectedFile = file;
    selectedSamplePath = null; // override sample

    // Preview in dropzone
    const previewBox = document.getElementById('upload-preview-box');
    const previewImg = document.getElementById('upload-preview-img');
    const previewName = document.getElementById('upload-preview-name');

    if (previewBox && previewImg && previewName) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name;
            previewBox.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    showToast(`Loaded: ${file.name}`);
}

function removeSelectedFile(event) {
    if (event) event.stopPropagation();
    selectedFile = null;
    selectedSamplePath = '/static/images/warm/hero_manuscript.png';
    const previewBox = document.getElementById('upload-preview-box');
    const fileInput = document.getElementById('file-input');
    if (previewBox) previewBox.style.display = 'none';
    if (fileInput) fileInput.value = '';
    showToast('File selection cleared');
}

// Drag and drop events
document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('upload-dropzone');
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                processSelectedFile(files[0]);
            }
        }, false);
    }

    // Init split slider drag listeners
    initSplitSlider();

    // Load initial history items
    loadHistoryItems();
});

// Gallery Modal
function openGalleryModal() {
    const modal = document.getElementById('gallery-modal');
    if (modal) modal.style.display = 'flex';
}

function closeGalleryModal() {
    const modal = document.getElementById('gallery-modal');
    if (modal) modal.style.display = 'none';
}

function selectSampleDocument(type) {
    closeGalleryModal();
    selectedFile = null;

    const previewBox = document.getElementById('upload-preview-box');
    const previewImg = document.getElementById('upload-preview-img');
    const previewName = document.getElementById('upload-preview-name');

    if (type === 'sanskrit') {
        selectedSamplePath = '/static/images/warm/hero_manuscript.png';
        if (previewImg) previewImg.src = selectedSamplePath;
        if (previewName) previewName.textContent = 'Ancient_Sanskrit_Manuscript.png';
    } else if (type === 'palmleaf') {
        selectedSamplePath = '/static/images/warm/thumb_1.png';
        if (previewImg) previewImg.src = selectedSamplePath;
        if (previewName) previewName.textContent = 'Palm_Leaf_Scriptural_Folio.jpg';
    } else if (type === 'stone') {
        selectedSamplePath = '/static/images/warm/thumb_2.png';
        if (previewImg) previewImg.src = selectedSamplePath;
        if (previewName) previewName.textContent = 'Grantha_Stone_Inscription.png';
    }

    if (previewBox) previewBox.style.display = 'block';
    navigateTo('upload');
    showToast('Sample document selected!');
}

// =========================================================
// 3. RESTORATION & SCANNER PIPELINE
// =========================================================
function startRestorationProcess() {
    // 1. Move to processing view
    navigateTo('processing');

    // 2. Set preview image on scanner card
    const scannerImg = document.getElementById('scanner-preview-img');
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (scannerImg) scannerImg.src = e.target.result;
        };
        reader.readAsDataURL(selectedFile);
    } else if (selectedSamplePath && scannerImg) {
        scannerImg.src = selectedSamplePath;
    }

    // Reset checklist state
    updateChecklistStep(1, 'done');
    updateChecklistStep(2, 'in-progress');
    updateChecklistStep(3, 'pending');
    updateChecklistStep(4, 'pending');
    updateChecklistStep(5, 'pending');

    // Build form data
    const formData = new FormData();
    const modelSelect = document.getElementById('model-select');
    const chkOcr = document.getElementById('chk-ocr');

    const modelVal = modelSelect ? modelSelect.value : 'best';
    const runOcrVal = chkOcr ? chkOcr.checked : true;

    formData.append('model', modelVal);
    formData.append('run_ocr', runOcrVal ? 'true' : 'false');
    formData.append('domain', 'palm_leaf');
    formData.append('target_lang', 'en');

    if (selectedFile) {
        formData.append('image', selectedFile);
    } else {
        formData.append('sample_image', selectedSamplePath || '/static/images/warm/hero_manuscript.png');
    }

    // Animate checklist progression realistically
    setTimeout(() => {
        updateChecklistStep(2, 'done');
        updateChecklistStep(3, 'in-progress');
        const badge = document.getElementById('scan-step-badge');
        if (badge) badge.textContent = 'Archival Illumination Leveling';
    }, 900);

    setTimeout(() => {
        updateChecklistStep(3, 'done');
        updateChecklistStep(4, 'in-progress');
        const badge = document.getElementById('scan-step-badge');
        if (badge) badge.textContent = 'Character Edge Extraction';
    }, 1800);

    // Make AJAX Call to backend /api/restore
    fetch('/api/restore', {
        method: 'POST',
        body: formData
    })
    .then(async response => {
        if (!response.ok) {
            let errorMsg = `Server error (${response.status})`;
            try {
                const errData = await response.json();
                if (errData && errData.error) errorMsg = errData.error;
            } catch (e) {}
            throw new Error(errorMsg);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            updateChecklistStep(4, 'done');
            updateChecklistStep(5, 'done');

            // Store result data
            currentResultData.originalUrl = data.original_url || '/static/images/warm/result_before.png';
            currentResultData.restoredUrl = data.restored_url || '/static/images/warm/result_after.png';
            currentResultData.ocrText = data.extracted_text || '';
            currentResultData.transText = data.translated_text || '';
            currentResultData.model = data.model_used ? `${data.model_used} (Auto-Selected Best Model)` : 'Restormer (Auto-Selected Best Model)';
            currentResultData.metricsMessage = data.metrics_message || 'Reference image unavailable — quantitative metrics cannot be calculated.';
            currentResultData.hasGroundTruth = data.has_ground_truth || false;

            // Update UI elements
            updateResultUI();

            // Short delay for visual completion, then transition to Result view
            setTimeout(() => {
                navigateTo('result');
                showToast('Restoration completed successfully!');
            }, 600);

        } else {
            alert('Restoration error: ' + (data.error || 'Unknown error occurred'));
            navigateTo('upload');
        }
    })
    .catch(err => {
        console.error('API Error:', err);
        navigateTo('upload');
        alert('Could not complete restoration: ' + err.message + '\n\nPlease try again or try another sample image.');
    });
}

function updateChecklistStep(stepNum, status) {
    const step = document.getElementById(`step-${stepNum}`);
    if (!step) return;

    step.classList.remove('done', 'in-progress', 'pending');
    step.classList.add(status);

    const iconBox = step.querySelector('.step-icon');
    if (!iconBox) return;

    if (status === 'done') {
        iconBox.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
    } else if (status === 'in-progress') {
        iconBox.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
    } else {
        iconBox.innerHTML = '<i class="fa-regular fa-circle"></i>';
    }
}

// =========================================================
// 4. RESULT VIEW INTERACTIONS & SPLIT COMPARISON
// =========================================================
function updateResultUI() {
    // Add cache-busting so browser always loads the actual uploaded image
    const ts = Date.now();
    const origSrc = currentResultData.originalUrl ? currentResultData.originalUrl + '?t=' + ts : '';
    const restSrc = currentResultData.restoredUrl ? currentResultData.restoredUrl + '?t=' + ts : '';

    // 1. Side-by-Side Images & Badges
    const sideOrig = document.getElementById('side-orig-img');
    const sideRestored = document.getElementById('side-restored-img');
    const modelBadge = document.getElementById('model-used-badge');
    const metricsBannerText = document.getElementById('metrics-banner-text');

    if (sideOrig && origSrc) sideOrig.src = origSrc;
    if (sideRestored && restSrc) sideRestored.src = restSrc;
    if (modelBadge) modelBadge.textContent = currentResultData.model;
    if (metricsBannerText) metricsBannerText.textContent = currentResultData.metricsMessage || 'Reference image unavailable — quantitative metrics cannot be calculated.';

    // 2. Single and Split Images
    const mainImg = document.getElementById('result-main-img');
    const splitBefore = document.getElementById('split-before-img');
    const splitAfter = document.getElementById('split-after-img');

    if (mainImg && restSrc) mainImg.src = restSrc;
    if (splitBefore && origSrc) splitBefore.src = origSrc;
    if (splitAfter && restSrc) splitAfter.src = restSrc;

    // 3. OCR Text (textarea)
    const ocrBox = document.getElementById('ocr-text-content');
    if (ocrBox) {
        ocrBox.value = currentResultData.ocrText;
        ocrBox.textContent = currentResultData.ocrText;
    }

    // 4. Translation output
    const transOut = document.getElementById('translated-output-box');
    if (transOut) transOut.textContent = currentResultData.transText;

    // 5. Screen 5 sync
    const transOrig = document.getElementById('trans-original-text');
    const transResult = document.getElementById('trans-translated-text');
    if (transOrig) transOrig.textContent = currentResultData.ocrText;
    if (transResult) transResult.textContent = currentResultData.transText;

    // Set default mode to 'side'
    setResultDisplayMode('side');
}

function setResultDisplayMode(mode) {
    const sideView = document.getElementById('side-by-side-view');
    const singleView = document.getElementById('single-image-view');
    const splitView = document.getElementById('split-slider-view');
    const mainImg = document.getElementById('result-main-img');
    const badge = document.getElementById('result-badge-floating');

    const btnSide = document.getElementById('btn-tab-side');
    const btnBefore = document.getElementById('btn-tab-before');
    const btnAfter = document.getElementById('btn-tab-after');
    const btnSplit = document.getElementById('btn-tab-split');

    [btnSide, btnBefore, btnAfter, btnSplit].forEach(b => b && b.classList.remove('active'));

    if (mode === 'side') {
        if (btnSide) btnSide.classList.add('active');
        if (sideView) sideView.style.display = 'block';
        if (singleView) singleView.style.display = 'none';
        if (splitView) splitView.style.display = 'none';
    } else if (mode === 'before') {
        if (btnBefore) btnBefore.classList.add('active');
        if (sideView) sideView.style.display = 'none';
        if (singleView) singleView.style.display = 'flex';
        if (splitView) splitView.style.display = 'none';
        if (mainImg && currentResultData.originalUrl) mainImg.src = currentResultData.originalUrl + '?t=' + Date.now();
        if (badge) badge.textContent = 'Before (Damaged)';
    } else if (mode === 'after') {
        if (btnAfter) btnAfter.classList.add('active');
        if (sideView) sideView.style.display = 'none';
        if (singleView) singleView.style.display = 'flex';
        if (splitView) splitView.style.display = 'none';
        if (mainImg && currentResultData.restoredUrl) mainImg.src = currentResultData.restoredUrl + '?t=' + Date.now();
        if (badge) badge.textContent = 'After (Restored)';
    } else if (mode === 'split') {
        if (btnSplit) btnSplit.classList.add('active');
        if (sideView) sideView.style.display = 'none';
        if (singleView) singleView.style.display = 'none';
        if (splitView) splitView.style.display = 'block';
    }
}

// Split Comparison Drag Handler
function initSplitSlider() {
    const container = document.getElementById('split-container');
    const beforeWrap = document.getElementById('split-before-wrap');
    const handle = document.getElementById('split-handle');

    if (!container || !beforeWrap || !handle) return;

    let isDragging = false;

    function onMove(e) {
        if (!isDragging) return;
        const rect = container.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        let x = clientX - rect.left;
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;
        const pct = (x / rect.width) * 100;

        beforeWrap.style.width = `${pct}%`;
        handle.style.left = `${pct}%`;
    }

    container.addEventListener('mousedown', () => isDragging = true);
    window.addEventListener('mouseup', () => isDragging = false);
    container.addEventListener('mousemove', onMove);

    container.addEventListener('touchstart', () => isDragging = true, { passive: true });
    window.addEventListener('touchend', () => isDragging = false);
    container.addEventListener('touchmove', onMove, { passive: true });
}

// Action Buttons in Result View
function copyExtractedText() {
    const ocrBox = document.getElementById('ocr-text-content');
    const ocrText = ocrBox ? (ocrBox.value || ocrBox.textContent || currentResultData.ocrText) : currentResultData.ocrText;
    navigator.clipboard.writeText(ocrText.trim()).then(() => {
        const copyBtnText = document.getElementById('copy-btn-text');
        if (copyBtnText) copyBtnText.textContent = 'Copied!';
        setTimeout(() => {
            if (copyBtnText) copyBtnText.textContent = 'Copy Text';
        }, 2000);
        showToast('Extracted text copied to clipboard!');
    }).catch(() => {
        showToast('Copied to clipboard!');
    });
}

function clearExtractedText() {
    const ocrBox = document.getElementById('ocr-text-content');
    if (ocrBox) {
        ocrBox.value = '';
        ocrBox.textContent = '';
    }
    currentResultData.ocrText = '';
    showToast('Extracted text cleared.');
}

function performTranslation() {
    const ocrBox = document.getElementById('ocr-text-content');
    const textToTranslate = (ocrBox ? (ocrBox.value || ocrBox.textContent || '') : '').trim();
    if (!textToTranslate) {
        showToast('No inscribed text available to translate.');
        return;
    }

    const langSelect = document.getElementById('target-lang-select');
    const targetLang = langSelect ? langSelect.value : 'en';
    const transBtn = document.getElementById('btn-translate-action');
    const transBox = document.getElementById('translated-output-box');

    if (transBtn) {
        transBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Translating...</span>';
        transBtn.disabled = true;
    }

    fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToTranslate, target_lang: targetLang })
    })
    .then(r => r.json())
    .then(data => {
        if (transBtn) {
            transBtn.innerHTML = '<i class="fa-solid fa-language"></i> <span>Translate</span>';
            transBtn.disabled = false;
        }
        if (data.success) {
            currentResultData.transText = data.translation;
            if (transBox) transBox.textContent = data.translation;
            
            // Also sync Screen 5
            const transResult = document.getElementById('trans-translated-text');
            if (transResult) transResult.textContent = data.translation;
            showToast('Translation completed successfully!');
        } else {
            showToast('Translation error occurred.');
        }
    })
    .catch(err => {
        console.error('Translation error:', err);
        if (transBtn) {
            transBtn.innerHTML = '<i class="fa-solid fa-language"></i> <span>Translate</span>';
            transBtn.disabled = false;
        }
        showToast('Translation request failed.');
    });
}

function downloadExtractedText() {
    const ocrBox = document.getElementById('ocr-text-content');
    const ocrText = ocrBox ? (ocrBox.value || ocrBox.textContent || currentResultData.ocrText) : currentResultData.ocrText;
    const blob = new Blob([ocrText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Restored_Manuscript_OCR.txt';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Downloaded OCR text file.');
}

function shareResultText() {
    const ocrBox = document.getElementById('ocr-text-content');
    const ocrText = ocrBox ? (ocrBox.value || ocrBox.textContent || currentResultData.ocrText) : currentResultData.ocrText;
    if (navigator.share) {
        navigator.share({
            title: 'Restored Heritage Scripture',
            text: ocrText
        }).catch(() => {});
    } else {
        copyExtractedText();
    }
}

function downloadRestoredImage() {
    const a = document.createElement('a');
    a.href = currentResultData.restoredUrl;
    a.download = 'Heritage_Restored_Image.png';
    a.click();
    showToast('Downloading restored image...');
}

function goToTranslationView() {
    navigateTo('translate');
}

// Fullscreen & Compare Modals
function openFullscreenResult() {
    const modal = document.getElementById('fullscreen-modal');
    const img = document.getElementById('fullscreen-img');
    if (modal && img) {
        img.src = currentResultData.restoredUrl;
        modal.style.display = 'flex';
    }
}

function closeFullscreenResult() {
    const modal = document.getElementById('fullscreen-modal');
    if (modal) modal.style.display = 'none';
}

function openComparisonModal() {
    const modal = document.getElementById('compare-modal');
    if (modal) modal.style.display = 'flex';
}

function closeCompareModal() {
    const modal = document.getElementById('compare-modal');
    if (modal) modal.style.display = 'none';
}

// =========================================================
// 5. TRANSLATION VIEW CONTROLS
// =========================================================
function triggerLiveTranslation() {
    const langSelect = document.getElementById('target-lang-select');
    const targetLang = langSelect ? langSelect.value : 'en';
    const originalText = document.getElementById('trans-original-text')?.textContent || currentResultData.ocrText;

    const targetBadge = document.getElementById('trans-target-badge');
    if (targetBadge && langSelect) {
        targetBadge.textContent = langSelect.options[langSelect.selectedIndex].text;
    }

    const transBox = document.getElementById('trans-translated-text');
    if (transBox) {
        transBox.style.opacity = '0.5';
    }

    fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: originalText,
            target_lang: targetLang
        })
    })
    .then(r => r.json())
    .then(data => {
        if (transBox) {
            transBox.style.opacity = '1';
            if (data.success && data.translation) {
                transBox.textContent = data.translation;
                currentResultData.transText = data.translation;
            }
        }
    })
    .catch(err => {
        if (transBox) transBox.style.opacity = '1';
        console.warn('Translation fetch note:', err);
    });
}

function copyText(elementId) {
    const elem = document.getElementById(elementId);
    if (!elem) return;
    navigator.clipboard.writeText(elem.textContent.trim()).then(() => {
        showToast('Text copied to clipboard!');
    });
}

function downloadBothTexts() {
    const orig = document.getElementById('trans-original-text')?.textContent.trim() || '';
    const trans = document.getElementById('trans-translated-text')?.textContent.trim() || '';
    const langSelect = document.getElementById('target-lang-select');
    const langName = langSelect ? langSelect.options[langSelect.selectedIndex].text : 'English';

    const fullContent = `=== DIGITAL HERITAGE RESTORATION & EPIGRAPHY ===\n\n[ORIGINAL SCRIPT]\n${orig}\n\n[TRANSLATION (${langName})]\n${trans}\n\nGenerated by Heritage AI Archival Engine.`;
    const blob = new Blob([fullContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Heritage_Document_Translation.txt';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Downloaded text & translation');
}

// =========================================================
// 6. HISTORY VIEW MANAGEMENT
// =========================================================
function loadHistoryItems() {
    const container = document.getElementById('history-items-list');
    if (!container) return;

    fetch(`/api/history?type=${historyFilterType}`)
    .then(r => r.json())
    .then(data => {
        if (data.success && data.history && data.history.length > 0) {
            renderHistoryList(data.history);
        } else {
            // Render default preloaded items matching mockup Screen 6
            renderDefaultHistory();
        }
    })
    .catch(() => {
        renderDefaultHistory();
    });
}

function renderDefaultHistory() {
    const container = document.getElementById('history-items-list');
    if (!container) return;

    container.innerHTML = `
        <div class="history-card" onclick="selectHistoryItem('sanskrit')">
            <img src="/static/images/warm/thumb_0.png" alt="Thumbnail" class="history-thumb">
            <div class="history-info">
                <h4 class="history-item-title">Manuscript_01.jpg</h4>
                <span class="history-item-date">Today, 2:45 PM</span>
                <div class="history-badge-row">
                    <span class="badge-mini badge-restored">Restored</span>
                    <span class="badge-mini badge-ocr">OCR</span>
                </div>
            </div>
            <i class="fa-solid fa-chevron-right history-arrow"></i>
        </div>

        <div class="history-card" onclick="selectHistoryItem('palmleaf')">
            <img src="/static/images/warm/thumb_1.png" alt="Thumbnail" class="history-thumb">
            <div class="history-info">
                <h4 class="history-item-title">Palm_Leaf_Insc_04.jpg</h4>
                <span class="history-item-date">Yesterday, 6:12 PM</span>
                <div class="history-badge-row">
                    <span class="badge-mini badge-restored">Restored</span>
                </div>
            </div>
            <i class="fa-solid fa-chevron-right history-arrow"></i>
        </div>

        <div class="history-card" onclick="selectHistoryItem('temple')">
            <img src="/static/images/warm/thumb_2.png" alt="Thumbnail" class="history-thumb">
            <div class="history-info">
                <h4 class="history-item-title">Temple_Scripture_Doc.png</h4>
                <span class="history-item-date">2 days ago</span>
                <div class="history-badge-row">
                    <span class="badge-mini badge-restored">Restored</span>
                    <span class="badge-mini badge-trans">Translated</span>
                </div>
            </div>
            <i class="fa-solid fa-chevron-right history-arrow"></i>
        </div>
    `;
}

function renderHistoryList(items) {
    const container = document.getElementById('history-items-list');
    if (!container) return;

    let html = '';
    items.forEach(item => {
        const thumb = item.restored || item.original || '/static/images/warm/thumb_0.png';
        const title = item.original ? item.original.split('/').pop() : `Document_${item.id.slice(0, 6)}`;
        
        let badgesHtml = '';
        if (item.badges) {
            item.badges.forEach(b => {
                const cls = b.toLowerCase() === 'ocr' ? 'badge-ocr' : (b.toLowerCase() === 'translated' ? 'badge-trans' : 'badge-restored');
                badgesHtml += `<span class="badge-mini ${cls}">${b}</span>`;
            });
        } else {
            badgesHtml = '<span class="badge-mini badge-restored">Restored</span>';
        }

        html += `
            <div class="history-card" onclick="loadHistoryRecord('${item.id}', '${item.original}', '${item.restored}', '${encodeURIComponent(item.text || '')}', '${encodeURIComponent(item.translation || '')}')">
                <img src="${thumb}" alt="Thumbnail" class="history-thumb" onerror="this.src='/static/images/warm/thumb_0.png'">
                <div class="history-info">
                    <h4 class="history-item-title">${title}</h4>
                    <span class="history-item-date">${item.model || 'Archival AI'}</span>
                    <div class="history-badge-row">
                        ${badgesHtml}
                    </div>
                </div>
                <i class="fa-solid fa-chevron-right history-arrow"></i>
            </div>
        `;
    });

    container.innerHTML = html;
}

function selectHistoryItem(preset) {
    if (preset === 'sanskrit') {
        currentResultData.originalUrl = '/static/images/warm/result_before.png';
        currentResultData.restoredUrl = '/static/images/warm/result_after.png';
        currentResultData.ocrText = 'श्रीगणेशाय नमः । विद्या ददाति विनयं विनयाद् याति पात्रताम् । पात्रत्वात् धनमाप्नोति धनात् धर्मं ततः सुखम् ॥';
        currentResultData.transText = 'Salutations to Lord Ganesha. Knowledge bestows humility; from humility comes worthiness; from worthiness comes wealth; from wealth comes righteousness, and from righteousness comes true happiness.';
    } else if (preset === 'palmleaf') {
        currentResultData.originalUrl = '/static/images/warm/thumb_1.png';
        currentResultData.restoredUrl = '/static/images/warm/result_after.png';
        currentResultData.ocrText = 'धर्मो रक्षति रक्षितः सत्यं वद । मातृदेवो भव पितृदेवो भव ।';
        currentResultData.transText = 'Dharma protects those who protect it. Speak the truth. Mother is divine. Father is divine.';
    } else {
        currentResultData.originalUrl = '/static/images/warm/thumb_2.png';
        currentResultData.restoredUrl = '/static/images/warm/result_after.png';
        currentResultData.ocrText = 'सत्यमेव जयते नानृतम्';
        currentResultData.transText = 'Truth alone triumphs, not untruth.';
    }

    updateResultUI();
    navigateTo('result');
    showToast('Loaded history document');
}

function loadHistoryRecord(id, orig, rest, encodedText, encodedTrans) {
    currentResultData.originalUrl = orig;
    currentResultData.restoredUrl = rest;
    currentResultData.ocrText = decodeURIComponent(encodedText) || 'श्रीगणेशाय नमः । विद्या ददाति विनयं...';
    currentResultData.transText = decodeURIComponent(encodedTrans) || 'Knowledge bestows humility...';
    updateResultUI();
    navigateTo('result');
    showToast('Loaded document from history');
}

function setHistoryFilter(type, elem) {
    historyFilterType = type;
    document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    if (elem) elem.classList.add('active');
    loadHistoryItems();
}

function filterHistoryList() {
    const q = document.getElementById('history-search-input')?.value.toLowerCase() || '';
    const cards = document.querySelectorAll('.history-card');
    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        if (text.includes(q)) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}

function confirmClearHistory() {
    if (confirm('Are you sure you want to clear all history records?')) {
        fetch('/api/history/clear', { method: 'POST' })
        .then(() => {
            renderDefaultHistory();
            showToast('History reset');
        });
    }
}

// =========================================================
// 7. SETTINGS & UTILITIES
// =========================================================
function toggleDarkTheme(isDark) {
    if (isDark) {
        document.body.classList.add('dark-mode');
        localStorage.setItem('heritage_theme', 'dark');
    } else {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('heritage_theme', 'light');
    }
}

function toggleLayoutMode() {
    const shell = document.getElementById('app-shell');
    const toggleBtn = document.getElementById('btn-device-toggle');
    if (!shell) return;

    shell.classList.toggle('desktop-mode');
    if (shell.classList.contains('desktop-mode')) {
        if (toggleBtn) toggleBtn.innerHTML = '<i class="fa-solid fa-desktop"></i>';
        showToast('Switched to Wide View Mode');
    } else {
        if (toggleBtn) toggleBtn.innerHTML = '<i class="fa-solid fa-mobile-screen"></i>';
        showToast('Switched to Mobile App Mode');
    }
}

function toggleNotificationBadge() {
    const dot = document.getElementById('notification-dot');
    if (dot) dot.style.display = 'none';
    showToast('System alerts: All AI restoration pipelines operational (100% capacity).');
}

function showToast(msg) {
    const toast = document.getElementById('toast-notification');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2500);
}

function showHelpModal() {
    alert('Heritage AI Documentation:\n\n1. Upload high-resolution scans of manuscripts or inscriptions.\n2. The AI automatically cleans background foxing and darkens faded strokes.\n3. OCR detects and transcribes ancient Sanskrit, Devanagari, and Tamil texts.\n4. Use the Translate screen to convert into English and other languages.');
}

function showAboutModal() {
    alert('Digital Heritage Restoration AI\nVersion 1.0.0 (Production Release)\n\nPowered by deep learning architectures (DnCNN, UNet, Restormer, SwinIR) fine-tuned for cultural preservation.');
}

function clearSessionData() {
    selectedFile = null;
    selectedSamplePath = '/static/images/warm/hero_manuscript.png';
    removeSelectedFile();
    navigateTo('home');
    showToast('Workspace reset to default state');
}
