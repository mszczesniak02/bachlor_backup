const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');
const loading = document.getElementById('loading');
const error = document.getElementById('error');

const result = document.getElementById('result');
const resultContent = document.getElementById('resultContent');

// Otwórz dialog przy kliknięciu
uploadArea.addEventListener('click', () => fileInput.click());

// Przeciąganie pliku
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    fileInput.files = e.dataTransfer.files;
    uploadForm.dispatchEvent(new Event('submit'));
});

// Zmiana pliku - pokaż preview wewnątrz upload-area
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            showPreviewInUploadArea(event.target.result, file.name, img.width, img.height);
            document.getElementById('analyzeBtn').style.display = 'block';
        };
        img.onerror = () => {
            showError('Nie można załadować pliku');
        };
        img.src = event.target.result;
    };
    reader.onerror = () => {
        showError('Błąd czytania pliku');
    };
    reader.readAsDataURL(file);
});

// Wysyłanie formularza
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
        showError('Proszę wybrać plik');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showLoading(true);
    hideMessages();

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResult(data);
        } else {
            showError(data.error || 'Błąd podczas analizy');
        }
    } catch (err) {
        showError('Błąd połączenia: ' + err.message);
    } finally {
        showLoading(false);
    }
});

function displayResult(data) {
    const info = data.info;
    const analysis = data.analysis;
    const images = data.images;
    
    let html = '';

    // --- Images Section ---
    if (images && (images.overlay || images.mask)) {
        html += '<div class="images-container" style="display: flex; flex-direction: column; gap: 20px; align-items: center; padding-bottom: 20px;">';
        
        if (images.overlay) {
            html += `<div class="thumbnail-section" style="width: 100%; max-width: 800px;">
                <div class="preview-header">Overlay</div>
                <img src="${images.overlay}" alt="Overlay" style="width: 100%; height: auto; border-radius: 8px;"/>
            </div>`;
        }
        // Heatmap removed
        if (images.mask) {
            html += `<div class="thumbnail-section" style="width: 100%; max-width: 800px;">
                <div class="preview-header">Mask</div>
                <img src="${images.mask}" alt="Mask" style="width: 100%; height: auto; border-radius: 8px;"/>
            </div>`;
        }
        html += '</div>';
    } else if (info.thumbnail) {
         html += `<div class="thumbnail-section">
            <div class="preview-header">Podgląd</div>
            <img src="${info.thumbnail}" alt="Podgląd" />
        </div>`;
    }

    // --- Basic Info REMOVED ---

    // --- Domain Controller ---
    if (analysis && analysis.domain_controller) {
        const dc = analysis.domain_controller;
        const color = dc.is_crack ? '#2c6e2c' : '#c33';
        html += `<div class="result-group" style="margin-top: 20px;">
            <h3 style="color: #667eea; margin-bottom: 10px;">Detekcja</h3>
            <div class="result-item" style="border-left-color: ${color};">
                <div class="result-label">Status</div>
                <div class="result-value" style="color: ${color}; font-weight: bold;">
                    ${dc.label} (${(dc.confidence * 100).toFixed(1)}%)
                </div>
            </div>
        </div>`;
    }

    // --- Classification ---
    if (analysis && analysis.classification) {
        const cl = analysis.classification;
        html += `<div class="result-group" style="margin-top: 20px;">
            <h3 style="color: #667eea; margin-bottom: 10px;">Klasyfikacja</h3>
            <div class="result-item">
                <div class="result-label">Kategoria</div>
                <div class="result-value">${cl.class_name}</div>
            </div>
             <div class="result-item">
                <div class="result-label">Pewność</div>
                <div class="result-value">${(cl.confidence * 100).toFixed(1)}%</div>
            </div>
        </div>`;
    }

    // --- Geometric Analysis ---
    if (analysis && analysis.geometric) {
        const geo = analysis.geometric;
        const basic = geo.basic || {};
        const width = geo.width_stats || {};
        
        const descStyle = 'font-size: 0.85em; color: #a0aec0; margin-top: 5px; line-height: 1.2;';

        html += `<div class="result-group" style="margin-top: 20px;">
            <h3 style="color: #667eea; margin-bottom: 10px;">Geometria</h3>
            
            <div class="result-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="result-item">
                    <div class="result-label">Długość</div>
                    <div class="result-value">${geo.length_pixels ? geo.length_pixels.toFixed(1) : 0} px</div>
                    <div style="${descStyle}">Suma pikseli wzdłuż szkieletu pęknięcia.</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Szerokość (Średnia)</div>
                    <div class="result-value">${width.mean_width ? width.mean_width.toFixed(2) : 0} px</div>
                    <div style="${descStyle}">Średnia odległość krawędzi od środka pęknięcia.</div>
                </div>
                 <div class="result-item">
                    <div class="result-label">Szerokość (Max)</div>
                    <div class="result-value">${width.max_width ? width.max_width.toFixed(2) : 0} px</div>
                    <div style="${descStyle}">Największa lokalna szerokość pęknięcia.</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Powierzchnia</div>
                    <div class="result-value">${basic.area_pixels ? basic.area_pixels.toFixed(0) : 0} px²</div>
                    <div style="${descStyle}">Całkowita liczba pikseli zaklasyfikowanych jako pęknięcie.</div>
                </div>
                 <div class="result-item">
                    <div class="result-label">Orientacja</div>
                    <div class="result-value">${basic.orientation ? (basic.orientation * 180 / Math.PI).toFixed(1) : 0}°</div>
                    <div style="${descStyle}">Kąt nachylenia głównej osi pęknięcia (-90° do 90°).</div>
                </div>
            </div>
        </div>`;

        // --- Advanced Analysis ---
        if (analysis.geometric.advanced) {
            const adv = analysis.geometric.advanced;
            html += `<div class="result-group" style="margin-top: 20px;">
                <h3 style="color: #667eea; margin-bottom: 10px;">Zaawansowana Analiza</h3>
                <div class="result-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                     <div class="result-item">
                        <div class="result-label">Krętość (Tortuosity)</div>
                        <div class="result-value">${adv.tortuosity ? adv.tortuosity.toFixed(3) : 1.0}</div>
                        <div style="${descStyle}">Stosunek długości rzeczywistej do odległości w linii prostej (1.0 = prosta).</div>
                    </div>
                     <div class="result-item">
                        <div class="result-label">Punkty Rozgałęzień</div>
                        <div class="result-value">${adv.branch_points_count !== undefined ? adv.branch_points_count : 0}</div>
                        <div style="${descStyle}">Liczba miejsc, gdzie pęknięcie się rozdwaja.</div>
                    </div>
                     <div class="result-item">
                        <div class="result-label">Punkty Końcowe</div>
                        <div class="result-value">${adv.endpoints_count !== undefined ? adv.endpoints_count : 0}</div>
                        <div style="${descStyle}">Liczba "ślepych zaułków" (końców) pęknięcia.</div>
                    </div>
                     <div class="result-item">
                        <div class="result-label">Gęstość Pęknięć</div>
                        <div class="result-value">${adv.crack_density ? adv.crack_density.toFixed(4) : 0}</div>
                        <div style="${descStyle}">Stosunek długości pęknięcia do powierzchni całego obrazu.</div>
                    </div>
                     <div class="result-item">
                        <div class="result-label">Intensywność Rozgałęzień</div>
                        <div class="result-value">${adv.branching_intensity ? adv.branching_intensity.toFixed(4) : 0}</div>
                        <div style="${descStyle}">Liczba punktów rozgałęzień przypadająca na piksel długości.</div>
                    </div>
                </div>
            </div>`;
        }
    }

    resultContent.innerHTML = html;
    result.classList.add('show');
}

function showError(msg) {
    error.textContent = msg;
    error.classList.add('show');
}



function showLoading(show) {
    loading.classList.toggle('show', show);
}

function hideMessages() {
    error.classList.remove('show');
    result.classList.remove('show');
}

function showPreviewInUploadArea(imageSrc, filename, width, height) {
    hideMessages();
    
    uploadArea.innerHTML = `
        <div class="preview-in-upload">
            <img src="${imageSrc}" alt="Podgląd ${filename}" class="preview-thumbnail"/>
            <div class="preview-details">
                <div class="preview-filename">${filename}</div>
                <div class="preview-dimensions">${width} × ${height} px</div>
            </div>
            <button type="button" class="change-file-btn" onclick="document.getElementById('fileInput').click()">
                Zmień plik
            </button>
        </div>
    `;
}
