const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadForm = document.getElementById('uploadForm');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const success = document.getElementById('success');
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
            displayResult(data.info);
            showSuccess('Analiza ukończona pomyślnie!');
        } else {
            showError(data.error || 'Błąd podczas analizy');
        }
    } catch (err) {
        showError('Błąd połączenia: ' + err.message);
    } finally {
        showLoading(false);
    }
});

function displayResult(info) {
    let html = '';

    // Miniaturka z serwera
    if (info.thumbnail) {
        html += `<div class="thumbnail-section">
            <div class="preview-header">📊 Wyniki Analizy</div>
            <img src="${info.thumbnail}" alt="Analiza ${info.filename}" />
        </div>`;
    }

    html += `<div class="result-item">
        <div class="result-label">Nazwa pliku</div>
        <div class="result-value">${info.filename}</div>
    </div>`;

    if (info.size) {
        html += `<div class="result-item">
            <div class="result-label">Wymiary</div>
            <div class="result-value">${info.size.width} × ${info.size.height} px</div>
        </div>`;

        html += `<div class="result-item">
            <div class="result-label">Liczba pikseli</div>
            <div class="result-value">${info.size.total_pixels.toLocaleString()}</div>
        </div>`;
    }

    if (info.format) {
        html += `<div class="result-item">
            <div class="result-label">Format</div>
            <div class="result-value">${info.format}</div>
        </div>`;
    }

    if (info.color_mode) {
        html += `<div class="result-item">
            <div class="result-label">Tryb kolorów</div>
            <div class="result-value">${info.color_mode}</div>
        </div>`;
    }

    if (info.brightness) {
        html += `<div class="result-item">
            <div class="result-label">Jasność średnia</div>
            <div class="result-value">${info.brightness.mean} / 255</div>
        </div>`;

        html += `<div class="result-item">
            <div class="result-label">Zakres jasności</div>
            <div class="result-value">${info.brightness.min} - ${info.brightness.max}</div>
        </div>`;
    }

    if (info.file_size_kb !== undefined) {
        html += `<div class="result-item">
            <div class="result-label">Rozmiar pliku</div>
            <div class="result-value">${info.file_size_kb} KB</div>
        </div>`;
    }

    resultContent.innerHTML = html;
    result.classList.add('show');
}

function showError(msg) {
    error.textContent = '❌ ' + msg;
    error.classList.add('show');
}

function showSuccess(msg) {
    success.textContent = '✅ ' + msg;
    success.classList.add('show');
}

function showLoading(show) {
    loading.classList.toggle('show', show);
}

function hideMessages() {
    error.classList.remove('show');
    success.classList.remove('show');
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
                ✎ Zmień plik
            </button>
        </div>
    `;
}
