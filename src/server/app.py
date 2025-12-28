from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Konfiguracja
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Stwórz folder uploads jeśli nie istnieje
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Sprawdź czy plik ma dozwoloną rozszerzenie"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image(filepath):
    """
    Analizuje obraz i zwraca informacje
    """
    try:
        with Image.open(filepath) as img:
            # Podstawowe informacje
            width, height = img.size
            format_img = img.format
            mode = img.mode

            # Konwertuj do grayscale dla analizy
            img_gray = img.convert('L')
            pixels = np.array(img_gray)

            # Statystyki
            mean_brightness = float(np.mean(pixels))
            std_brightness = float(np.std(pixels))
            min_brightness = int(np.min(pixels))
            max_brightness = int(np.max(pixels))

            return {
                'success': True,
                'info': {
                    'filename': Path(filepath).name,
                    'size': {
                        'width': width,
                        'height': height,
                        'total_pixels': width * height
                    },
                    'format': format_img,
                    'color_mode': mode,
                    'brightness': {
                        'mean': round(mean_brightness, 2),
                        'std': round(std_brightness, 2),
                        'min': min_brightness,
                        'max': max_brightness
                    },
                    'file_size_kb': round(os.path.getsize(filepath) / 1024, 2)
                }
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Główna strona"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint do uploadu pliku
    """
    # Sprawdź czy plik jest w request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Brak pliku w żądaniu'
        }), 400

    file = request.files['file']

    # Sprawdź czy plik został wybrany
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Plik nie został wybrany'
        }), 400

    # Sprawdź rozszerzenie
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Nieobsługiwany format pliku. Dozwolone: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Zabezpiecz nazwę pliku
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Zapisz plik
        file.save(filepath)

        # Przeanalizuj obraz
        result = analyze_image(filepath)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Błąd przy przetwarzaniu: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'OK',
        'message': 'Serwer działa'
    }), 200


if __name__ == '__main__':
    print("🚀 Serwer Flask uruchomiony na http://localhost:5000")
    print("📁 Folder uploads: " + os.path.abspath(UPLOAD_FOLDER))
    app.run(debug=True, host='0.0.0.0', port=5000)
