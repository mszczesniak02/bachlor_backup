#autopep8: off
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import numpy as np
from PIL import Image
import base64
from io import BytesIO

import sys

# Add src to path to allow imports from sibling packages
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

from geometric_analysis.pipeline import CrackAnalysisPipeline

# Konfiguracja
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Stwórz folder uploads jeśli nie istnieje
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize Pipeline (Loads models into memory)
# Prevent double loading when using Flask reloader (debug=True)
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    print("⏳ Loading AI Models...")
    PIPELINE = CrackAnalysisPipeline()
    print("✅ AI Models Loaded!")
else:
    print("⏳ Skipping Model Load in Reloader Process...")
    PIPELINE = None


def allowed_file(filename):
    """Sprawdź czy plik ma dozwoloną rozszerzenie"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img_numpy):
    """Encodes numpy image (RGB) to base64 string"""
    try:
        # Check if float, convert to uint8
        if img_numpy.dtype != np.uint8:
            img_numpy = (img_numpy * 255).astype(np.uint8)

        img_pil = Image.fromarray(img_numpy)
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def convert_numpy_types(obj):
    """
    Recursively converts NumPy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def analyze_image(filepath):
    """
    Analizuje obraz i zwraca informacje
    """
    try:
        # Basic File Info
        file_stats = {
            'filename': Path(filepath).name,
            'file_size_kb': round(os.path.getsize(filepath) / 1024, 2)
        }

        with Image.open(filepath) as img:
            file_stats['size'] = {
                'width': img.width, 
                'height': img.height,
                'total_pixels': img.width * img.height
            }
            file_stats['format'] = img.format
            file_stats['mode'] = img.mode

        # Run AI Analysis
        print(f"Running pipeline on {filepath}...")
        results, feature_images = PIPELINE.run_pipeline(filepath)

        # Prepare response
        response_data = {
            'success': True,
            'info': file_stats,
            'analysis': {
                'domain_controller': results.get('domain_controller'),
                'segmentation': results.get('segmentation_completed'),
                'classification': results.get('classification'),
                'geometric': results.get('geometric_analysis')
            },
            'images': {}
        }

        # Encode result images
        if feature_images:
            if 'original' in feature_images:
                response_data['images']['original'] = image_to_base64(
                    feature_images['original'])
            if 'overlay' in feature_images:
                response_data['images']['overlay'] = image_to_base64(
                    feature_images['overlay'])
            if 'heatmap' in feature_images:
                response_data['images']['heatmap'] = image_to_base64(
                    feature_images['heatmap'])
            if 'binary_mask' in feature_images:
                response_data['images']['mask'] = image_to_base64(
                    feature_images['binary_mask'])

        # Sanitize for JSON
        response_data = convert_numpy_types(response_data)

        return response_data

    except Exception as e:
        print(f"Error in analyze_image: {e}")
        import traceback
        traceback.print_exc()
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
