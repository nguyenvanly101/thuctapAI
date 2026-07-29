import os
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from model_engine import PersimmonClassifier, CLASSES, CLASS_DIR_MAP

app = Flask(__name__)
app.config['SECRET_KEY'] = 'persimmon_classification_secret_2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['DATASET_FOLDER'] = os.path.join(os.path.dirname(__file__), 'dataset')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
for sub_dir in CLASS_DIR_MAP.keys():
    os.makedirs(os.path.join(app.config['DATASET_FOLDER'], sub_dir), exist_ok=True)

# Initialize Classifier Engine (Default: EfficientNetV2)
classifier = PersimmonClassifier(model_type="efficientnet_v2")

ALLOWED_IMAGE_EXTS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_VIDEO_EXTS = {'mp4', 'avi', 'mov', 'mkv'}

def is_allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_image():
    """Predict persimmon type from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh được tải lên'}), 400
    
    file = request.files['image']
    if file.filename == '' or not is_allowed_file(file.filename, ALLOWED_IMAGE_EXTS):
        return jsonify({'error': 'Định dạng file không hợp lệ (Chấp nhận JPG, PNG, WEBP)'}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    
    results, err = classifier.predict(save_path)
    if err:
        return jsonify({'error': err}), 500
    
    results['image_url'] = f'/static/uploads/{filename}'
    return jsonify(results)

@app.route('/api/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Get current statistics of collected dataset"""
    stats = {}
    total_images = 0
    total_videos = 0

    for dir_name, display_name in CLASS_DIR_MAP.items():
        path = os.path.join(app.config['DATASET_FOLDER'], dir_name)
        files = os.listdir(path) if os.path.exists(path) else []
        img_count = sum(1 for f in files if is_allowed_file(f, ALLOWED_IMAGE_EXTS))
        vid_count = sum(1 for f in files if is_allowed_file(f, ALLOWED_VIDEO_EXTS))
        
        total_images += img_count
        total_videos += vid_count
        
        stats[dir_name] = {
            'display_name': display_name,
            'image_count': img_count,
            'video_count': vid_count,
            'total_files': len(files)
        }
        
    return jsonify({
        'classes': stats,
        'total_images': total_images,
        'total_videos': total_videos,
        'model_status': 'Custom Trained' if classifier.is_custom_trained else 'Pre-trained + Domain Heuristic',
        'current_architecture': classifier.model_type.upper()
    })

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset_item():
    """Upload multiple images or videos into specific dataset folder"""
    target_class = request.form.get('class_name')
    if target_class not in CLASS_DIR_MAP:
        return jsonify({'error': 'Loại quả hồng không hợp lệ'}), 400
    
    uploaded_files = request.files.getlist('files') or request.files.getlist('file')
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ''):
        return jsonify({'error': 'Chưa chọn file để tải lên'}), 400
    
    dest_dir = os.path.join(app.config['DATASET_FOLDER'], target_class)
    saved_count = 0
    total_extracted_frames = 0

    for file in uploaded_files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            save_path = os.path.join(dest_dir, filename)
            file.save(save_path)
            saved_count += 1
            
            # If file is a video, extract frames automatically!
            if is_allowed_file(filename, ALLOWED_VIDEO_EXTS):
                frames = extract_video_frames(save_path, dest_dir, prefix=os.path.splitext(filename)[0])
                total_extracted_frames += frames
        
    return jsonify({
        'message': f'Đã lưu thành công {saved_count} tệp vào thư mục [{CLASS_DIR_MAP[target_class]}]',
        'saved_count': saved_count,
        'extracted_frames': total_extracted_frames
    })

def extract_video_frames(video_path, output_dir, prefix="frame", frame_interval=15):
    """Extract frames from uploaded video file using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved >= 15: # Extract max 15 clear frames
            break
        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score > 60: # Blur check
                frame_filename = f"vid_{prefix}_f{saved:03d}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
                saved += 1
        count += 1
    cap.release()
    return saved

@app.route('/api/train', methods=['POST'])
def train_model():
    """Trigger fine-tuning of the model on the current dataset"""
    data = request.get_json() or {}
    epochs = int(data.get('epochs', 10))
    lr = float(data.get('learning_rate', 0.0003))
    
    success, result = classifier.train_on_dataset(
        dataset_dir=app.config['DATASET_FOLDER'],
        epochs=epochs,
        lr=lr
    )
    
    if not success:
        return jsonify({'error': result}), 400
        
    return jsonify({
        'message': 'Huấn luyện mô hình thành công!',
        'history': result,
        'model_status': 'Custom Fine-Tuned Weights Active'
    })

@app.route('/api/switch_architecture', methods=['POST'])
def switch_architecture():
    """Switch model backbone (efficientnet_v2, resnet50, mobilenet_v3)"""
    data = request.get_json() or {}
    arch = data.get('architecture', 'efficientnet_v2').lower()
    
    global classifier
    classifier = PersimmonClassifier(model_type=arch)
    
    return jsonify({
        'message': f'Đã chuyển đổi mô hình sang [{arch.upper()}]',
        'current_architecture': classifier.model_type.upper(),
        'is_custom_trained': classifier.is_custom_trained
    })

@app.route('/dataset_file/<class_name>/<filename>')
def serve_dataset_file(class_name, filename):
    """Serve dataset images/videos directly for gallery preview"""
    if class_name not in CLASS_DIR_MAP:
        return "Invalid class", 400
    dir_path = os.path.join(app.config['DATASET_FOLDER'], class_name)
    return send_from_directory(dir_path, filename)

@app.route('/api/dataset_items', methods=['GET'])
def list_dataset_items():
    """READ: Get all files in dataset categorized by class with metadata"""
    class_filter = request.args.get('class_name', 'all')
    items = []

    target_classes = [class_filter] if class_filter in CLASS_DIR_MAP else list(CLASS_DIR_MAP.keys())

    for c_name in target_classes:
        dir_path = os.path.join(app.config['DATASET_FOLDER'], c_name)
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    is_img = is_allowed_file(f, ALLOWED_IMAGE_EXTS)
                    is_vid = is_allowed_file(f, ALLOWED_VIDEO_EXTS)
                    if is_img or is_vid:
                        items.append({
                            'filename': f,
                            'class_name': c_name,
                            'class_display': CLASS_DIR_MAP[c_name],
                            'file_url': f'/dataset_file/{c_name}/{f}',
                            'size_kb': round(stat.st_size / 1024, 1),
                            'is_video': is_vid
                        })
                        
    return jsonify({
        'items': items,
        'count': len(items)
    })

@app.route('/api/delete_dataset_item', methods=['POST'])
def delete_dataset_item():
    """DELETE: Remove a file from the dataset"""
    data = request.get_json() or {}
    c_name = data.get('class_name')
    filename = secure_filename(data.get('filename', ''))

    if c_name not in CLASS_DIR_MAP or not filename:
        return jsonify({'error': 'Tham số không hợp lệ'}), 400

    file_path = os.path.join(app.config['DATASET_FOLDER'], c_name, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'message': f'Đã xóa tệp {filename} khỏi dataset'})
    return jsonify({'error': 'Không tìm thấy tệp cần xóa'}), 404

@app.route('/api/move_dataset_item', methods=['POST'])
def move_dataset_item():
    """UPDATE: Move/reclassify a dataset item to another class folder"""
    data = request.get_json() or {}
    src_class = data.get('source_class')
    dst_class = data.get('target_class')
    filename = secure_filename(data.get('filename', ''))

    if src_class not in CLASS_DIR_MAP or dst_class not in CLASS_DIR_MAP or not filename:
        return jsonify({'error': 'Tham số phân loại không hợp lệ'}), 400

    src_path = os.path.join(app.config['DATASET_FOLDER'], src_class, filename)
    dst_path = os.path.join(app.config['DATASET_FOLDER'], dst_class, filename)

    if os.path.exists(src_path):
        import shutil
        shutil.move(src_path, dst_path)
        return jsonify({'message': f'Đã chuyển {filename} sang [{CLASS_DIR_MAP[dst_class]}]'})
    return jsonify({'error': 'Không tìm thấy tệp cần chuyển'}), 404

@app.route('/api/rename_dataset_item', methods=['POST'])
def rename_dataset_item():
    """UPDATE: Rename a dataset item file"""
    data = request.get_json() or {}
    c_name = data.get('class_name')
    old_name = secure_filename(data.get('old_filename', ''))
    new_name = secure_filename(data.get('new_filename', ''))

    if c_name not in CLASS_DIR_MAP or not old_name or not new_name:
        return jsonify({'error': 'Tham số đổi tên không hợp lệ'}), 400

    dir_path = os.path.join(app.config['DATASET_FOLDER'], c_name)
    src_path = os.path.join(dir_path, old_name)
    dst_path = os.path.join(dir_path, new_name)

    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
        return jsonify({'message': f'Đã đổi tên tệp thành {new_name}'})
    return jsonify({'error': 'Không tìm thấy tệp'}), 404

if __name__ == '__main__':
    print("=" * 60)
    print("  ỨNG DỤNG NHẬN DIỆN PHÂN LOẠI QUẢ HỒNG (EFFICIENTNETV2)")
    print("  Hồng Trung Quốc | Hồng Lạng Sơn | Hồng Đà Lạt")
    print("  Server running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)

