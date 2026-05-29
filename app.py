import os
import cv2
import numpy as np
import base64
from flask import Flask, request, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Путь к локальной модели
MODEL_PATH = os.environ.get('MODEL_PATH', 'best.pt')
model = YOLO(MODEL_PATH)

# Минимальный порог уверенности (0.5 = 50%)
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.6))

# Коэффициенты конвертации
CONVERSION = {
    'apple':    0.15,
    'cucumber': 0.1,
    'tomato':   0.2
}
UNIT = {
    'apple':    'л компота',
    'cucumber': 'банки',
    'tomato':   'банки'
}

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Инференс с порогом уверенности
    results = model(img, conf=CONFIDENCE_THRESHOLD)[0]

    counts = {'apple': 0, 'cucumber': 0, 'tomato': 0}
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            confidence = float(box.conf.cpu().numpy())
            cls_name = model.names[cls_id]
            
            # Дополнительная проверка: учитываем только нужные классы
            # и уверенность выше порога
            if cls_name in counts and confidence >= CONFIDENCE_THRESHOLD:
                counts[cls_name] += 1

    # Формируем текстовый отчёт
    lines = []
    total_compot = 0.0
    total_banki = 0.0
    
    for cls_name in ['apple', 'cucumber', 'tomato']:
        cnt = counts[cls_name]
        if cnt > 0:
            value = cnt * CONVERSION[cls_name]
            unit = UNIT[cls_name]
            
            # Более читаемое форматирование
            if cls_name == 'apple':
                lines.append(f'<span class="highlight green-highlight">Яблоки: {cnt} шт.</span> ≈ <strong>{value:.2f} {unit}</strong>')
                total_compot += value
            elif cls_name == 'cucumber':
                lines.append(f'<span class="highlight green-highlight">Огурцы: {cnt} шт.</span> ≈ <strong>{value:.2f} {unit}</strong>')
                total_banki += value
            elif cls_name == 'tomato':
                lines.append(f'<span class="highlight">Помидоры: {cnt} шт.</span> ≈ <strong>{value:.2f} {unit}</strong>')
                total_banki += value

    if not lines:
        summary = '<div class="empty-state">На изображении не найдено яблок, огурцов или помидоров</div>'
    else:
        summary = '<br>'.join(lines)

    # Рисуем аннотированное изображение
    annotated_img = results.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64, summary, total_compot, total_banki

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    text_info = None
    total_compot = 0.0
    total_banki = 0.0

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            image_bytes = file.read()
            try:
                img_data, text_info, total_compot, total_banki = process_image(image_bytes)
            except Exception as e:
                text_info = f'<div class="empty-state">❌ Ошибка обработки: {e}</div>'

    return render_template('index.html', 
                         img_data=img_data, 
                         text_info=text_info,
                         total_compot=total_compot,
                         total_banki=total_banki)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)