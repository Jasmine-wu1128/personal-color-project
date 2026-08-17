from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

def map_to_16_seasons(h, s, v):
    """
    16 型個人色彩判斷引擎
    h (0~180): 色調 Hue
    s (0~255): 飽和度 Saturation
    v (0~255): 明度 Value
    """
    # 1. 特徵分數歸一化 (-1.0 到 1.0)
    if h < 25:
        warm_score = (15 - h) / 15.0
    elif h > 150:
        warm_score = -((h - 150) / 30.0)
    else:
        warm_score = 0.0

    light_score = (v - 128.0) / 128.0
    clear_score = (s - 100.0) / 100.0

    is_warm = warm_score >= 0
    is_light = light_score >= 0
    is_clear = clear_score >= 0

    # 2. 16 型判斷邏輯
    if is_warm:
        if is_light:
            if abs(clear_score) > abs(light_score) and is_clear:
                season = "亮春 (Bright Spring)"
            elif abs(warm_score) > abs(light_score):
                season = "暖春 (Warm Spring)"
            elif clear_score < -0.2:
                season = "柔春 (Light/Soft Spring)"
            else:
                season = "淺春 (Light Spring)"
        else:
            if not is_clear and abs(clear_score) > abs(light_score):
                season = "柔秋 (Soft Autumn)"
            elif abs(warm_score) > abs(light_score):
                season = "暖秋 (Warm Autumn)"
            elif not is_light and abs(light_score) > 0.4:
                season = "深秋 (Deep Autumn)"
            else:
                season = "強秋 (Strong/Vibrant Autumn)"
    else:
        if is_light:
            if abs(light_score) > abs(clear_score) and is_light:
                season = "淺夏 (Light Summer)"
            elif abs(warm_score) < -0.4:
                season = "冷夏 (Cool Summer)"
            elif not is_clear:
                season = "柔夏 (Soft Summer)"
            else:
                season = "清夏 (Clear Summer)"
        else:
            if not is_light and abs(light_score) > 0.4:
                season = "深冬 (Deep Winter)"
            elif abs(warm_score) < -0.4:
                season = "冷冬 (Cool Winter)"
            elif is_clear and abs(clear_score) > 0.3:
                season = "亮冬 (Bright Winter)"
            else:
                season = "鮮冬 (Vibrant/Clear Winter)"

    return {
        "season": season,
        "scores": {
            "warmness": round(warm_score, 2),
            "lightness": round(light_score, 2),
            "clearness": round(clear_score, 2)
        },
        "is_warm": is_warm,
        "is_light": is_light,
        "is_clear": is_clear
    }


def analyze_skin(image_bytes):
    """
    OpenCV + K-Means 膚色分析核心邏輯（加入尺寸標準化、像素抽樣與主色彩權重修正）
    """
    # 1. 圖片解碼
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return None

    # 尺寸標準化：統一縮放至 500x500，確保跨裝置解析度一致
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)

    # 轉為 HSV 色彩空間
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 膚色範圍遮罩過濾 (HSV 閥值)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
    skin_pixels = hsv_img[mask > 0]

    # 3. 若過濾後像素不足，降級採用圖像中央區域
    if len(skin_pixels) < 100:
        h, w, _ = img.shape
        skin_pixels = hsv_img[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)].reshape(-1, 3)

    # 【速度優化】：限制最多抽樣 5,000 個像素點，大幅提昇運算速度
    if len(skin_pixels) > 5000:
        np.random.seed(42)
        indices = np.random.choice(len(skin_pixels), 5000, replace=False)
        skin_pixels = skin_pixels[indices]

    # K-Means 聚類取出最多的主要膚色群心
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(skin_pixels)
    
    # 計算每個群集的像素數量，取數量最多的群心，避免誤取邊緣雜色
    counts = np.bincount(labels)
    dominant_cluster_idx = np.argmax(counts)
    dominant_hsv = kmeans.cluster_centers_[dominant_cluster_idx]

    return {
        'avg_h': float(dominant_hsv[0]),
        'avg_s': float(dominant_hsv[1]),
        'avg_v': float(dominant_hsv[2])
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "未收到圖片檔案"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "未選擇檔案"}), 400

        # 重置檔案讀取指標，避免讀取到空位元組
        file.seek(0)
        image_bytes = file.read()

        if not image_bytes:
            return jsonify({"error": "圖片資料為空"}), 400

        # 執行影像處理
        raw_result = analyze_skin(image_bytes)
        if raw_result is None:
            return jsonify({"error": "無法解碼圖片，請更換清晰照片"}), 400

        # 進行 16 型判定
        analysis_data = map_to_16_seasons(
            raw_result['avg_h'],
            raw_result['avg_s'],
            raw_result['avg_v']
        )

        return jsonify(analysis_data)

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": f"伺服器處理失敗: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)