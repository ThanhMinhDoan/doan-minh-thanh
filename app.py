"""
Ứng dụng nhận dạng người vs không phải người
Sinh viên: Đoàn Minh Thành
MSSV: 223332848
Lớp: Kỹ thuật Robot và trí tuệ nhân tạo K63
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Cấu hình trang
st.set_page_config(
    page_title="Nhận Dạng Người - Đoàn Minh Thành",
    page_icon="👤",
    layout="centered"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .student-info {
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .human {
        background-color: #d4edda;
        color: #155724;
    }
    .non-human {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Nhận Dạng Người vs Không Phải Người</h1>
    <p>Sử dụng mô hình CNN</p>
</div>
""", unsafe_allow_html=True)

# Thông tin sinh viên
st.markdown("""
<div class="student-info">
    <p><strong>Sinh viên:</strong> Đoàn Minh Thành</p>
    <p><strong>MSSV:</strong> 223332848</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Hằng số
IMG_SIZE = 64

@st.cache_resource
def load_model():
    """Load model đã huấn luyện"""
    try:
        model = keras.models.load_model('human_detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None

def preprocess_image(image):
    """Tiền xử lý ảnh để dự đoán"""
    # Resize ảnh
    image = image.resize((IMG_SIZE, IMG_SIZE))
    # Chuyển sang RGB nếu cần
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Chuyển thành numpy array
    img_array = np.array(image)
    # Rescale
    img_array = img_array / 255.0
    # Thêm batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image):
    """Dự đoán ảnh"""
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return prediction

# Load model
model = load_model()

if model is not None:
    # Upload ảnh
    st.subheader("Tải ảnh lên để kiểm tra")
    uploaded_file = st.file_uploader(
        "Chọn một ảnh...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Hỗ trợ các định dạng: JPG, JPEG, PNG, BMP, WEBP"
    )
    
    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        # Nút dự đoán
        if st.button("Nhận dạng", type="primary", use_container_width=True):
            with st.spinner("Đang phân tích..."):
                prediction = predict(model, image)
                
                # Hiển thị kết quả
                if prediction > 0.5:
                    confidence = prediction * 100
                    st.markdown(f"""
                    <div class="result-box non-human">
                        ❌ KHÔNG PHẢI NGƯỜI<br>
                        <small>Độ tin cậy: {confidence:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = (1 - prediction) * 100
                    st.markdown(f"""
                    <div class="result-box human">
                        ✅ LÀ NGƯỜI<br>
                        <small>Độ tin cậy: {confidence:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Vui lòng đặt file `human_detection_model.h5` vào cùng thư mục với app.py")
    st.info("""
    **Hướng dẫn:**
    1. Huấn luyện model trên Google Colab bằng notebook đã cung cấp
    2. Download file `human_detection_model.h5` 
    3. Đặt file vào cùng thư mục với `app.py`
    4. Chạy lại ứng dụng: `streamlit run app.py`
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Deep Learning - Nhận dạng người sử dụng CNN<br>
    © 2026 Đoàn Minh Thành - 223332848
</div>
""", unsafe_allow_html=True)

