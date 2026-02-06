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
import requests
from io import BytesIO

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
    <p><strong>MSV:</strong> 223332848</p>
    <p><strong>Lớp:</strong> Kỹ thuật robot và trí tuệ nhân tạo K63</p>
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
    # Tạo tabs cho các phương thức nhập ảnh
    tab1, tab2, tab3 = st.tabs(["📁 Tải ảnh lên", "📷 Webcam", "🔗 URL ảnh"])
    
    image = None
    
    # Tab 1: Upload ảnh
    with tab1:
        st.subheader("Tải ảnh lên để kiểm tra")
        uploaded_file = st.file_uploader(
            "Chọn một ảnh...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Hỗ trợ các định dạng: JPG, JPEG, PNG, BMP, WEBP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            
            if st.button("Nhận dạng", type="primary", use_container_width=True, key="btn_upload"):
                with st.spinner("Đang phân tích..."):
                    prediction = predict(model, image)
                    
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
    
    # Tab 2: Webcam
    with tab2:
        st.subheader("Chụp ảnh từ Webcam")
        camera_image = st.camera_input("Chụp ảnh từ webcam của bạn")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            if st.button("Nhận dạng", type="primary", use_container_width=True, key="btn_webcam"):
                with st.spinner("Đang phân tích..."):
                    prediction = predict(model, image)
                    
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
    
    # Tab 3: URL ảnh
    with tab3:
        st.subheader("Nhập URL ảnh")
        image_url = st.text_input(
            "Nhập đường dẫn URL của ảnh:",
            placeholder="https://example.com/image.jpg",
            help="Dán đường link trực tiếp đến ảnh (JPG, PNG, WEBP...)"
        )
        
        if image_url:
            try:
                # Thêm headers để tránh bị chặn
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': image_url
                }
                response = requests.get(image_url, headers=headers, timeout=15, allow_redirects=True)
                response.raise_for_status()
                
                # Kiểm tra content type
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    st.error("URL không trỏ đến file ảnh trực tiếp. Vui lòng sử dụng link ảnh gốc (click chuột phải vào ảnh → Sao chép địa chỉ hình ảnh)")
                else:
                    image_data = BytesIO(response.content)
                    image = Image.open(image_data)
                    # Đảm bảo ảnh được load hoàn toàn
                    image.load()
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(image, caption="Ảnh từ URL", use_container_width=True)
                    
                    if st.button("Nhận dạng", type="primary", use_container_width=True, key="btn_url"):
                        with st.spinner("Đang phân tích..."):
                            prediction = predict(model, image)
                            
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
            except requests.exceptions.RequestException as e:
                st.error(f"Không thể tải ảnh từ URL: {e}")
            except Exception as e:
                st.error(f"Không thể xử lý ảnh. Hãy thử dùng link ảnh trực tiếp (kết thúc bằng .jpg, .png, .webp...)")
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
