# Phân Loại Người Đeo Khẩu Trang

Dự án sử dụng học sâu để nhận diện và phân loại người đeo khẩu trang theo 3 nhóm:

- Đeo khẩu trang **đúng cách**  
- Đeo khẩu trang **không đúng cách**  
- **Không đeo** khẩu trang  

---

## Mục tiêu kỹ thuật

- Tốc độ xử lý: **> 5 FPS**
- Kích thước mô hình: **≤ 7 BFLOPS**

---

## Dataset

Bộ dữ liệu huấn luyện bao gồm các hình ảnh khuôn mặt của người đeo khẩu trang, được phân loại thành ba nhóm:

### **Training Data**
- **Class 1 (Đeo khẩu trang đúng cách)**: 3657 ảnh
- **Class 2 (Đeo khẩu trang sai cách)**: 4400 ảnh
- **Class 3 (Không đeo khẩu trang)**: 3703 ảnh

**Link Dataset**: [https://www.kaggle.com/datasets/thuongbuirvc/dataset-version001/data](#)

### **Testing Data**
- **Class 1 (Đeo khẩu trang đúng cách)**: 250 ảnh
- **Class 2 (Đeo khẩu trang sai cách)**: 250 ảnh
- **Class 3 (Không đeo khẩu trang)**: 250 ảnh

---

### Cấu hình phần cứng thử nghiệm

- **Thiết bị**: Raspberry Pi 5 – RAM 8GB  
- **Máy tính**: Dell Precision 7550  
- **CPU**: Intel Core i7-10850H  
- **GPU**: Không sử dụng GPU  
- **Camera**: NetCAM PC 930 – 30 FPS

---

## Kết Quả Đạt Được

- **Độ chính xác trên tập validation và training với metric accuracy**: > 99%
- **Độ chính xác trên tập testing**: > 99%

### **Hiệu suất FPS**:

- **Trên PC với webcam**: Khoảng **20 FPS**
- **Trên Raspberry Pi 5**: Khoảng **15 FPS**

---

## Notebook 1: `PreprocessingData.ipynb`

### Bước 1: Phát hiện và cắt khuôn mặt

- Sử dụng `haarcascade_frontalface_default.xml` để phát hiện khuôn mặt. Mục tiêu của bước này là phát hiện và cắt khuôn mặt ra khỏi ảnh gốc. 
- Nếu khuôn mặt được phát hiện, khuôn mặt sẽ được lưu vào thư mục đầu ra để tạo tập dữ liệu huấn luyện, giúp giảm ảnh hưởng của phông nền:
  - Cắt khuôn mặt và lưu ảnh vào `output_folder`
- Nếu không phát hiện được khuôn mặt:
  - Giữ nguyên ảnh gốc để xử lý tiếp ở bước 2

#### Luồng xử lý:

1. Duyệt ảnh trong `input_folder`  
2. Phát hiện khuôn mặt bằng Haar Cascade  
3. Cắt & lưu ảnh nếu có khuôn mặt  
4. Nếu không có → sao chép ảnh gốc sang `output_folder`

### Bước 2: Phát hiện đặc trưng (mắt hoặc mặt)

- Bước này dùng để loại bỏ những ảnh không phải khuôn mặt (ảnh background) khỏi tập dữ liệu. 
- Hệ thống sẽ kiểm tra xem có đặc trưng nào như mắt hoặc mặt xuất hiện trong ảnh không.
- Dùng `haarcascade_eye.xml` hoặc `haarcascade_frontalface_default.xml`
- Loại bỏ ảnh không chứa khuôn mặt hoặc mắt

#### Luồng xử lý:

1. Chuyển ảnh sang **grayscale**  
2. Phát hiện đặc trưng (mắt hoặc mặt)  
3. Nếu phát hiện:
   - Cắt và lưu ảnh chứa đặc trưng  
4. Nếu không phát hiện:
   - Loại bỏ hoặc sao chép ảnh để xử lý sau

---

## Notebook 2: `TrainingModelWithEfficientNetB0.ipynb`

### Mục tiêu

Huấn luyện mô hình phân loại ảnh khuôn mặt thành 3 lớp:

- Đeo đúng  
- Đeo sai  
- Không đeo khẩu trang

### Pipeline chính:

#### 1. Tiền xử lý dữ liệu

- Sử dụng `ImageDataGenerator`:
  - Rescale ảnh về [0, 1]
  - Chia tập train/test theo tỉ lệ 80/20
  - Tự động gán nhãn từ tên thư mục

#### 2. Xây dựng mô hình

- Sử dụng `EfficientNetB0` (pretrained, `include_top=False`)
- Thêm các lớp đầu ra:
  - `GlobalAveragePooling2D`
  - `Dense(1024, activation='relu')`
  - `Dense(3, activation='softmax')`

#### 3. Huấn luyện

- **Loss**: `sparse_categorical_crossentropy`  
- **Optimizer**: `Adam`  
- **Metric**: `accuracy`  

#### 4. Lưu mô hình

- Lưu dưới định dạng `.h5` để sử dụng sau

---

## Notebook 3: `TestingModelOnPC.ipynb`

### Mục tiêu

Kiểm thử mô hình đã huấn luyện trên video hoặc camera thời gian thực, hiển thị kết quả phân loại và bounding box khuôn mặt.

### Tính năng

- Dự đoán 3 lớp khẩu trang: đúng – sai – không đeo
- Hiển thị nhãn phân loại + độ tin cậy (% confidence)
- Vẽ bounding box quanh khuôn mặt
- Tính toán và hiển thị FPS (khung hình/giây)

### Cấu hình phần cứng thử nghiệm

- **Máy tính**: Dell Precision 7550  
- **CPU**: Intel Core i7-10850H  
- **GPU**: Không sử dụng GPU  
- **Camera**: NetCAM PC 930 – 30 FPS

### Đầu ra

- Cửa sổ hiển thị hình ảnh từ camera
- Có vẽ nhãn và bounding box
- Có hiển thị FPS góc trên bên trái

---

## Notebook 4: `TestingModelOnRaspberryPi5.ipynb`

### 🎯 Mục tiêu

Thực hiện nhận diện khẩu trang thời gian thực với TensorFlow Lite – tối ưu cho thiết bị nhẹ như Raspberry Pi.

### Luồng xử lý chính

1. **Phát hiện khuôn mặt**  
   - Sử dụng mô hình phát hiện đối tượng (SSD, MobileNet-SSD, v.v.).
   - Chuyển tọa độ box từ tỷ lệ → pixel.

2. **Chuẩn bị ảnh đầu vào**  
   - Cắt khuôn mặt → resize (224x224) → normalize [0, 1].
   - Thêm batch dimension.

3. **Dự đoán bằng TFLite**  
   - Nạp dữ liệu → `interpreter.set_tensor(...)`
   - Gọi `interpreter.invoke()` để suy luận.
   - Lấy nhãn dự đoán và độ tin cậy.

4. **Hiển thị kết quả**  
   - Vẽ bounding box + text (label + confidence %).
   - Dùng màu khác nhau để phân biệt nhãn.

5. **Tính FPS**  
   - Cứ mỗi 10 khung hình → tính thời gian → cập nhật FPS.

6. **Vòng lặp chính**  
   - Hiển thị camera liên tục.
   - Thoát bằng phím "q".
   - Giải phóng camera và cửa sổ khi kết thúc.

### Cấu hình phần cứng thử nghiệm

- **Thiết bị**: Raspberry Pi 5 – RAM 8GB  
- **Camera**: NetCAM PC 930 – 30 FPS

---

## Cấu trúc thư mục

```
├── PreprocessingData.ipynb
├── TrainingModelWithEfficientNetB0.ipynb
├── TestingModelOnPC.ipynb
├── TestingModelOnRaspberryPi5.ipynb
├── model/
│   └── efficientnetb0.h5
│   └── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│   └── efficientnetb0.tflite
├── haarcascade/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
└── README.md
```

---

