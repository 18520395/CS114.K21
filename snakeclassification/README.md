# VENOMOUS SNAKE CLASSIFICATION

Mỗi năm có từ 80000 đến 140000 người trên thế giới thiệt mạng vì bị rắn độc cắn. Đây được xem là một trong những nguyên nhân mang đến chết chóc lớn nhất đối với loài người nhưng chưa được nhắc tới nhiều, cũng như chưa được đánh giá một cách đầy đủ. Theo thống kê, châu Á và châu Phi là hai khu vực có số người tử vong vì bị rắn độc cắn nhiều nhất với con số lần lượt là 57.000 - 100.000 người và 20.000 - 32.000 người, tiếp theo là Mỹ Latinh - Caribe (3.400 - 5.000), châu Đại Dương (200 - 500) và cuối cùng là châu Âu (30 - 130). Việt Nam là nơi cư ngụ của gần 200 loài rắn, trong đó 53 loài là rắn độc chủ yếu thuộc hai họ rắn lục và rắn hổ. Các vết cắn của rắn độc để lại những hậu quả rất nặng nề, các vết cắn của rắn không độc vẫn rất đáng lưu tâm. Mặc dù hầu hết các loài rắn không chủ động tấn công con người tuy nhiên việc trang bị kiến thức kỹ càng để tránh các tình huống nguy hiểm. Xuất phát từ vấn đề đó, đề tài phân loại rắn độc và không độc giúp tăng hiểu biết về các loài rắn để có được tâm lý vững vàng trong những tỉnh huống đối mặt với các loài rắn khác nhau dù độc hay không. Đề tài này chỉ là bài toán nhỏ và đơn giản trong vấn đề xử lý các vụ rắn cắn.
 
INPUT: Image of a snake

OUTPUT: The snake is venomous or nonvenomous

Dataset:
- From Kaggle: https://www.kaggle.com/sameeharahman/preprocessed-snake-images/data#
- From github: https://github.com/arjun921/Indian-Snakes-Dataset

Dataset: https://drive.google.com/file/d/1x5RF0s1zbmxgKfSTAIf8v0mfer5dQmE9/view?usp=sharing
- Tổng số ảnh: 8890 ảnh
Trong đó: 4808 ảnh rắn độc | 4082 ảnh rắn không độc

Tiền xử lý dữ liệu:
- Thay đổi kích thước ảnh để phù hợp với mạng VGG16 đang sử dụng:
```
list_image = []
for (j, imagePath) in enumerate(image_path):
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    
    image = np.expand_dims(image, 0)
    image = imagenet_utils.preprocess_input(image)
    list_image.append(image)
    
list_image = np.vstack(list_image)
```
Trích xuất đặc trưng:
 ```
features = model.predict(list_image)
features = features.reshape((features.shape[0], 512*7*7))
```
Chia dữ liệu Training/Testing: 80% | 20%

VGG16:
- VGG16 là mạng convolutional neural network được đề xuất bởi K. Simonyan and A. Zisserman, University of Oxford. Model sau khi train bởi mạng VGG16 đạt độ chính xác 92.7% top-5 test trong dữ liệu ImageNet gồm 14 triệu hình ảnh thuộc 1000 lớp khác nhau
- Giữ lại phần ConvNet trong CNN và bỏ đi FCs
- Dùng output của ConvNet để làm input cho SVM và Logistic Regression

Lựa chọn mô hình:
- SVM (Support Vector Machine)
- Logistic Regression
- Random Forest Classifier 

Đánh giá & kết luận:
- Accuracy score của model: 
+ SVM: 0.81
+ Logistic Regression: 0.83
+ Random Forest Classifier: 0.80
- Tuy tỷ lệ dự đoán đúng của model khá cao nhưng model lại phản ứng không tốt với các ảnh dùng để test
- Có thể gặp phải overfitting

Nguyên nhân: Do dataset được xây dựng trên một số loài cụ thể nên các loài ngoài vùng sẽ khó nhận biết, dataset chưa nhiều

Khó khăn:
- Vì hình ảnh các loài rắn không thực sự nhiều và bị trùng lấp nhiều nên xây dựng dataset khó
- Các hình ảnh về các loài rắn không độc ít 
- Việc tự thu thập data trong thực tế mang tính bất khả thi cao 


