from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="ch")
# 要识别图片的路径：
img_path = "v2-a593d1448fc27918accaa6b104573840_r.png"
# 识别结果：
result = ocr.ocr(img_path, cls=True)
# 结果输出展示：
for line in result[0]:
    print(line)
