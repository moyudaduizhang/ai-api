import dlib
import numpy as np
import cv2
import pymysql
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sshtunnel import SSHTunnelForwarder
from fastapi import HTTPException
from pydantic import BaseModel
from PIL import Image
import pytesseract
from io import BytesIO
import io
from paddleocr import PaddleOCR

# 通过SSH连接云服务器
server = SSHTunnelForwarder(
    ssh_address_or_host=("39.101.64.176", 22),  # 云服务器地址IP和端口port
    ssh_username="root",  # 云服务器登录账号admin
    ssh_password="20041118Xy?",  # 云服务器登录密码password
    remote_bind_address=('localhost', 3306)  # 数据库服务地址ip,一般为localhost和端口port，一般为3306
)
# 云服务器开启
server.start()
# 初始化 Dlib 检测器和模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1(
    "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class User(BaseModel):
    student_id: str
    file: UploadFile = File(...)


# 初始化 FastAPI 应用
app = FastAPI()


def decoding_FaceStr(encoding_str):
    # print("name=%s,encoding=%s" % (name, encoding))
    # 将字符串转为numpy ndarray类型，即矩阵
    # 转换成一个list
    dlist = encoding_str.strip(' ').split(',')
    # 将list中str转换为float
    dfloat = list(map(float, dlist))
    face_encoding = np.array(dfloat)
    return face_encoding


def encoding_FaceStr(image_face_encoding):
    # 将numpy array类型转化为列表
    encoding__array_list = image_face_encoding.tolist()
    # 将列表里的元素转化为字符串
    encoding_str_list = [str(i) for i in encoding__array_list]
    # 拼接列表里的字符串
    encoding_str = ','.join(encoding_str_list)
    return encoding_str


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "欢迎使用新苗人脸api（测试中）",
        "api_version": "1.0",
        "endpoints": [
            {"name": "register-face", "description": "人脸注册"},
            {"name": "recognize-face", "description": "人脸识别"}
        ],
    }


# 人脸录入接口
@app.post("/register-face/")
async def register_face(student_id: str, file: UploadFile = File(...)):
    """
        Register a face for a student.

        Args:

            student_id (str): The ID of the student.
            file (UploadFile): The image file containing the face to be registered.

        Raises:

            HTTPException: If no face is detected in the uploaded image.
        """
    # 函数体

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 使用 Dlib 检测人脸
    dets = detector(img, 1)
    if len(dets) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    # 取第一个检测到的人脸
    shape = predictor(img, dets[0])

    face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
    face_descriptor = np.array(face_descriptor)
    face_str = encoding_FaceStr(face_descriptor)
    # 连接到数据库
    conn = pymysql.connect(host='127.0.0.1',  # 此处必须是是127.0.0.1
                           port=server.local_bind_port,
                           user="root",  # mysql的登录账号admin
                           password="20041118Xy?",  # mysql的登录密码pwd
                           db="faces",  # mysql中要访问的数据表
                           charset='utf8')  # 表的字符集
    cursor = conn.cursor()

    # 将人脸特征和学生学号存入数据库

    cursor.execute('INSERT INTO face_data (face_encoding, student_id) VALUES (%s, %s)',
                   (face_str, student_id))
    conn.commit()

    # 关闭数据库连接
    cursor.close()
    conn.close()

    return JSONResponse(content={"result": "Face registered"})


# 人脸识别接口
@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    """
        Recognizer a face for a student.

        Args:

            file (UploadFile): The image file containing the face to be recognized.

        Raises:

            HTTPException: If no face is detected in the uploaded image.
        """
    # 读取上传的图像文件并进行人脸检测
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    dets = detector(img, 1)
    if len(dets) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    # 取第一个检测到的人脸并计算特征描述子
    shape = predictor(img, dets[0])
    face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
    face_descriptor = np.array(face_descriptor)

    # 连接到数据库
    conn = pymysql.connect(host='127.0.0.1',  # 此处必须是是127.0.0.1
                           port=server.local_bind_port,
                           user="root",  # mysql的登录账号admin
                           password="20041118Xy?",  # mysql的登录密码pwd
                           db="faces",  # mysql中要访问的数据表
                           charset='utf8')  # 表的字符集
    cursor = conn.cursor()

    # 从数据库中获取存储的人脸特征和学号
    cursor.execute('SELECT face_encoding, student_id FROM face_data')
    rows = cursor.fetchall()

    import pickle

    # 遍历数据库中的人脸特征，计算与上传图像的特征的相似度
    for row in rows:
        stored_face_descriptor = row[0]
        face_encoding = decoding_FaceStr(stored_face_descriptor)
        similarity = np.linalg.norm(face_encoding - face_descriptor)

        # 如果相似度低于阈值，识别为已注册的人脸
        if similarity < 0.5:
            student_id = row[1]
            return JSONResponse(content={"result": f"Face recognized as student {student_id}"})

    # 关闭数据库连接
    cursor.close()
    conn.close()

    return JSONResponse(content={"result": "Face not recognized"})


@app.post("/paddle_ocr")
async def paddle_ocr(image: UploadFile = File(...)):
    # 读取上传的图片文件
    contents = await image.read()

    # 使用 PaddleOCR 进行图片识别
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(contents, cls=True)

    # 整理识别结果
    recognized_text = []
    for line in result[0]:
        recognized_text.append(line[1][0])

    return {"recognized_text": recognized_text}


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
