from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sqlalchemy import Column, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from PIL import Image
from io import BytesIO
import pickle
from typing import Dict
import numpy as np
import uvicorn
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import dlib
import mysql.connector

app = FastAPI()

# 修改数据库连接 URL，使用 mysql-connector-python 作为 MySQL 驱动程序
DATABASE_URL = "mysql+mysqlconnector://yuxialuozi:Lycdemima1@@localhost/budstudent"
Base = declarative_base()


class FaceEncoding(Base):
    __tablename__ = "face_recognition"

    id = Column(String(50), primary_key=True, index=True)
    face_encoding = Column(LargeBinary)
    student_id = Column(String(50), index=True)


# 创建数据库表
engine = mysql.connector.connect(host="localhost", user="yuxialuozi", password="Lycdemima1@", database="budstudent")
cursor = engine.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_recognition (
        id VARCHAR(50) PRIMARY KEY,
        face_encoding LONGBLOB,
        student_id VARCHAR(50)
    )
""")
engine.commit()
cursor.close()
engine.close()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def compute_face_encoding(image):
    img = np.array(Image.open(image))
    dets = detector(img, 1)

    if len(dets) == 0:
        raise HTTPException(status_code=400, detail="图像中未检测到人脸")

    shape = predictor(img, dets[0])
    face_encoding = np.array(face_reco_model.compute_face_descriptor(img, shape))
    return face_encoding


@app.post("/register-face/")
async def register_face(student_id: int = Form(...), image: UploadFile = File(...)):
    try:
        face_encoding = compute_face_encoding(BytesIO(await image.read()))
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = mysql.connector.connect(host="localhost", user="yuxialuozi", password="Lycdemima1@",
                                       database="budstudent")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO face_recognition (face_encoding, student_id,create_at,update_at) VALUES (%s, %s, "
                       "%s,%s)", (pickle.dumps(face_encoding), student_id, current_time, update_time))
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"message": "人脸注册成功"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize-face/")
async def recognize_face(image: UploadFile = File(...)):
    try:
        face_encoding = compute_face_encoding(BytesIO(await image.read()))
        conn = mysql.connector.connect(host="localhost", user="yuxialuozi", password="Lycdemima1@",
                                       database="budstudent")
        cursor = conn.cursor()
        cursor.execute("SELECT id, face_encoding, student_id FROM face_recognition")
        result = cursor.fetchall()
        cursor.close()
        conn.close()

        for item in result:
            stored_face_encoding = pickle.loads(item[1])
            distance = np.linalg.norm(face_encoding - stored_face_encoding)
            if distance < 0.5:
                return JSONResponse(content={"student_id": item[2]}, status_code=200)

        return JSONResponse(content={"message": "未找到匹配项"}, status_code=404)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_recognized_text(recognized_text: list) -> Dict[str, str]:
    processed_text = {
        "学院": "",
        "专业": "",
        "学制": "",
        "籍贯": "",
        "出生年月": "",
        "学号": "",
        "姓名": "",
        "性别": "",
        "入学年月": ""
    }

    for line in recognized_text:
        if "学院：" in line:
            processed_text["学院"] = line.replace("学院：", "")
        elif "专业：" in line:
            processed_text["专业"] = line.replace("专业：", "")
        elif "班" in line:
            processed_text["专业"] += line
        elif "学制：" in line:
            processed_text["学制"] = line.replace("学制：", "")
        elif "籍贯：" in line:
            processed_text["籍贯"] = line.replace("籍贯：", "")
        elif "出生年月：" in line:
            processed_text["出生年月"] = line.replace("出生年月：", "")
        elif "学号：" in line:
            processed_text["学号"] = line.replace("学号：", "")
        elif "姓名：" in line:
            processed_text["姓名"] = line.replace("姓名：", "")
        elif "性别：" in line:
            processed_text["性别"] = line.replace("性别：", "")
        elif "入学年月：" in line:
            processed_text["入学年月"] = line.replace("入学年月：", "")

    return processed_text


@app.post("/paddle_ocr", tags=["文字识别"])
async def paddle_ocr(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        result = ocr.ocr(contents, cls=True)
        recognized_text = [line[1][0] for line in result[0]]
        processed_text = await process_recognized_text(recognized_text)
        return JSONResponse(content={"recognized_text": processed_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail="OCR 处理失败")


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
