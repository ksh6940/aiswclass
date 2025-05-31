import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os

#### 모델 코드
def loadmodel(filename):
    np.set_printoptions(suppress=True)
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(f"./upload/{filename}").convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", confidence_score)

#### 모델 코드 끝

st.title('This is my first DeepLearning webapp')
st.info('건강한 산호초와 백색의 죽은 산호초를 분류해서 해양의 오염 정도를 판단하는 딥러닝 개발')
c1, c2 = st.columns(2)
with c1: 
    st.subheader('건강한 산호초')
    st.image('./healthy_coral.png')
with c2:
    st.subheader('죽은 산호초')
    st.image('./bleached_coral.png')

uploaded_file = st.file_uploader("산호 이미지를 업로드하세요 (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    save_path = os.path.join('./upload', uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("분류 시작"):
        st.write("🔍 이미지를 분석 중입니다...")
        class_name, confidence = loadmodel(uploaded_file.name)
        
        st.success("✅ 분류 결과")
        st.markdown(f"**예측된 클래스:** `{class_name}`")
        st.markdown(f"**신뢰도:** `{confidence:.2%}`")
        
        if "bleached" in class_name.lower():
            st.warning("⚠️ 이 산호는 죽은 산호초로 분류되었습니다. 해양 환경에 악영향이 있을 수 있습니다.")
        else:
            st.info("🌿 이 산호는 건강한 산호초로 분류되었습니다.")
