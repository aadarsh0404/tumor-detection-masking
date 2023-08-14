import streamlit as st
import os
import numpy as np
import cv2
from glob import glob

st.title('Brain Tumor Detection')

uploaded_image = st.file_uploader(label='Upload Brain X-Ray Image',
                 type=['png','jpg'],
                 accept_multiple_files=False,
                 )

im_col1, im_col2 = st.columns(2)
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    cv2.imwrite('./uploaded/' + uploaded_image.name, opencv_image)

    # Now do something with the image! For example, let's display it:
    with im_col1:
        st.header('Uploaded Image')
        st.image(cv2.resize(opencv_image, (320, 320)), channels="BGR")

output_image = None
button_col1, button_col2 = st.columns(2)
with button_col1:
    if uploaded_image is None:
        predict = st.button('Detect Tumor', disabled=True)
    else:
        predict = st.button('Detect Tumor', disabled=False)

if predict:
    with st.spinner('Detecting...'):
        os.system("yolo task=segment mode=predict model=./weights/bestl.pt conf=0.25 source=uploaded/"+uploaded_image.name+" project=OUTPUT name=run save")
    output_image_path = glob('./output/**/' + uploaded_image.name)
    output_image = cv2.imread(output_image_path[0], 1)

    with im_col2:
        st.header('Predicted Image')
        st.image(cv2.resize(output_image, (320, 320)), channels="BGR")

with button_col2:
    if output_image is not None:
        clear = st.button('Clear Output', disabled=False)
    else:
        clear = st.button('Clear Output', disabled=True)


if clear:
    st.session_state['output_image'] = None