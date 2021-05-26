
import tensorflow as tf
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# define container
header = st.beta_container()
features = st.beta_container()
classification = st.beta_container()

def display_image(img1, img2, img3):
  col1, col2, col3 = st.beta_columns(3)
  image1 = Image.open(img1)
  # image1 = cv2.resize(image1, dsize=(311, 311))
  col1.image(image1, use_column_width=True)

  image2 = Image.open(img2)
  # image2 = cv2.resize(image2, dsize=(311, 311))
  col2.image(image2, use_column_width=True)

  image3 = Image.open(img3)
  # image3 = cv2.resize(image3, dsize=(311, 311))
  col3.image(image3, use_column_width=True)
  # images = [img1, img2, img3]
  # st.image(images, width=200)

# header section
with header:
  st.title('IDENTIFIKASI PENYAKIT PNEUMONIA COVID-19 BERBASIS CITRA COMPUTED TOMOGRAPHY (CT) MENGGUNAKAN DEEP TRANSFER LEARNING')
  st.write('Ini adalah aplikasi web klasifikasi gambar sederhana untuk memprediksi dan mengidentifikasi kehadiran pneumonia yang disebabkan oleh COVID-19 berdasarkan pencitraan CT')
  st.subheader('Batasan Penelitian:')
  st.write('* Penelitian ini hanya melakukan identifikasi untuk dua kelas yaitu COVID-19 dan Non-COVID-19. Pneumonia Non-COVID-19 tidak diikutsertakan.')
  st.write('* Daerah yang menjadi fokus pada penelitian ini adalah bagian toraks (dada) pasien.')

with features:
  st.header('Fitur Klinis Pneumonia COVID-19 pada Citra CT')
  st.write('Fitur utama dari pneumonia COVID-19 pada CT adalah adanya *Ground Glass Opacity (GGO)*, *crazy paving pattern*, atau *consolidation*. Temuan CT pada tahap awal infeksi COVID-19 bilateral, multilobar GGO dengan distribusi periferal atau posterior yang dominan berada pada lobus bagian bawah.')
  # GGO
  st.markdown('* **Ground Glass Opacity (GGO):** GGO adalah area dengan peningkatan opasitas pada paru-paru yang terlihat samar namun tidak menutupi tepi pembuluh paru-paru.')
  display_image('image/ggo/AnotaGÇí+¦o 2020-04-27 233937.png', 'image/ggo/AnotaGÇí+¦o 2020-04-28 132704.png', 'image/ggo/AnotaGÇí+¦o 2020-04-28 133031.png')
  
  # crazy paving pattern
  st.markdown('* **Crazy Paving Pattern:** Pola ini ditandai dengan penebalan interlobular septa dan garis intralobular yang menumpuk pada GGO dengan bentuk yang tidak beraturan.')
  display_image('image/crazy paving pattern/AnotaGÇí+¦o 2020-04-28 103436.png', 'image/crazy paving pattern/AnotaGÇí+¦o 2020-04-28 104056.png', 'image/crazy paving pattern/AnotaGÇí+¦o 2020-04-28 131523.png')
  
  # consolidation
  st.markdown('* **Consolidation:** Consolidation tampak seperti peningkatan atenuasi parenkim paru-paru homogen yang menyamarkan batas pembuluh darah dan dinding saluran pernapasan.')
  display_image('image/consolidation/AnotaGÇí+¦o 2020-04-28 151224.png', 'image/consolidation/AnotaGÇí+¦o 2020-04-28 185240.png', 'image/consolidation/AnotaGÇí+¦o 2020-04-29 181246.png')

with classification:
  st.header('Prediksi Pneumonia COVID-19')
  st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('../Model/FinalModel_DenseCov.hd5')
  return model

with st.spinner('Loading model into memory...'):
  model = load_model()

classes = ['COVID-19', 'Non-COVID-19']

def preprocess_img(image):
  size = (256, 256)    
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image = np.asarray(image)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  img_resize = (cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST))/255.
  img_reshape = img_resize[np.newaxis,...]
  return img_reshape

file = st.file_uploader("Silahkan unggah gambar CT paru-paru Anda", type=["jpg", "png", "jpeg"])

if file is not None:
  content = Image.open(file)
  st.image(content, width=300)

  st.write("")
  st.subheader("Hasil Prediksi: ")
  with st.spinner("Classifying..."):
    prediction = model.predict(preprocess_img(content))
    label = np.argmax(prediction, axis=1)
    st.write(classes[label[0]])
    st.subheader('Nilai Probabilitas: ')
    st.write(prediction)
    st.text("Keterangan (0: COVID-19, 1: Non-COVID-19)")