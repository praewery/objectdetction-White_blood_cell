import streamlit as st  #Web App
from PIL import Image, ImageOps #Image Processing
import time
from unittest import result
import numpy as np
from icevision import tfms
from icevision.models.checkpoint import *
import easyocr as ocr  #OCR

def get_detection(img_path):
 
  #Get_Idcard_detail(file_path=img_path)
  img = Image.open(img_path)
  img = ImageOps.exif_transpose(img) # fix image rotating
  width, height = img.size # get img_input size
  if (width == 1280) and (height == 1280):
    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.5)
  else:
    #im = im.convert('L') #Convert to gray
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(1280)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (1280, 1280))
    new_im.paste(img, ((1280-new_size[0])//2,
                        (1280-new_size[1])//2))
    pred_dict  = model_type.end2end_detect(new_im, valid_tfms, model, class_map=class_map, detection_threshold=0.5)
    labels, acc = pred_dict['detection']['labels'], pred_dict['detection']['scores']


    try:
        labels, acc = pred_dict['detection']['labels'][0], pred_dict['detection']['scores'][0]
        acc = acc * 100
        if labels == "Neg":
            labels = "Negative"
        elif labels == "Pos":
            labels = "Positive"
        st.success(f"Result : {labels} with {round(acc, 2)}% confidence.")
    except IndexError:
        st.error("Not found WB  cell image! ; try to take image again..")
        labels = "None"
        acc = 0



def get_img_detection(img_path):#Get_Idcard_detail(file_path=img_path)
      img = Image.open(img_path)
      img = ImageOps.exif_transpose(img)
      width, height = img.size # get img_input size
      if (width == 1280) and (height == 1280):
        new_im = img
      else:
        old_size = img.size  # old_size[0] is in (width, height) format
        ratio = float(1280)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (1280, 1280))
        new_im.paste(img, ((1280-new_size[0])//2,
                        (1280-new_size[1])//2))
      pred_dict  = model_type.end2end_detect(new_im, valid_tfms, model, class_map=class_map, detection_threshold=0.5)
      return pred_dict['img']
 







#set default size as 1280 x 1280
def img_resize(input_path,img_size): # padding
  desired_size = img_size
  im = Image.open(input_path)
  im = ImageOps.exif_transpose(im) # fix image rotating
  width, height = im.size # get img_input size
  if (width == 1280) and (height == 1280):
    new_im = im
  else:
    #im = im.convert('L') #Convert to gray
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    
    

  return new_im

### Model
checkpoint_path = "./swin_s10.pt"

checkpoint_and_model = model_from_checkpoint(checkpoint_path, 
    model_name='mmdet.retinanet', 
    backbone_name='swin_s_p4_w7_fpn_1x_coco',
    img_size=384, 
    is_coco=False)

model_type = checkpoint_and_model["model_type"]
backbone = checkpoint_and_model["backbone"]
class_map = checkpoint_and_model["class_map"]
img_size = checkpoint_and_model["img_size"]
#model_type, backbone, class_map, img_size

model = checkpoint_and_model["model"]

device=next(model.parameters()).device

img_size = checkpoint_and_model["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])


########## Set up sidebar

st.sidebar.image("./logo.png")
st.sidebar.header("AI BUILDERS")
def load_image(image_file):
    img = Image.open(image_file)
    return img

########## Selectbox
activities = ["About", "Detection"]
choice = st.sidebar.selectbox("Select Activty",activities)

if choice =='About' :
    st.subheader("About")
    st.title("Objectdetection white bloodcell")
    st.write("ปัจจุบันเทคโนโลยีคอมพิวเตอร์ได้กลายเป็นสิ่งสำคัญที่เข้ามามีบทบาทกับชีวิตมนุษย์ในหลายๆด้าน รวมถึงด้านวิทยาศาสตร์เเละการเเพทย์ ได้มีการนำ Machine learning & AI เข้ามาเป็นส่วนนึงของการรักษา เพื่อพัฒนาให้การรักษามีประสิทธิภาพเเละ เเม่นยำมากยิ่งขึ้น !")
    st.write("Leukemia หรือ มะเร็งเม็ดเลือดขาวเป็นโรคมะเร็งที่เกิดขึ้นในไขกระดูก เกิดจากมีเซลล์เม็ดเลือดขาวตัวอ่อนเติบโตมาผิดปกติโดยไม่ทราบสาเหตุ การแบ่งตัวอย่างไม่หยุดของเซลล์เหล่านี้ ได้ไปรบกวนการสร้างเม็ดเลือดปกติชนิดอื่นของไขกระดูก ทำให้เม็ดเลือดแดง เม็ดเลือดขาวปกติ และเกล็ดเลือดลดลง ส่งผลให้ผู้ป่วยมีภาวะโลหิตจาง มีเลือดออกผิดปกติ มีจ้ำเลือดตามร่างกาย ติดเชื้อง่าย นอกจากนี้เซลล์มะเร็งยังสามารถไปสะสมตามอวัยวะอื่นๆ เช่น ตับ ม้าม ต่อมน้ำเหลือง ทำให้ผู้ป่วยมีต่อมน้ำเหลือง ตับ ม้ามโต")

    st.image('./leukemia2.jpg')
    st.write("มะเร็งเม็ดเลือดขาวชนิดเฉียบพลันจัดเป็นโรคมะเร็งที่มีความรุนแรงสูง พบได้ทุกเพศทุกวัย พบมากขึ้นในผู้สูงอายุ และเป็น 1 ใน 10 โรคมะเร็งที่พบบ่อยในประเทศไทย")
    
    
    st.write('มะเร็งเม็ดเลือดขาวสามารถแบ่งได้หลายแบบ ได้แก่ แบ่งตามระยะการเกิดโรค และแบ่งตามชนิดของเซลล์มะเร็ง')
    
    st.write('แบ่งตามระยะเวลาเกิด')
    
    st.write('1.มะเร็งเม็ดเลือดขาวชนิดเฉียบพลัน (acute leukemia) คือการที่เซลล์ตัวอ่อนของเม็ดเลือดขาวแบ่งตัวอย่างรวดเร็ว อาการของโรคจะเกิดขึ้นอย่างฉับพลันและรุนแรง ผู้ป่วยจึงจำเป็นต้องได้รับการรักษาอย่างทันท่วงที')
    st.write('2.มะเร็งเม็ดเลือดขาวชนิดเรื้อรัง (chronic leukemia) คือการที่เซลล์เม็ดเลือดขาวถูกผลิตออกมามากเกินไป ทำให้ผู้ป่วยมีเม็ดเลือดขาวมากกว่าปกติ เนื่องจากความผิดปกติเกิดขึ้นอย่างช้าๆ ผู้ป่วยอาจไม่มีอาการผิดปกติใดๆ เลยเป็นเวลานับปี แต่สามารถตรวจพบได้จากการตรวจเลือด')
    
    st.write('แบ่งตามชนิดของเซลล์มะเร็ง')
    
    st.write('1.มะเร็งเม็ดเลือดขาวชนิดมัยอีโลจีนัส (myelogenous leukemia) เป็นชนิดของมะเร็งที่เกิดจากเซลล์ในสาย myeloid เติบโตผิดปกติ')
    st.write('2.มะเร็งเม็ดเลือดขาวชนิดลิมโฟซิติก (lymphocytic leukemia) เป็นชนิดของมะเร็งที่เกิดจากเซลล์ในสาย lymphoid')
    
    st.subheader("ทั้งนี้ การแบ่งชนิดของมะเร็งเม็ดเลือดขาวจะมีผลต่อการเลือกวิธีการรักษา เนื่องจากมะเร็งเม็ดเลือดขาวแต่ละชนิดมีการดำเนินโรคและการพยากรณ์โรคที่แตกต่างกัน")
    
    st.write("การทำโปรเจคในครั้งนี้ผู้จัดทำได้ทำการเเบ่งประเภท white blood cell เป็น 12 ชนิดดังนี้")
    st.subheader("-Atypical lymphocyte")#1
    st.subheader("-Band Neutrophil")#2
    st.subheader("-Basophil")#3
    st.subheader("-Blast")#4
    st.subheader("-Eosinophil")#5
    st.subheader("-Lymphocyte")#6
    st.subheader("-Metamyelocyte")#7
    st.subheader("-Monocyte")#8
    st.subheader("-Myelocyte")#9
    st.subheader("-NRC")#10
    st.subheader("-Promyelocyte")#11
    st.subheader("-Segmented neutrophil")#12
    
    st.write("อย่างไรก็ตาม การนำ AI เข้ามาประยุกต์ใช้กับการเเพทย์เพื่อพัฒนาให้การรักษามีประสิทธิภาพมากยิ่งขึ้น เเม้ว่า AI จะสามารถทำนายได้อย่างเเม่นยำเเละรวดเร็วเเต่สิ่งสำคัญที่จะมองผ่านไม่ได้เลยนั่นก็คือความน่าเชื่อถือเเละประสบการณ์ของบุคลากรทางการเเพทย์ โดยมี AI เป็นฝ่ายสนับสนุนเพื่อประสิทธิภาพในการรักษาที่ดียิ่งขึ้น")
    
     
       
elif choice == "Detection":
    st.subheader("White blood cell Detection")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file is not None:
        # To See details
        image_file.seek(0)
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
        st.write(file_details)
        # To View Uploaded Image
        
        
        st.image(get_img_detection(image_file))
        get_detection(image_file)
        
        
    
                
              
    
 
     

   
   
st.sidebar.subheader('prepared by Praew ')
st.sidebar.subheader('[mysterious-hedgehods]')
st.sidebar.write('project นี้จัดทำขึ้นภายใต้การดูเเลของโครงการ AI Builders 2022 ที่ช่วยสนับสนุน เเนะนำ เเละสอนให้เราสามารถพัฒนา AI เพื่อเเก้ปัญหาเเละนำไปประยุกต์ใช้ในชีวิตจริงได้')
st.sidebar.markdown('---')






    








