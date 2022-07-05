import cv2
import glob
import numpy as np
import json
import os
import random

def mix_jsons(load_data_dir,save_data_dir,hv_flip=False):
    with open(load_data_dir, "r", encoding="UTF-8") as f:
        json_str = f.read()
    your_dict = json.loads(json_str) #json格式转python格式
    your_dict["version"] = "4.5.6"
    your_dict["imageData"]= None #python中None即是null
    your_dict["imagePath"]= os.path.basename(save_data_dir)[:-5]+'.jpg' #获取文件名
    if hv_flip ==True:
        imgH = your_dict["imageHeight"]
        imgW = your_dict["imageWidth"]        
        shapes = your_dict["shapes"]  #label保存在shapes属性里面      
        for label in range(len(shapes)): #label的总数
            for k in range(len(shapes[label]["points"])): #label中标签点的总数
                shapes[label]["points"][k][0] = imgW - shapes[label]["points"][k][0] #x坐标
                shapes[label]["points"][k][1] = imgH - shapes[label]["points"][k][1] #y坐标
    with open(save_data_dir, "w", encoding="UTF-8") as f1:
        json.dump(your_dict,f1, ensure_ascii=False,indent = 2)

#随机调节亮度和对比度
def contrast_brightness_image(img_dir, a, g):
    src1=cv2.imread(img_dir)
    h, w, ch = src1.shape#获取shape的数值，height和width、通道
    src2 = np.zeros([h, w, ch], src1.dtype)#(色素全为零，输出为全黑图片)
    src3 = np.ones([h, w, ch], src1.dtype)*255#(色素全为255，输出为全白图片)
    i = np.random.random_integers(2,3)
    if i == 2:
        dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    else:
        dst = cv2.addWeighted(src1, a, src3, 1-a, g)
    print(os.path.dirname(img_dir)+"/"+os.path.basename(img_dir)[:-5]+".json",'\n')
    jf = os.path.dirname(img_dir)+"/"+os.path.basename(img_dir)[:-5]+".json"
    jout = os.path.dirname(os.path.dirname(jf))+"/data_augment/"+"cb"+os.path.basename(jf)#[2:]
    print(jout)
    mix_jsons(jf,jout,hv_flip=False)
    print(jout[:-5]+'.jpg')
    cv2.imwrite(jout[:-5]+'.jpg',dst)
    return dst

def cutout(img_dir,num):
    src1 = cv2.imread(img_dir)
    h, w, ch = src1.shape
    for i in range(num):
        x1 = np.random.random_integers(100,750)
        y1 = np.random.random_integers(100,800)
        w = np.random.random_integers(10,60)
        h = np.random.random_integers(10,60)
        cv2.rectangle(src1,(x1, y1), (x1+w, y1+h), (128, 128, 128), -1)
    print(os.path.dirname(img_dir)+"/"+os.path.basename(img_dir)[:-5]+".json",'\n')
    jf = os.path.dirname(img_dir)+"/"+os.path.basename(img_dir)[:-5]+".json"
    jout = os.path.dirname(os.path.dirname(jf))+"/data_augment/"+"ct"+os.path.basename(jf)#[2:]
    print(jout)
    mix_jsons(jf,jout,hv_flip=False)
    print(jout[:-5]+'.jpg')
    cv2.imwrite(jout[:-5]+'.jpg',src1)
    return src1

def hv_flip(img_dir):
    src1 = cv2.imread(img_dir)
    src2 = cv2.flip(src1, -1)#1表示水平垂直翻折
    print(os.path.dirname(img_dir)+"/"+os.path.basename(img_dir)[:-5]+".json",'\n')
    jf = os.path.dirname(img_dir)+"/"+os.path.basename(img_dir)[:-5]+".json"
    jout = os.path.dirname(os.path.dirname(jf))+"/data_augment/"+"hf"+os.path.basename(jf)#[2:]
    print(jout)
    mix_jsons(jf,jout,hv_flip=True)
    print(jout[:-5]+'.jpg')
    cv2.imwrite(jout[:-5]+'.jpg',src2)
    return src2


def data_augment(img_dir):
    img=cv2.imread(img_dir)
    cv2.imshow("original",img)
    new1=contrast_brightness_image(img_dir,random.uniform(0.5,0.9),0)
    cv2.imshow("con-bri-demo", new1)
    new2=cutout(img_dir,num=10)
    cv2.imshow("cutout-demo",new2)
    new3=hv_flip(img_dir)
    cv2.imshow("hv-flip-demo",new3)


def main():
    for img_dir in glob.glob("G:/2020-09-27 Before exposure/1/1/"+"*.jpeg"):
        print(img_dir)
        data_augment(img_dir)
        cv2.waitKey(200)
        cv2.destroyAllWindows()
        print('========================================')



if __name__ =="__main__":
    main()