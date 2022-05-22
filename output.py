from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
config1 = 'configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_1.py'
config2 = 'configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_2.py'
#权重文件
checkpoint1 = 'work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/swin_epoch_150_1.pth'
checkpoint2 = 'work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/swin_epoch_150_2.pth'
device = 'cuda:0'
img_path1 = 'test_images_final'
img_path2 = './'
img_dir = os.listdir(img_path1)
img =img_dir[3]
img_ = '00028.jpg'
score_thr =0.3
os.makedirs('./save_partition' ,exist_ok=True)
model1 = init_detector(config1, checkpoint1, device=device)
model2 = init_detector(config2, checkpoint2, device=device)
class_name1 = ('7','156','9','256','10','280','148')
class_name2 = ('228','25','6','485','673','222','401','392','402','235','115','398','480','8','310','105','387','394','110','41','430')
id1 = '1'
id2 = '2'
# colors for visualization
#保存预测位置信息
def partition_ouput(img_path,img_dir,model,class_name,id):
    with open('./save_partition/partition'+id +'.txt',mode = 'a+',encoding = 'utf-8') as f:
        for img in img_dir:
            img_ = os.path.join(img_path,img)
            result = inference_detector(model, img_)
            bbox_result, _ = result
            bboxes_ = np.vstack(bbox_result)
            box_len = bboxes_.shape[0]
            bboxes = []
            for i in range(box_len):
                if bboxes_[i][4] >= score_thr:
                    bboxes.append(bboxes_[i])
            bboxes = np.array(bboxes)
            for i in range(bboxes.shape[0]):
                if bboxes[i][4] < score_thr:
                    bboxes = np.delete(bboxes,i,0)
            for i in range(bboxes.shape[0]):
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                f.write(img +','+class_name[labels[i]]+','+str(bboxes[i][0]*4)+','+str(bboxes[i][1]*4)+','+str(bboxes[i][2]*4)+','+str(bboxes[i][3]*4)+','+str(bboxes[i][4])+'\n')
        f.close()
partition_ouput(img_path1,img_dir,model1,class_name1,id1)
partition_ouput(img_path1,img_dir,model2,class_name2,id2)
def classification(img_path,img_dir,model,class_name):
    with open('./save_partition/classification.txt', mode='a+', encoding='utf-8') as f:
        class_dict = {}
        for img in img_dir:
            img_ = os.path.join(img_path,img)
            result = inference_detector(model, img_)
            bbox_result, _ = result
            bboxes = np.vstack(bbox_result)
            for i in range(bboxes.shape[0]):
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                for i in labels:
                    class_dict[i] = class_dict.get(i,1)+1
        for key,value in  class_dict.items():
            f.write(class_name[key]+':'+str(value)+'\n')
        f.close()
#classification(img_path,img_dir,model)
#画出预测框
def draw_image1(img_path,img,model,class_name):
    box_color = (255,0,255)
    img_ = os.path.join(img_path, img)
    image = cv2.imread(img_)
    result = inference_detector(model, img_)
    bbox_result, _ = result
    bboxes_ = np.vstack(bbox_result)
    box_len = bboxes_.shape[0]
    bboxes = []
    for i in range(box_len):
        if bboxes_[i][4] >= score_thr:
            bboxes.append(bboxes_[i])
    bboxes = np.array(bboxes)
    for i in range(bboxes.shape[0]):
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
    for i in range(bboxes.shape[0]):
        x1,y1,x2,y2 = int(bboxes[i][0]),int(bboxes[i][1]),int(bboxes[i][2]),int(bboxes[i][3])
        cv2.rectangle(image, (x1,y1), (x2, y2), color=box_color, thickness=2)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    ax = plt.gca()
    for i in range(bboxes.shape[0]):
        text = f'{class_name[labels[i]]}: {bboxes[i][4]:0.2f}'
        x1, y1= int(bboxes[i][0]), int(bboxes[i][1])
        ax.text(x1+10, y1-20, text, fontdict=dict(fontsize=5, color='black',family='monospace'),
                bbox=dict(facecolor='r', alpha=0.3))
    plt.axis('off')
    plt.show()
def draw_image2(img_path1,img_path2,img,model,class_name):
    box_color = (255,0,255)
    img_1 = os.path.join(img_path1, img)
    img_2 = os.path.join(img_path2, img)
    image1 = cv2.imread(img_1)
    image2 = cv2.imread(img_2)
    result = inference_detector(model, img_1)
    bbox_result, _ = result
    bboxes_ = np.vstack(bbox_result)
    box_len = bboxes_.shape[0]
    bboxes = []
    for i in range(box_len):
        if bboxes_[i][4] >= score_thr:
            bboxes.append(bboxes_[i])
    bboxes = np.array(bboxes)
    for i in range(bboxes.shape[0]):
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
    for i in range(bboxes.shape[0]):
        x1,y1,x2,y2 = int(bboxes[i][0])*4,int(bboxes[i][1])*4,int(bboxes[i][2])*4,int(bboxes[i][3])*4
        cv2.rectangle(image2, (x1,y1), (x2, y2), color=box_color, thickness=5)
    image = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()
    for i in range(bboxes.shape[0]):
        text = f'{class_name[labels[i]]}: {bboxes[i][4]:0.2f}'
        x1, y1 = int(bboxes[i][0])*4, int(bboxes[i][1])*4
        ax.text(x1+10, y1-10, text, fontdict=dict(fontsize=5, color='black', family='monospace'),
                bbox=dict(facecolor='r', alpha=0.3, pad=1))
    plt.axis('off')
    plt.show()
#draw_image1(img_path1, img_,model2,class_name2)
#draw_image2(img_path1,img_path2, img_,model2,class_name2)
#show_result_pyplot(model, img, result)
#show_result(img_path1, img_)