import cv2  
import numpy as np  
import time

#提示比赛开始
def countdown(seconds):
    while seconds > 0:
        print(seconds,end=" ")
        time.sleep(1)
        seconds -= 1
 #图片比较
def bijiaotupian(sherutupian):
    # 加载两张图片 
    img1 = cv2.imread(f'{sherutupian}')  
    #cv2.imshow("sherutupian",img1)
    #cv2.waitKey(0)  
    #cv2.destroyAllWindows()
    for i in range(1,4):
        img2 = cv2.imread(f'{i}.jpg')  
        #cv2.imshow(f"{i}",img2)
        #cv2.waitKey(0)  
        #cv2.destroyAllWindows()
        # 转换为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 提取SIFT特征  
        sift = cv2.xfeatures2d.SIFT_create()  
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)  
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)  
        
        # 计算特征向量之间的距离，判断是否匹配  
        #bf = cv2.BFMatcher() 
        #  # 创建FLANN匹配器
        flann = cv2.FlannBasedMatcher()

    # 使用KNN算法进行特征匹配
        matches = flann.knnMatch(descriptors1, descriptors2, k=2) 
      #  matches = bf.knnMatch(descriptors1, descriptors2, k=2)  
        
        # 过滤误匹配点  
        good_matches = []  
        for m, n in matches:  
            if m.distance < 0.8 * n.distance:  
                good_matches.append(m)  
                
        # 判断图片相似性  
        if len(good_matches) >30:  
            #print("两张图片相似")  
            return int(i)
        i=i+1
    return 0

def yucomputerbiiao(duishou_number):
    import random
    ai_number=random.randint(1,3)
    if ai_number==1:
        print("电脑出布")
    elif ai_number==2:
        print("电脑出剪刀")
    else:
         print("电脑出石头")
    user_number=duishou_number
    if ai_number==user_number:
        img3 = cv2.imread('pj.jpg')
        cv2.imshow("pj.jpg",img3)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
      # print("平局")
    else:
        if (user_number>ai_number) and (not (user_number==3 and ai_number==1)):
            img4=cv2.imread('shengli.jpg')
            cv2.imshow("shengli.jpg",img4)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
     # print("你胜利了")
        elif(user_number==1) and (ai_number==3):
             img4=cv2.imread('shengli.jpg')
             cv2.imshow("shengli.jpg",img4)
             cv2.waitKey(0)  
             cv2.destroyAllWindows()
     # print("你胜利了")
        else:
            img5=cv2.imread('loss.jpg')
            cv2.imshow("loss.jpg",img5)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
      #      print("你输了")
        
countdown(2)
#从摄像头截取一张照片
# 打开摄像头设备，参数0表示使用默认摄像头
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#cv2.CAP_DSHOW解决了启动摄像头很慢的问题   在cmd环境下运行 setx OPENCV_VIDEOIO_PRIORITY_MSMF 0
# 检查是否成功打开摄像头
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
# 循环读取摄像头的视频流
while True:
    # 从摄像头读取一帧图像
    ret, frame = cap.read()
    # 检查是否成功读取图像
    if not ret:
        break
    # 显示图像
    cv2.imshow("Camera", frame)
    #保存图片
      
   #timestamp=time.strftime("%Y%m%d-%H%M%S")
    #cv2.imwrite(f'capture-{timestamp}.jpg',frame)
    cv2.imwrite("duishou.jpg",frame)  
    #print("Image'对手.jpg'save.")  
    #C:/Users/Administrator/Pictures/picture/
    # 等待按下q键退出循环
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    #time.sleep(5)
# 释放摄像头设备并关闭窗口
cap.release()
cv2.destroyAllWindows()

f='duishou.jpg'
w=bijiaotupian(f)
print(w)
if w==0:
    print("图像没法识别，请重新开始")
    exit()
else:
    yucomputerbiiao(w)

