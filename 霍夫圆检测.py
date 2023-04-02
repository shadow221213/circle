import cv2
import matplotlib.pyplot as plt

# 初始化
img = []
img_gray = []
img_gas = []
img_cay = []
img_dst = []
circles_cay = []
circles_dst = []
true_flag = []

# 常用常量
lowThreshold = 360  # 最小阈值
ratio = 3  # 倍率
mask_set = 5  # 掩模大小
cap_cnt = 12  # 分析图片数量

for i in range(cap_cnt):
    image = cv2.imread("./test_img/img" + str(i) + ".jpg")
    # 变换图片大小
    (height, width) = image.shape[:2]
    new_width = width * ratio
    new_height = height * ratio
    img.append(cv2.resize(image, (new_width, new_height)))
    
    # 转换为灰度图像
    img_gray.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY))
    # 高斯滤波
    img_gas.append(cv2.GaussianBlur(img_gray[i], (mask_set, mask_set), 0))
    # 边缘检测
    img_cay.append(
            cv2.Canny(img_gas[i], lowThreshold, lowThreshold * (ratio + 2), apertureSize = mask_set,
                      L2gradient = True))
    # 中值滤波
    img_dst.append(cv2.medianBlur(img_gray[i], mask_set))
    # 霍夫圆检测
    circles_cay.append(cv2.HoughCircles(img_cay[i], cv2.HOUGH_GRADIENT, 1, 50, param1 = 100, param2 = 30,
                                        minRadius = 10 * ratio, maxRadius = 15 * ratio))
    circles_dst.append(cv2.HoughCircles(img_dst[i], cv2.HOUGH_GRADIENT, 1, 50, param1 = 100, param2 = 30,
                                        minRadius = 10 * ratio, maxRadius = 15 * ratio))

# 判断是否检测到圆
for i in range(cap_cnt):
    if (circles_dst[i] is not None) or (circles_cay[i] is not None):
        true_flag.append(i)
    else:
        # 显示图像
        plt.subplot(121)
        plt.imshow(img[i], cmap = 'gray')
        plt.title('None'), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(img_cay[i], cmap = 'gray')
        plt.title('canny'), plt.xticks([]), plt.yticks([])
        plt.show( )

if len(true_flag) == 0:
    print("没有检测到圆")
else:
    print("检测到圆的图片有", len(true_flag))
    print("检测到圆的图片编号为：", true_flag)
    # 将检测结果绘制在图像上
    for i in true_flag:
        X_cay = []
        Y_cay = []
        R_cay = []
        # 红色为高斯canny检测
        if circles_cay[i] is not None:
            cnt = 0
            for circle in circles_cay[i][0, :]:  # 遍历矩阵的每一行的数据
                X_cay.append(circle[0])
                Y_cay.append(circle[1])
                R_cay.append(circle[2])
                # 绘制圆形
                cv2.circle(img[i], (int(X_cay[cnt]), int(Y_cay[cnt])), int(R_cay[cnt]), (255, 0, 0), 15)
                # 绘制圆心
                cv2.circle(img[i], (int(X_cay[cnt]), int(Y_cay[cnt])), 1, (255, 0, 0), -1)
                cnt += 1
        
        X_dst = []
        Y_dst = []
        R_dst = []
        # 绿色为中值滤波检测
        if circles_dst[i] is not None:
            cnt = 0
            for circle in circles_dst[i][0, :]:  # 遍历矩阵的每一行的数据
                X_dst.append(circle[0])
                Y_dst.append(circle[1])
                R_dst.append(circle[2])
                # 绘制圆形
                cv2.circle(img[i], (int(X_dst[cnt]), int(Y_dst[cnt])), int(R_dst[cnt]), (0, 255, 0), 10)
                # 绘制圆心
                cv2.circle(img[i], (int(X_dst[cnt]), int(Y_dst[cnt])), 1, (0, 255, 0), -1)
                cnt += 1
        
        X_rel = []
        Y_rel = []
        R_rel = []
        # 判断两种检测结果是否一致
        for j in range(len(X_dst)):
            flag_rel = False
            for k in range(len(X_cay)):
                if abs(X_cay[k] - X_dst[j]) <= 3 and abs(Y_cay[k] - Y_dst[j]) <= 3:
                    X_rel.append((X_cay[k] + X_dst[j]) / 2)
                    Y_rel.append((Y_cay[k] + Y_dst[j]) / 2)
                    R_rel.append((R_cay[k] + R_dst[j]) / 2)
                    flag_rel = True
                    break
            if flag_rel == False:
                X_rel.append(X_dst[j])
                Y_rel.append(Y_dst[j])
                R_rel.append(R_dst[j])
        
        print("cay:", X_cay, Y_cay, R_cay)
        print("dst:", X_dst, Y_dst, R_dst)
        print("rel:", X_rel, Y_rel, R_rel)
        print("")
        
        # 蓝色为两种检测结果的交集
        for j in range(len(X_rel)):
            # 绘制圆形
            cv2.circle(img[i], (int(X_rel[j]), int(Y_rel[j])), int(R_rel[j]), (0, 0, 255), 5)
            # 绘制圆心
            cv2.circle(img[i], (int(X_rel[j]), int(Y_rel[j])), 1, (0, 0, 255), -1)
        
        # 显示图像
        plt.subplot(121)
        plt.imshow(img[i], cmap = 'gray')
        plt.title('img'), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(img_cay[i], cmap = 'gray')
        plt.title('canny'), plt.xticks([]), plt.yticks([])
        plt.show( )