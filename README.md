# imageCompared
图片印记识别对比


1、印记种类：原图，旋转180°，上下镜像，左右镜像，曝光加深，曝光减轻

2、边界判断方法：
    
    # 阈值分割
    # thresholdSegment()

    # 边界分割（边缘检测）
    edgeSegmentation()

    # 区域分割（区域生成）
    # regionSegmentation()

    # SVM分割（支持向量机）
    # svmSegment()

    # 分水岭分割
    # watershedSegment()

    # Kmeans分割
    # kmeansSegment()
    
3、结果对比：略，最终使用edgeSegmentation（边界分割），副作用是曝光相关处理未解决
    ![image](https://user-images.githubusercontent.com/72720742/169765790-28affe49-ba16-43c2-ade7-ab762cc6448f.png)


4、对分割好的印记进行处理并对比：
    
    对于180度旋转后的印记，五种对比方法结果如下：
    
    均值哈希算法相似度： 0.76
    差值哈希算法相似度： 0.55
    感知哈希算法相似度： 0.77
    三直方图算法相似度： 1.0
    单通道的直方图算法相似度： 1.0
