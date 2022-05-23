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
