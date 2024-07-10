## ram_experiment
万物识别实验管理


### 目录结构

    competition
    ├─data：试验数据
        ├─clean_data_5037：5000类数据,数据清洗后，修复58个格式异常文件，删除一个带有子目录的异常文件夹
        ├─clean_data_5037_10：基于clean_data_5037，裁剪10%的边
        ├─clean_data_5037_correct_1：基于clean_data_5037，将所有预测正确的图像复制到新目录（缺点：预测正确的图像可能会因为作为真值的时候预测错误而被删除，导致依然存在错误数据）
        ├─clean_data_5037_correct_2：基于clean_data_5037_correct_1，copy所有文件夹数量大于2的文件夹到新目录
        ├─clean_data_5037_correct_3：基于clean_data_5037_correct_2，copy所有文件夹数量大于3的文件夹到新目录
        ├─clean_data_5037_correct_4：基于clean_data_5037，移动所有预测错误的图像到新目录（这次我们使用的是错误目录）
        ├─clean_data_5037：5000类数据
        ├─新零售图片数据_Trax_部分：35类数据
        ├─新零售图片数据_Trax_部分_corrcet：基于35类数据，将所有预测正确的图像复制到新目录
    ├─model：
        ├─clip_model：clip模型
        ├─faiss_model：faiss模型
    ├─output：
        ├─faiss_model：在实验中生成的不同的faiss模型
            ├─35_category_name：35类数据，保存类别名称对应向量（用于正常集成yolo分类器之后）
            ├─35_image_path：35类数据，保存图片路径名称对应向量（用于可视化分析预测错误问题，预测结果分析保存此目录下）
            ├─5037_category_name：5037类数据，保存类别名称对应向量（用于正常集成yolo分类器之后）
            ├─clean_data_5037：5037类数据，保存图片路径名称对应向量（用于可视化分析预测错误问题，预测结果分析保存此目录下）
            ├─clean_data_5037_correct_1：基于clean_data_5037，将所有预测正确的图像复制到新目录
            ├─clean_data_5037_correct_3：基于clean_data_5037_correct_2，copy所有文件夹数量大于3的文件夹到新目录
    ├─1_clip_count_image_text_similarity：计算clip图像文本相似度
    ├─2_clip_text_to_image_multi_category：计算clip用文字查询图片多类区分度
    ├─3_clip_image_to_text：图片查询文字，传入图片利用文字解码器解读出文字
    ├─5_1_clip_and_faiss_New_Retail_Categories：利用clip将图像解码成向量，然后向量查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后faiss索引查询faiss索引，然后
    ├─5_3_visual_error：基于预测错误的日志，按照左边真值图、右边预测值图，拼接出来
    ├─5_4_error_picture_name：输出预测错误的日志
    ├─5_5_correct_picture_name：输出预测正确的日志
    ├─5_6_visual_correct：基于预测正确的日志，按照左边真值图、右边预测值图，拼接出来
    ├─7_create_vector_library：创建向量库
    ├─8_save_clip_model：保存clip模型
    ├─9_crop_images：裁掉边，剪切成小图
    ├─10_data_clean：清洗5000类的数据
    ├─11_0_copy_pic_from_mulit_pic：将图像类别文件夹中存在多张图像复制到新目录
    ├─11_copy_correct_pic：将正确预测的图像复制到新目录
    
