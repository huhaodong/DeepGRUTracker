# change log

---

|版本号|时间|说明|对应分支|
|------|------|------|------|
|v0.0.1|2018/11/25|完成有反馈模型|model_add_sequence|
|v0.0.2|2018/11/28|去除反馈机制。只利用GRU内部的特征保留机制。对应分支modify_model|modify_model|
|v1.0.1|2018/12/5|添加config模块,完成第一版本的模型|v1.0.1-completed|
|v2.0.1|2018/12/14|增加模型复杂度，模型学习残差|v2.0.1-residual_deep_GRU_tracker|
|v2.0.2|2018/12/15|去除tracker循环操作，为每层全连接层加入归一化操作|v2.0.2-no_result_circular_residual_deepGruTracker|