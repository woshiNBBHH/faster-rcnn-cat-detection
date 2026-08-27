# Cat Faster R-CNN

基于 PyTorch/Torchvision 的猫目标检测项目，模型为 Faster R-CNN + ResNet-50-FPN。

## PyCharm 设置

1. 使用 PyCharm 打开本目录。
2. 创建 Python 3.10 或 3.11 虚拟环境。
3. 在 PyCharm Terminal 执行：`pip install -r requirements.txt`。
4. 检查 `configs/` 中的数据路径和超参数。默认数据目录是项目内的 `datastes/`。

## 运行

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/exp002_lr_001.yaml
python train.py --config configs/exp003_batch_8.yaml
python train.py --config configs/exp004_freeze_backbone.yaml
python evaluate.py --config configs/baseline.yaml --checkpoint outputs/exp001_baseline/best.pt --split test
python predict.py /path/to/image.jpg --config configs/baseline.yaml --checkpoint outputs/exp001_baseline/best.pt
tensorboard --logdir outputs
```

每份配置使用独立的输出目录，训练过程会生成 `metrics.csv`、TensorBoard 日志、`last.pt` 和验证集 mAP 最优的 `best.pt`，不同实验不会相互覆盖。

`metrics.csv` 和 `training_curves.png` 会记录/展示训练损失、mAP、mAP@0.5、TP、FP、FN、Precision、Recall、过检率和漏检率。其中：

- 过检率 = FP / (TP + FP)
- 漏检率 = FN / (TP + FN)

执行测试集评估后，会在对应实验目录的 `evaluation_test/` 下生成 `metrics.json`、训练曲线和过检/漏检图片的“原标注—模型预测”对照图。

## 数据集可视化

```bash
python visualize_dataset.py
```

结果保存在 `outputs/dataset_visualization/`，包括训练/验证/测试数量、猫实例数量、图片尺寸分布、宽高比分布、分辨率统计 JSON 和标注样例图。

## 数据约定

代码使用项目内的 `datastes/{train,val,test}`。每个分组目录同时存放同名图片和标注，例如 `cat000001.jpg` 与 `cat000001.json`。程序按文件名配对并忽略 Labelme JSON 内的旧 `imagePath`。普通矩形直接使用；有向矩形和多边形会转换为水平外接框。
