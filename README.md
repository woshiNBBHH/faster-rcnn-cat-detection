# Cat and Dog Faster R-CNN

基于 PyTorch/Torchvision 的猫/狗目标检测项目，模型为 Faster R-CNN + ResNet-50-FPN。原来的猫单类别配置继续保留。

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

## 猫狗数据集自动预标注

新数据集使用独立目录 `cat_dog_dataset_53337/`，不会与原来的 `datastes/` 混合。

先使用 COCO 预训练 Faster R-CNN 生成猫、狗两类 Labelme 矩形预标注：

```bash
python auto_label_labelme.py \
  --image-dir /path/to/cat_dog_detection_datasets/images \
  --output-dir /path/to/cat_dog_detection_datasets/annotations_labelme \
  --score-threshold 0.8
```

自动标注完成并人工抽检后，按 8:1:1 建立独立训练集：

```bash
python prepare_cat_dog_dataset.py \
  --image-dir /path/to/cat_dog_detection_datasets/images \
  --annotation-dir /path/to/cat_dog_detection_datasets/annotations_labelme \
  --output-root cat_dog_dataset_53337 \
  --mode hardlink
```

训练猫狗双类别模型：

```bash
python train.py --config configs/cat_dog_baseline.yaml
```

### Kaggle 双 T4 训练

`train.py` 同时兼容单卡和 PyTorch DistributedDataParallel（DDP）。Kaggle
选择 `GPU T4 x2` 后，使用下面的命令启动两个训练进程：

```bash
torchrun --standalone --nproc_per_node=2 \
  train.py --config configs/cat_dog_baseline.yaml
```

配置中的 `training.batch_size` 是**每张 GPU**的批量大小。因此双卡时：

```text
global_batch_size = training.batch_size × 2
```

例如需要保持原单卡全局 batch size 4 时，将配置写成 `batch_size: 2`；若配置
仍为 `batch_size: 4`，双卡全局 batch size 将变成 8。`data.num_workers` 同样是
每个进程的 worker 数，Kaggle 只有 4 个 CPU 核时建议设为 2。

只有 rank 0 会执行完整验证、写 TensorBoard/CSV、绘图和保存检查点。保存的
`best.pt`/`last.pt` 不带 `module.` 前缀，可直接用于现有单卡 `evaluate.py` 和
`predict.py`。

从旧 V1 权重继续微调时，在配置中增加：

```yaml
training:
  initial_checkpoint: /kaggle/input/.../cat_dog_v1_best.pt
```

该选项只加载模型权重并从新的 epoch 1 开始，不恢复旧优化器和学习率状态。

自动预标注不是人工真值。正式训练前应抽检漏标、错标、猫狗类别混淆及重复框。

每份配置使用独立的输出目录，训练过程会生成 `metrics.csv`、TensorBoard 日志、`last.pt` 和验证集 mAP 最优的 `best.pt`，不同实验不会相互覆盖。

`metrics.csv` 和 `training_curves.png` 会记录/展示训练损失、mAP、mAP@0.5、TP、FP、FN、Precision、Recall、过检率和漏检率。其中：

- 过检率 = FP / (TP + FP)
- 漏检率 = FN / (TP + FN)

执行测试集评估后，会在对应实验目录的 `evaluation_test/` 下生成 `metrics.json`、训练曲线和过检/漏检图片的“原标注—模型预测”对照图。

## 第一版猫狗脸两级检测

第一版流水线先使用已有猫狗模型定位身体，再在身体区域内检测
`cat_face` / `dog_face`。ArcFace 身份识别将在脸部检测数据稳定后接入。

### 1. 抽取首批标注图片

下面的命令按猫狗个体均衡抽取各 500 张，每个个体最多 5 张，避免标注集被少数长视频占满：

```bash
python sample_face_annotation_images.py \
  --image-dir /path/to/03_siftimage \
  --output-dir face_annotation_batch_v1 \
  --cat-count 500 \
  --dog-count 500 \
  --max-per-identity 5
```

使用 LabelMe 只画矩形框，标签必须是 `cat_face` 或 `dog_face`。框应包含眼睛、
鼻子、嘴和主要脸部轮廓；耳朵可以部分位于框外。无清晰脸部的图片保存空 JSON，
作为负样本。`annotation_manifest.csv` 用于记录人工复核状态。

### 2. 验证标注并划分数据

```bash
python prepare_face_dataset.py \
  --image-dir face_annotation_batch_v1 \
  --annotation-dir face_annotation_batch_v1 \
  --output-root cat_dog_face_dataset \
  --mode hardlink
```

脚本会检查标签名称，并按动物个体进行互斥的 8:1:1 划分，避免同一只动物的相邻
视频帧同时进入训练集和测试集。包含猫狗同框的图片可增加
`--allow-mixed-species`。

### 3. 训练脸部检测模型

```bash
python train.py --config configs/cat_dog_face_baseline.yaml
```

Kaggle 双 T4：

```bash
torchrun --standalone --nproc_per_node=2 \
  train.py --config configs/cat_dog_face_baseline.yaml
```

### 4. 两级推理并导出脸部裁剪图

```bash
python cascade_predict.py /path/to/image.jpg \
  --body-config configs/cat_dog_baseline.yaml \
  --body-checkpoint /path/to/cat_dog_best.pt \
  --face-config configs/cat_dog_face_baseline.yaml \
  --face-checkpoint outputs/cat_dog_face_baseline/best.pt \
  --face-crop-dir outputs/face_crops \
  --output outputs/cascade_prediction.jpg
```

默认只保留身体类别与脸部类别一致的结果，例如 `cat` 身体内只接受
`cat_face`。脸部裁剪图随后用于制作 ArcFace 个体识别数据集。

## 数据集可视化

```bash
python visualize_dataset.py
```

结果保存在 `outputs/dataset_visualization/`，包括训练/验证/测试数量、猫实例数量、图片尺寸分布、宽高比分布、分辨率统计 JSON 和标注样例图。

## 数据约定

代码使用项目内的 `datastes/{train,val,test}`。每个分组目录同时存放同名图片和标注，例如 `cat000001.jpg` 与 `cat000001.json`。程序按文件名配对并忽略 Labelme JSON 内的旧 `imagePath`。普通矩形直接使用；有向矩形和多边形会转换为水平外接框。
# Initial 4,000-image cat/dog baseline

Create the fixed source-balanced manifest (1,000 images from each of COCO,
LVIS, Open Images, and VOC; seed 42):

```bash
python sample_cat_dog_initial_4000.py \
  --image-dir /path/to/cat_dog_detection_datasets/images \
  --output-root cat_dog_initial_4000 \
  --mode manifest
```

On the training machine, use `--mode symlink` to materialize the selected
images without copying their contents. Generate and manually review Labelme
annotations before using `configs/cat_dog_baseline_4000.yaml` for training.
