# Yolo记录（2024/8/28）
Yolo模型的训练、部署与测试.

(训练参考[Ultralytics](https://docs.ultralytics.com/zh)用的yolov8模型、数据集widerperson和VOC,部署与测试在lubancat1(rk3566)上进行，采用了python与c++推理测试)

用到的几个库，把当时的版本保存在了谷歌网盘，万一有更新，可以下载参考

[ultralytics_yolov8](https://drive.google.com/file/d/1ccteWOeOiJjrcKGMi2es5P_StVqmI6_B/view?usp=drive_link)

[rknn-toolkit2](https://drive.google.com/file/d/186N91rykxVWSbg6AHgMm-iWSNUxuMZFL/view?usp=drive_link)

[rknn-toolkit2](https://drive.google.com/file/d/1sawPAAHRQuTVY86luOmkiL3NK97d7e2P/view?usp=drive_link)

## Training
参考官网教程:(https://docs.ultralytics.com/zh/quickstart/)  如有变化，以官方教程为准。

（1） 环境
```
# 创建激活环境。
conda create --name yolo
conda activate yolo
```
```
# 后续环境删除（如需要）。
conda env remove --name yolo
```
```
# 安装官方库和并拉取官方代码。
pip install ultralytics
git clone https://github.com/ultralytics/ultralytics
# 安装对应依赖。官网代码好像删除了requirements.txt文件，后续可根据教程进行环境配置，requirements.txt文件也将在本项目中提供。
pip install -r requirements.txt
```
（2） 数据集处理（两种）

一种为官方已经配置好的、内置的数据集。具体含有哪些数据集及相关的调用下载训练可参考：(https://docs.ultralytics.com/datasets/)  下面以VOC数据集为例
```
# 在模型训练的代码中，将data传入的配置文件改为对应的yaml文件
results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
```
另一种为官方中未配置的数据集，则需要自己去搜集并进行处理，在此以WiderPerson数据集为例，需要将数据集处理为对应的yolo格式（WiderPerson数据集处理代码放在项目中）
```
WiderPerson/
        |->images/
        |    |->train/0001.jpg
        |    |->val/0001.jpg
        |    |->test/0001.jpg
        |->labels/
        |    |->train/0001.txt
        |    |->val/0001.txt
        |    |->train.cache
        |    |->val.cache 
        |->WiderPerson.yaml
```
需要根据数据集的特点（类别数等），来调整WiderPerson的配置文件（WiderPerson.yaml），然后把data传入的改为对应配置文件地址
```
results = model.train(data="/home/deeplearn/JupyterlabRoot/erdongsanshi/yolo/datasets/WiderPerson/WiderPerson.yaml", epochs=100, imgsz=640)
```
（3） 训练

配置完后根据官方提供的代码进行训练，可以更换不同的模型，具体也参考官网教程(https://docs.ultralytics.com/models/) 我是通过写入.py文件然后python xxx.py进行训练的。
```
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```
训练还有其他许多参数可以进行调整，比如patience耐心值、wandb可视化等，具体看教程(https://docs.ultralytics.com/modes/train/#train-settings) yolo的指标主要看mAP50、mAP50-95及Confidence

训练完成后，在生成的文件夹中的子文件夹weights中找到对应的best.pt模型文件，即完成yolo及对应的数据集训练。

## Deployment and Inference
这边建议部署再创建一个环境（如上），因为不同的代码库所需的环境依赖有冲突。

首先是接下来会用到的代码库（按使用顺序排列，记得git clone一下）：

（1）https://github.com/airockchip/ultralytics_yolov8.git

（2）https://github.com/airockchip/rknn-toolkit2.git

（3）https://github.com/airockchip/rknn_model_zoo.git


因为我的部署目标开发板为rk3566，所以需要把训练得到的pt模型文件转换为对应的rknn模型文件，常见的转换方式有两种：

1：pt——>onnx——>rknn        
2：pt——>torchscript——>rknn。        
按照教程及其他博主的方案来看，以第一种为多。

但这里针对rk系列的部署设备来说，有一个陷阱：
```
model = YOLO("/home/deeplearn/JupyterlabRoot/erdongsanshi/yolo/yolov8n_coco.pt")  # load a custom trained model
model.export(format="onnx")
```
根据ultralytics提供的教程，它可以直接用上面的代码生成onnx文件。然后随之，我们就会用rockchip给的教程将onnx转换为rknn模型，再进行部署。陷阱就是上面的这些转换流程估计都会很顺利，可在rockchip官方例程中就会测试失败，甚至在板子上推理失败。原因是，针对yolov8模型来说，官方给出的模型最后的输出格式为[1,84,8400]，但这种输出格式不能满足rockchip系列教程中给出的后处理代码，然后导致无法测试成功或者推理成功（事实上rockchip工作人员针对官方给出的yolov8输出头做了修改，来更好的适配RKNPU以及应对量化和其他的优化），简单的说就是模型与代码后处理的代码不适配，需要对模型的输出头进行一定的修改，具体的修改详见（https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8）

所以若直接使用官方的onnx模型转换，会导致接下来在测试和推理失败，所以需要导出RKNPU适配模型，具体教程rockchip工作人员已公开，详见(https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.zh-CN.md)。
```
# 使用之前记得安装好https://github.com/airockchip/ultralytics_yolov8所需依赖
# 调整 ./ultralytics/cfg/default.yaml 中 model 文件路径，默认为 yolov8n.pt，若自己训练模型，请调接至对应的路径。支持检测、分割模型。
# 如填入 yolov8n.pt 导出检测模型
# 如填入 yolov8-seg.pt 导出分割模型

# 进入到~/JupyterlabRoot/erdongsanshi/ultralytics_yolov8目录然后进行终端运行指令
export PYTHONPATH=./  # 将当前目录添加到Python的搜索路径中。这样做的目的是为了让Python能够在当前目录下找到ultralytics模块或其他被引用的脚本文件。避免和环境中安装的ultralytics冲突
python ./ultralytics/engine/exporter.py

# 执行完毕后，会生成 ONNX 模型. 假如原始模型为 yolov8n.pt，则生成 yolov8n.onnx 模型。
```
根据上面的代码，就能顺利生成三个输出的onnx模型用于rockchip教程中的测试及推理代码。

得到onnx模型后，就需要再将onnx模型转换为rknn格式的模型来进行部署。首先先介绍rknn的开发环境，分为Toolkit2、Toolkit Lite2、RKNPU2。对于相应的rk系列官方教程(https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/env/env.html)

以我的理解是。Toolkit2运行在PC平台，并有模型转换、量化功能、模型推理等功能。（重点：PC端）        Toolkit Lite2则是Rockchip NPU平台的编程接口(Python)，用于在板端部署RKNN模型。        RKNPU2则是平台的编程C接口

首先是用Toolkit2进行模型转换，需要先配置PC上的环境，教程为(https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/env/toolkit2.html) ，唯一需要注意的是官方教程中的whl文件为1.5版本的，以及过旧，在后续对yolov8进行转换推理会出现报错，所以建议安装最新的whl文件具体可在该地址进行下载(https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages)

安装完后就可以进行模型转换和在PC上进行模拟推理，用(https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8/python) 这里的转换代码，在教程中，是转换rknn模型的代码后直接进行模拟推理的，但代码库提供的只是进行模型转换，所以我进行了代码修改，在进行模型转换后直接进行推理进行验证。代码放在该项目中。
```
# 若想更改测试的图片记得改convert.py文件中的DATASET_PATH和IMG_PATH，CLASSES改为训练模型的检测类别，DEFAULT_RKNN_PATH为转换rknn模型后，模型生成保存的地方。
# 用法为
python convert.py XXX.onnx rk3566
```
之后即可生成对应的rknn模型。并可以进行后续得部署。

本人部署在lubancat-1开发板中，然后对应的开发板使用教程可参考(https://doc.embedfire.com/linux/rk356x/quick_start/zh/latest/README.html) 在本测试中，使用的是ubuntu20.04。对于板端除了终端指令，就是文件传输问题，可以使用开发板教程，本人也用过git bash对板端进行文件传输。这里不多做介绍。

***************************** 基于python接口的部署 *****************************
```
# 将对应系统烧录后，可以做一些关于python的基础操作，详见(https://doc.embedfire.com/linux/rk356x/Python/zh/latest/base/brief.html) ，并创建一个python虚拟环境，来安装一些依赖
python3 -m venv .toolkit2_env  # 创建python虚拟环境，和conda的那个类似
# 激活进入环境
source .toolkit2_env/bin/activate

# 然后安装根据板端的python版本下载对应的rknn库Toolkit Lite2，这里一样的，需要下载最新版本的toolkit-lite2(https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages) 以防后面出现版本兼容的问题
pip3 install packages/rknn_toolkit2-1.4.0_22dcfef4-cp38-cp38-linux_x86_64.whl # 该whl文件在上面的路径里，其他需要的库，自己下载

# 然后还有一个问题就是：板端推理测试，会调用librknnrt.so库，该库是一个板端的runtime库。默认板卡镜像/usr/lib目录下有librknnrt.so库，默认的库版本可能是1.5.0，这个版本也是不够的，可以下载最新的上传到板端，将对应位置的librknnrt.so进行替换。新版位置(https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64) 后续有更新，可以以官网最新为主，本项目中也会保存项目使用的版本。

# 上面的完成后就可以把在pc上转换的rknn模型文件、推理代码test.py以及测试推理的图片传输到板端。在激活对应的python环境后，检查下test.py中的文件加载路径和保存路径，就可以在板端进行推理，在比较pc端的convert.py和test.py时，可以发现主要就是把toolkit库改用toolkit-lite2
python test.py
```
至此，基于python的推理成功。

***************************** 基于C++接口的部署 *****************************
关于C++的部署官方教程比较散，还是用到上面的rknn_model_zoo库，然后可以按照以下两个教程来(https://github.com/airockchip/rknn_model_zoo/blob/main/docs/Compilation_Environment_Setup_Guide.md#linux-platform) (https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8)

```
具体的，首先是需要下载安装交叉编译器，来编译Cplusplus，这里不做详述

之后用到rknn_model_zoo/example/yolov8/cpp/，如果自己有训练模型的，记得把生成出来的rknn模型放在yolov8/model里面，然后要改cpp/postprocess.h和cpp/postprocess.cc，第一个文件是改OBJ_CLASS_NUM，NMS_THRESH还有BOX_THRESH。第二个文件改LABEL_NALE_TXT_PATH（可自建）

然后按照第一个链接给的教程

# go to the rknn_model_zoo root directory
cd <rknn_model_zoo_root_path> 切到rknn_model_zoo库的路径

# if GCC_COMPILER not found while building, please set GCC_COMPILER path
export GCC_COMPILER=<GCC_COMPILER_PATH> 安装的编译器路径

# 格式：./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d <model_name>

# for RK3566
./build-linux.sh -t rk3566 -a aarch64 -d yolov8

运行后生成build和install文件夹，然后可以把install文件夹中的文件推送到板子上，就可以编译推理（我直接把rknn_model_zoo/install/rk356x_linux_aarch64/rknn_yolov8_demo进行压缩然后传到了板子上，至于板子上要不要交叉编译器我忘记了，本人对c++也不是很熟）
```



******************* 后续 ***********************

后续有时间会对具体的一些细节进行完善更新，并尝试加速。若没时间，也希望后面看到此记录的人进行更多的尝试。

