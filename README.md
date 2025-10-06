# GAT CUDA Extension

PyTorch C++/CUDA扩展模块，用于图注意力网络(GAT)的高性能计算。

## 项目结构

```
GAT_cuda/
├── set_up.py              # python库构建脚本
├── test_gat_extension.py  # 在python中运行c++实现的gat相关函数
├── run.py                 # 测速脚本
├── read_files.py          # 用于读取cora数据集和citeseer数据集
├── README.md              # 项目说明
├── include/
│   ├── gat_layer.h        # GAT层头文件
│   └── utils.cuh          # CUDA工具函数头文件
├── src/
│   ├── gat_layer.cu       # GAT层实现
|   └── bindings.cpp       # python接口注册文件
```

## 功能特性

- **GAT层实现**：gat层的前向传播
- **CUDA优化**：使用共享内存优化的矩阵乘法核函数
- **导入pytorch**: 可以将c++实现的gat层在python的pytorch中使用

## 参考运行环境(作者运行环境)

- Ubuntu 24.04.1 LTS
- Python 3.10
- PyTorch 2.8.0+cu126
- CUDA 12.6
- C++17编译器

## 构建和运行

### 1.在项目根目录下使用指令运行`set_up.py`
```bash
python set_up.py build_ext --inplace
```
完成后将会在根目录下生成`.so`文件

### 2.运行`test_gat_extension.py`
```bash
python test_gat_extension.py
```
注：如果连接不到pytorch可以尝试在运行脚本前输入指令：
```bash
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```


### 2.运行`run.py`统计时间
