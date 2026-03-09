import onnx


def count_onnx_params(model_path):
    model = onnx.load(model_path)
    total_params = 0

    for tensor in model.graph.initializer:  # initializer存储所有权重参数
        param_count = 1
        for dim in tensor.dims:
            param_count *= dim
        total_params += param_count

    print(f"Total parameters: {total_params:,}")
    return total_params


# 使用示例
count_onnx_params("your_model.onnx")