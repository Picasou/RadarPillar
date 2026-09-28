#pragma once
#include <memory>
#include <string>

// 推理后端抽象: PC=ONNX Runtime / 板端=VART graph_runner(同接口, 编译期选择)
class DpuBackend {
public:
    virtual ~DpuBackend() = default;
    // spatial [1,C,ny,nx] 行主序 → logits [1,11,H,W] 行主序
    virtual void infer(const float* spatial, float* logits) = 0;
    virtual const char* name() const = 0;
};

std::unique_ptr<DpuBackend> create_dpu_backend(const std::string& model_path);
