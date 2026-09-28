#include "dpu_backend.h"

#include <stdexcept>

#if defined(USE_VART)
// 板端真 runner(VART GraphRunner, compiled xmodel): 搬 test_vart_sim.cpp 已跑通的
// find_dpu → create_runner("run") → fix↔float 转换逻辑; PC 侧无法编译验证, 依赖 T5 SDK 交叉编译
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner.hpp"
#include "xir/graph/graph.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/tensor/tensor.hpp"

namespace {
class VartBoardBackend : public DpuBackend {
public:
    explicit VartBoardBackend(const std::string& xmodel_path) {
        graph_ = xir::Graph::deserialize(xmodel_path);
        xir::Subgraph* dpu = find_dpu(graph_->get_root_subgraph());
        if (!dpu) throw std::runtime_error("xmodel 无 DPU 子图(非 compiled 形态?)");
        runner_ = std::unique_ptr<vart::Runner>(vart::Runner::create_runner(dpu, "run"));
        in_t_ = runner_->get_input_tensors()[0];
        out_t_ = runner_->get_output_tensors()[0];
        auto ish = in_t_->get_shape(), osh = out_t_->get_shape();   // NHWC [1,H,W,C]
        if (ish.size() != 4 || osh.size() != 4)
            throw std::runtime_error("xmodel IO shape 与预期不符(须 4 维 NHWC)");
        // 通道数运行时读取(前段 vfe_out_dim / 后段随 head 布局: 11=2D trunk, 13=vel, 12+=3D head)
        H_ = ish[1]; W_ = ish[2]; C_ = ish[3]; OH_ = osh[1]; OW_ = osh[2]; OC_ = osh[3];
        in_fix_ = in_t_->get_attr<int>("fix_point");
        out_fix_ = out_t_->get_attr<int>("fix_point");
        in_buf_ = std::make_unique<vart::mm::HostFlatTensorBuffer>(in_t_);
        out_buf_ = std::make_unique<vart::mm::HostFlatTensorBuffer>(out_t_);
    }

    void infer(const float* spatial, float* logits) override {
        /*
        板端前向: CHW float → 量化(×2^fix, int8 饱和) → NHWC → runner → int8 NHWC → 反量化 → CHW
        */
        const size_t in_n = (size_t)H_ * W_ * C_;
        const float in_scale = std::exp2f((float)in_fix_);
        int8_t* ip = reinterpret_cast<int8_t*>(
            reinterpret_cast<void*>(in_buf_->data({0, 0, 0, 0}).first));
        for (int h = 0; h < H_; ++h)
            for (int w = 0; w < W_; ++w)
                for (int c = 0; c < C_; ++c) {
                    float v = std::nearbyint(spatial[(size_t)c * H_ * W_ + (size_t)h * W_ + w] * in_scale);
                    if (v > 127.f) v = 127.f;
                    if (v < -128.f) v = -128.f;
                    ip[((size_t)h * W_ + w) * C_ + c] = (int8_t)v;
                }
        std::vector<vart::TensorBuffer*> ins{in_buf_.get()}, outs{out_buf_.get()};
        auto job = runner_->execute_async(ins, outs);
        runner_->wait(job.first, -1);
        const size_t out_n = (size_t)OH_ * OW_ * OC_;
        const int8_t* op = reinterpret_cast<const int8_t*>(
            reinterpret_cast<void*>(out_buf_->data({0, 0, 0, 0}).first));
        const float out_scale = std::exp2f((float)-out_fix_);
        for (int h = 0; h < OH_; ++h)
            for (int w = 0; w < OW_; ++w)
                for (int c = 0; c < OC_; ++c)
                    logits[(size_t)c * OH_ * OW_ + (size_t)h * OW_ + w] =
                        (float)op[((size_t)h * OW_ + w) * OC_ + c] * out_scale;
        (void)in_n; (void)out_n;
    }

    const char* name() const override { return "vart-board(dpu-run)"; }

private:
    static xir::Subgraph* find_dpu(xir::Subgraph* s) {
        if (s->has_attr("device") && s->get_attr<std::string>("device") == "DPU") return s;
        for (auto* c : s->get_children())
            if (auto* r = find_dpu(c)) return r;
        return nullptr;
    }
    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<vart::Runner> runner_;
    std::unique_ptr<vart::mm::HostFlatTensorBuffer> in_buf_, out_buf_;
    const xir::Tensor* in_t_ = nullptr;
    const xir::Tensor* out_t_ = nullptr;
    int H_ = 0, W_ = 0, C_ = 0, OH_ = 0, OW_ = 0, OC_ = 0, in_fix_ = 0, out_fix_ = 0;
};
}  // namespace
#endif  // USE_VART(板端真 VART runner)

#if defined(USE_VARTCPU)
// PC 跑量化 xmodel(VART cpu-runner, 单帧已对拍 QDQ 误差同量级)
#include <memory>
#include <string>

#include "cpu_runner.hpp"
#include "cpu_tb_factory.hpp"
#include "xir/graph/graph.hpp"
#include "xir/tensor/tensor.hpp"

namespace {
class VartCpuBackend : public DpuBackend {
public:
    explicit VartCpuBackend(const std::string& xmodel_path) {
        graph_ = xir::Graph::deserialize(xmodel_path);
        auto* root = graph_->get_root_subgraph();
        runner_ = std::make_unique<vart::cpu::CPURunner>(root);
        auto& fac = vart::cpu::CPUTBFactory::Instance();
        xir::Op* in_op = nullptr;
        xir::Op* out_op = nullptr;
        for (auto* op : graph_->get_head_ops()) {
            auto t = std::string(op->get_type());
            if (t != "const" && t != "const-fix") in_op = op;
        }
        for (auto* op : graph_->get_tail_ops()) {
            auto t = std::string(op->get_type());
            if (t != "const" && t != "const-fix") out_op = op;
        }
        in_tb_ = fac.create_or_get(in_op);
        out_tb_ = fac.create_or_get(out_op);
        in_t_ = in_tb_->get_tensor();
        out_t_ = out_tb_->get_tensor();
        auto ish = in_t_->get_shape(), osh = out_t_->get_shape();
        if (ish.size() != 4 || osh.size() != 4)
            throw std::runtime_error("xmodel IO shape 与预期不符(须 4 维 NHWC)");
        H_ = ish[1]; W_ = ish[2]; C_ = ish[3];   // 通道运行时读(32=vfe_out_dim 常态, 泛化 vel/3D head)
        OH_ = osh[1]; OW_ = osh[2]; OC_ = osh[3];
    }

    void infer(const float* spatial, float* logits) override {
        /*
        xmodel 前向: CHW→NHWC 直写内部 tb(输入 float, 量化在图内) → 非对称 execute → NHWC→CHW
        */
        float* ip = reinterpret_cast<float*>(
            reinterpret_cast<void*>(in_tb_->data({0, 0, 0, 0}).first));
        for (int h = 0; h < H_; ++h)
            for (int w = 0; w < W_; ++w) {
                float* dst = ip + ((size_t)h * W_ + w) * C_;
                for (int c = 0; c < C_; ++c)
                    dst[c] = spatial[(size_t)c * H_ * W_ + (size_t)h * W_ + w];
            }
        auto job = runner_->execute_async({}, {out_tb_});
        runner_->wait(job.first, -1);
        float* op = reinterpret_cast<float*>(
            reinterpret_cast<void*>(out_tb_->data({0, 0, 0, 0}).first));
        for (int h = 0; h < OH_; ++h)
            for (int w = 0; w < OW_; ++w) {
                const float* src = op + ((size_t)h * OW_ + w) * OC_;
                for (int c = 0; c < OC_; ++c)
                    logits[(size_t)c * OH_ * OW_ + (size_t)h * OW_ + w] = src[c];
            }
    }

    const char* name() const override { return "vart-cpu(xmodel-int8)"; }

private:
    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<vart::cpu::CPURunner> runner_;
    vart::TensorBuffer* in_tb_ = nullptr;
    vart::TensorBuffer* out_tb_ = nullptr;
    const xir::Tensor* in_t_ = nullptr;
    const xir::Tensor* out_t_ = nullptr;
    int H_ = 0, W_ = 0, C_ = 0, OH_ = 0, OW_ = 0, OC_ = 0;
};
}  // namespace
#endif  // USE_VARTCPU(PC 跑量化 xmodel; 板端走上方 USE_VART 真 runner)

// ONNX Runtime 后端(PC 常驻: S1 图对拍 / FP32 全链); 板端 BOARD 构建无 ORT 依赖
#if !defined(USE_VART)
#include <onnxruntime_cxx_api.h>

#include <array>
#include <vector>

namespace {
class OrtBackend : public DpuBackend {
public:
    explicit OrtBackend(const std::string& onnx_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "radar_pipeline"),
          session_(env_, onnx_path.c_str(), Ort::SessionOptions{}) {
        auto in = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        auto out = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        in_.assign(in.begin(), in.end());
        out_.assign(out.begin(), out.end());
    }

    void infer(const float* spatial, float* logits) override {
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
            const_cast<float*>(spatial), in_elems(), in_.data(), in_.size());
        const char* in_names[] = {"spatial_features"};
        const char* out_names[] = {"dpu_out"};
        auto out = session_.Run(Ort::RunOptions{}, in_names, &tensor, 1, out_names, 1);
        // 拷出(接口约定: 调用方管理 logits 缓冲)
        const float* src = out.front().GetTensorData<float>();
        std::memcpy(logits, src, sizeof(float) * out_elems());
    }

    const char* name() const override { return "onnxruntime-cpu"; }

private:
    size_t in_elems() const {
        size_t n = 1;
        for (int64_t d : in_) n *= (size_t)d;
        return n;
    }
    size_t out_elems() const {
        size_t n = 1;
        for (int64_t d : out_) n *= (size_t)d;
        return n;
    }
    Ort::Env env_;
    Ort::Session session_;
    std::vector<int64_t> in_, out_;
};
}  // namespace
#endif  // !USE_VART

std::unique_ptr<DpuBackend> create_dpu_backend(const std::string& model_path) {
    /*
    后端工厂: .xmodel → VART(板端真 runner / PC cpu-runner 按编译宏分流); 其他 → ONNX Runtime
    */
    const bool is_xmodel = model_path.size() > 7 &&
        model_path.compare(model_path.size() - 7, 7, ".xmodel") == 0;
#if defined(USE_VART)
    if (is_xmodel) return std::make_unique<VartBoardBackend>(model_path);
#elif defined(USE_VARTCPU)
    if (is_xmodel) return std::make_unique<VartCpuBackend>(model_path);
#endif
#if !defined(USE_VART)
    return std::make_unique<OrtBackend>(model_path);
#else
    throw std::runtime_error("BOARD 构建仅支持 .xmodel(compiled), 无 ORT 后端: " + model_path);
#endif
}
