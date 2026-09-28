#include "point_io.h"

#include <fstream>
#include <stdexcept>

std::vector<float> read_points_bin(const std::string& path, int& num_points) {
    /*
    点文件读取: raw f32 little-endian, 列数固定 6
    */
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("点文件打不开: " + path);
    f.seekg(0, std::ios::end);
    long bytes = f.tellg();
    f.seekg(0);
    if (bytes % (6 * sizeof(float)) != 0)
        throw std::runtime_error("点文件字节数非 6 列倍数: " + path);
    num_points = (int)(bytes / (6 * sizeof(float)));
    std::vector<float> buf((size_t)num_points * 6);
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    return buf;
}

void write_f32_bin(const std::string& path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("写出失败: " + path);
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)(n * sizeof(float)));
}

std::vector<float> read_f32_bin(const std::string& path, size_t expect_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("读入失败: " + path);
    std::vector<float> buf(expect_n);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(expect_n * sizeof(float)));
    if (f.gcount() != (std::streamsize)(expect_n * sizeof(float)))
        throw std::runtime_error("读入字节数不符: " + path);
    return buf;
}
