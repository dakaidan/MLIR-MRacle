#include "conquer/loaders/data/npy_loader.h"
#include "conquer/loaders/data/data.h"
#include "conquer/loaders/manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

std::vector<conquer::TensorAllocation> conquer::NpyDataLoader::loadData(const std::string &dataPath) {
    std::ifstream file(dataPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("NpyDataLoader: Could not open file " + dataPath);
    }

    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("NpyDataLoader: Invalid .npy file (magic string mismatch)");
    }

    uint8_t major_v, minor_v;
    file.read(reinterpret_cast<char *>(&major_v), 1);
    file.read(reinterpret_cast<char *>(&minor_v), 1);

    uint32_t header_len = 0;
    if (major_v == 1) {
        uint16_t len;
        file.read(reinterpret_cast<char *>(&len), 2);
        header_len = len;
    } else if (major_v == 2 || major_v == 3) {
        file.read(reinterpret_cast<char *>(&header_len), 4);
    } else {
        throw std::runtime_error("NpyDataLoader: Unsupported .npy version");
    }

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    TensorAllocation alloc;

    std::string descr = extract_dict_value(header, "'descr':");
    if (descr.empty())
        throw std::runtime_error("NpyDataLoader: Could not find 'descr' in header");

    std::erase(descr, '\'');
    std::erase(descr, '\"');

    bool requires_f8_to_f4_downcast = false;

    if (descr.find("f8") != std::string::npos) {
        alloc.dtype = DataType::FP32;
        requires_f8_to_f4_downcast = true;
    } else if (descr.find("f4") != std::string::npos) {
        alloc.dtype = DataType::FP32;
    } else if (descr.find("f2") != std::string::npos) {
        alloc.dtype = DataType::FP16;
    } else if (descr.find("i1") != std::string::npos || descr.find("u1") != std::string::npos) {
        alloc.dtype = DataType::INT8;
    } else if (descr.find("i4") != std::string::npos) {
        alloc.dtype = DataType::INT32;
    } else {
        throw std::runtime_error("NpyDataLoader: Unsupported numpy data type: " + descr);
    }

    if (const std::string fortran_order = extract_dict_value(header, "'fortran_order':");
        fortran_order.find("True") != std::string::npos) {
        throw std::runtime_error("NpyDataLoader: Fortran order arrays are not currently supported");
    }

    std::string shape_str = extract_dict_value(header, "'shape':");
    size_t start = shape_str.find('(');
    size_t end = shape_str.find(')');
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("NpyDataLoader: Could not parse shape");
    }

    std::string dims = shape_str.substr(start + 1, end - start - 1);
    size_t pos = 0;
    while ((pos = dims.find(',')) != std::string::npos) {
        std::string token = dims.substr(0, pos);
        trim(token);
        if (!token.empty())
            alloc.shape.push_back(std::stoll(token));
        dims.erase(0, pos + 1);
    }
    trim(dims);
    if (!dims.empty())
        alloc.shape.push_back(std::stoll(dims));

    auto current_pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    file.seekg(current_pos, std::ios::beg);

    const size_t data_size = end_pos - current_pos;

    if (requires_f8_to_f4_downcast) {
        size_t num_elements = data_size / sizeof(double);
        std::vector<double> f64_buffer(num_elements);
        if (!file.read(reinterpret_cast<char *>(f64_buffer.data()), static_cast<long>(data_size))) {
            throw std::runtime_error("NpyDataLoader: Failed to read f8 binary data payload");
        }

        alloc.buffer.resize(num_elements * sizeof(float));
        auto f32_ptr = reinterpret_cast<float *>(alloc.buffer.data());

        for (size_t i = 0; i < num_elements; ++i) {
            f32_ptr[i] = static_cast<float>(f64_buffer[i]);
        }
    } else {
        alloc.buffer.resize(data_size);
        if (!file.read(alloc.buffer.data(), static_cast<long>(data_size))) {
            throw std::runtime_error("NpyDataLoader: Failed to read binary data payload");
        }
    }

    return {std::move(alloc)};
}

[[nodiscard]] bool conquer::NpyDataLoader::supportsExtension(const std::string &extension) const { return extension == "npy"; }

void conquer::NpyDataLoader::trim(std::string &s) {
    s.erase(s.begin(), std::ranges::find_if(s, [](const unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](const unsigned char ch) { return !std::isspace(ch); }).base(),
            s.end());
}

std::string conquer::NpyDataLoader::extract_dict_value(const std::string &header, const std::string &key) {
    const size_t key_pos = header.find(key);
    if (key_pos == std::string::npos)
        return "";

    size_t val_start = key_pos + key.length();

    while (val_start < header.length() && std::isspace(header[val_start])) {
        val_start++;
    }

    size_t val_end = val_start;
    bool in_tuple = false;
    bool in_quote = false;

    while (val_end < header.length()) {
        if (const char c = header[val_end]; c == '\'' || c == '\"') {
            in_quote = !in_quote;
        } else if (!in_quote) {
            if (c == '(') {
                in_tuple = true;
            } else if (c == ')') {
                in_tuple = false;
            } else if (!in_tuple && (c == ',' || c == '}')) {
                break;
            }
        }
        val_end++;
    }

    if (val_end == val_start)
        return "";

    std::string val = header.substr(val_start, val_end - val_start);
    trim(val);
    return val;
}
