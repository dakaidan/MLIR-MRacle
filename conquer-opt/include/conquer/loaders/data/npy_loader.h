#pragma once
#include "conquer/loaders/data/data.h"

#include <string>


namespace conquer {

class NpyDataLoader : public DataLoader {
  public:
    std::vector<TensorAllocation> loadData(const std::string &dataPath) override;

    [[nodiscard]] bool supportsExtension(const std::string &extension) const override;

  private:
    static void trim(std::string &s);

    static std::string extract_dict_value(const std::string &header, const std::string &key);
};

} // namespace conquer