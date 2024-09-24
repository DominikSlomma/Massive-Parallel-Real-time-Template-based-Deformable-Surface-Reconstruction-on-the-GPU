#ifndef SFTGT_H
#define SFTGT_H


#include "GT_compare/GroundTruth_compare.h"
#include <yaml-cpp/yaml.h>
#include <string>

namespace stbr {
class GroundTruth;

class SfTGT : public GroundTruth_compare {
public:
    SfTGT(const YAML::Node &config);

    void compareWithGroundTruth(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, std::vector<Eigen::Vector3d> &gt_pc) override;
private:
    const YAML::Node config_;
    int FrameNo_;
    int max_number_ = 0;
    std::string path_;
};
}

#endif // SFTGT_H