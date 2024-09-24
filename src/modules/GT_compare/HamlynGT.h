#ifndef HAMLYNGT_H
#define HAMLYNGT_H


#include "GT_compare/GroundTruth_compare.h"
#include <yaml-cpp/yaml.h>
#include <string>

namespace stbr {
class GroundTruth;

class HamlynGT : public GroundTruth_compare {
public:
    HamlynGT(const YAML::Node &config);

    void compareWithGroundTruth(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, std::vector<Eigen::Vector3d> &gt_pc) override;
private:
    const YAML::Node config_;
    int FrameNo_;
    int FPS_;
    double mutliplier_;
    int modulo_;
    double addition_; 
    std::string path_;
};
}
#endif // HAMLYNGT_H