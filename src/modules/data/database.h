#ifndef DATABASE_H
#define DATABASE_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>

namespace Viewer {
    class MeshViewer;
}
namespace stbr {
class System;

class database {
    public:
        database();

        void setVertices(std::vector<Eigen::Vector3d> vertices);
        void setVerticesAndTriangles(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles);
        void setTexture(cv::Mat texture);
        void setPause();
        void setUnpause();
        void setTerminate();
        void setGT(std::vector<Eigen::Vector3d> gt_pc);

        void getGT(std::vector<Eigen::Vector3d> &gt_pc);
        bool isTerminated();
        bool isPause();
        void getVertices(std::vector<Eigen::Vector3d> &vertices);
        void getTriangles(std::vector<Eigen::Vector3i> &triangles);
        void getTexture(cv::Mat &texture);
        void getVerticesAndTriangles(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles);


    private:
        std::vector<Eigen::Vector3d> gt_pc_;
        std::vector<Eigen::Vector3d> vertices_;
        std::vector<Eigen::Vector3i> triangles_;
        cv::Mat texture_;
        bool pause_ = false;
        bool terminate_ = false;

        mutable std::mutex mtx_gt_;
        mutable std::mutex mtx_pause_;
        mutable std::mutex mtx_terminate_;
        mutable std::mutex mtx_vertex_;
        mutable std::mutex mtx_texture_;
};
} // namespace
#endif