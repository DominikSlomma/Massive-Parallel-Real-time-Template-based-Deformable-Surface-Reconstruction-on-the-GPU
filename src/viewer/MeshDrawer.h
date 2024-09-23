#ifndef MESHDRAWER_H
#define MESHDRAWER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>


namespace Viewer {

class MeshDrawer {
    public:
        MeshDrawer(cv::Mat img, std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, const YAML::Node &config);
        void addTextureImage(cv::Mat &im);
        void addVertices(std::vector<Eigen::Vector3d> &vertices);
        void addTriangles(std::vector<Eigen::Vector3i> &triangles);

        void drawMesh(double alpha=1, bool useEdges=true);

    private:
        void compute_uvs();
        std::vector<Eigen::Vector3i> triangles_;
        std::vector<Eigen::Vector3d> vertices_;
        std::vector<Eigen::Vector2d> uvs_;
        cv::Mat texture_;

        double fx_=0;
        double fy_=0;
        double cx_=0;
        double cy_=0;
        int width_;
        int height_;
};
} // Namespace

#endif // MESHDRAWER_H