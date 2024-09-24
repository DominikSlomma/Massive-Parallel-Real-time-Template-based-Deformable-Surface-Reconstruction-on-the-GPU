#include "database.h"
namespace stbr {
database::database() {}

void database::setVertices(std::vector<Eigen::Vector3d> vertices) {
    std::lock_guard<std::mutex> lock(mtx_vertex_);
    vertices_ = vertices;
}

void database::setVerticesAndTriangles(std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles) {
    std::lock_guard<std::mutex> lock(mtx_vertex_);
    vertices_ = vertices; 
    triangles_ = triangles;
}

void database::getVerticesAndTriangles(std::vector<Eigen::Vector3d> &vertices, std::vector<Eigen::Vector3i> &triangles) {
    std::lock_guard<std::mutex> lock(mtx_vertex_);

    vertices = vertices_;
    triangles =triangles_;
}

void database::setTexture(cv::Mat texture) {
    std::lock_guard<std::mutex> lock(mtx_texture_);
    texture_ = texture;
}

void database::getVertices(std::vector<Eigen::Vector3d> &vertices) {
    std::lock_guard<std::mutex> lock(mtx_vertex_);
    vertices = vertices_;
}

void database::getTriangles(std::vector<Eigen::Vector3i> &triangles) {
    std::lock_guard<std::mutex> lock(mtx_vertex_);
    triangles = triangles_;
}

void database::getTexture(cv::Mat &texture) {
    std::lock_guard<std::mutex> lock(mtx_texture_);
    texture = texture_;
}

void database::setPause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    pause_ = true;
}

void database::setUnpause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    pause_ = false;
}

bool database::isPause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_;
}

void database::setTerminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_ = true;
}

bool database::isTerminated() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_;
}

void database::setGT(std::vector<Eigen::Vector3d> gt_pc) {
    std::lock_guard<std::mutex> lock(mtx_gt_);
    gt_pc_ = gt_pc;
}

void database::getGT(std::vector<Eigen::Vector3d> &gt_pc) {
    std::lock_guard<std::mutex> lock(mtx_gt_);
    gt_pc = gt_pc_;
}

} // namespace