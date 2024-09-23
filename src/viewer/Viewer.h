#ifndef MESHVIEWER_H
#define MESHVIEWER_H

#include <string>
#include <memory>
#include <mutex>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <pangolin/display/display.h>
// #include <pangolin/display/view.h>
// #include <pangolin/handler/handler.h>
// #include <pangolin/gl/gldraw.h>

#include <pangolin/var/var.h>
// #include <pangolin/var/varextra.h>
// #include <pangolin/gl/gl.h>
// #include <pangolin/gl/gldraw.h>
// #include <pangolin/display/display.h>
// #include <pangolin/display/view.h>
// #include <pangolin/display/widgets.h>
// #include <pangolin/display/default_font.h>
#include <pangolin/handler/handler.h>
#include <yaml-cpp/yaml.h>
namespace stbr {
    class database;
}


namespace Viewer {

class MeshDrawer;

class MeshViewer {
    public:
        MeshViewer();
        MeshViewer(stbr::database *db, cv::Mat img, std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, const YAML::Node &config);


        void run();
    private:

        void create_menu_panel();
        void request_terminate(); 
        void draw_current_cam_pose();
        void draw_camera() const;
        void draw_frustum() const;
        void draw_current_mesh();
        void draw_current_gt_pc();

        int width_ = 1080;
        int height_ = 720;
        std::string viewer_name_ = "Map Viewer";
        cv::Mat img_;
        std::vector<Eigen::Vector3d> vertices_;
        std::vector<Eigen::Vector3i> triangles_;
        bool isFinished_ = false;

        MeshDrawer *mDrawer = nullptr;
        stbr::database *db_ = nullptr;
        // menu panel
        std::unique_ptr<pangolin::Var<bool>> menu_terminate_;
        std::unique_ptr<pangolin::Var<bool>> menu_pause_;
        std::unique_ptr<pangolin::Var<bool>> menu_show_GT_;
        std::unique_ptr<pangolin::Var<bool>> menu_show_mesh_;
        std::unique_ptr<pangolin::Var<float>> menu_frm_size_;
        
        // // camera renderer
        std::unique_ptr<pangolin::OpenGlRenderState> s_cam_;
};

// inline void draw_line(const float x1, const float y1, const float z1,
//                               const float x2, const float y2, const float z2) const {
//     glVertex3f(x1, y1, z1);
//     glVertex3f(x2, y2, z2);
// }

}

#endif // MESHVIEWER_H

