#include "Viewer.h"
#include "MeshDrawer.h"
#include "database.h"

#include <pangolin/pangolin.h>
using namespace stbr;

namespace Viewer {

MeshViewer::MeshViewer() {}

MeshViewer::MeshViewer(database *db, cv::Mat img, std::vector<Eigen::Vector3d> vertices, std::vector<Eigen::Vector3i> triangles, const YAML::Node &config) {
    img_ = img;
    vertices_ = vertices;
    db_ = db;
    triangles_ = triangles;
    mDrawer = new MeshDrawer(img, vertices, triangles, config);
}

void MeshViewer::create_menu_panel() {
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(150));
    menu_show_mesh_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Mesh", true, true));
    menu_show_GT_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show GT", false, true));
    menu_pause_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Pause", false, true));
    menu_terminate_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Terminate", false, false));
    menu_frm_size_ = std::unique_ptr<pangolin::Var<float>>(new pangolin::Var<float>("menu.Frame Size", 1.0, 1e-1, 1e1, true));
}   

void MeshViewer::request_terminate() {
    isFinished_ = true;
    db_->setTerminate();
}

void MeshViewer::draw_current_cam_pose() {

    glLineWidth(2);
    GLfloat color[] = {0.75f, 0.75f, 1.0f};
    glColor3fv(color);
    draw_camera();
}

void MeshViewer::draw_camera() const {
    glPushMatrix();
    draw_frustum();
    glPopMatrix();
}

void MeshViewer::draw_frustum() const {
    constexpr int image_width = 1080 ;
    constexpr int image_height = 720 ;
    constexpr float horizontal_fov = 2 * M_PI / 3; // 120 deg
    const float focal_length_pix = 0.5 * image_width / std::tan(0.5 * horizontal_fov);
    Eigen::Matrix3d Kinv = Eigen::Matrix3d::Identity();
    Kinv(0, 0) = 1.0 / focal_length_pix;
    Kinv(1, 1) = 1.0 / focal_length_pix;
    Kinv(0, 2) = -0.5 * image_width / focal_length_pix;
    Kinv(1, 2) = -0.5 * image_height / focal_length_pix;
    const float z = *menu_frm_size_ / image_width * focal_length_pix;
    pangolin::glDrawFrustum(Kinv, image_width, image_height, z);
}

void MeshViewer::draw_current_mesh() {
    

    // different approach!
    // for now I assume that I need a pair of different triangles and different veritces

    db_->getTexture(img_);
    db_->getVertices(vertices_);
    // db_->getTriangles(triangles_);
    // db_->getVerticesAndTriangles(vertices_, triangles_);
    // Get Image and get Vertices
    mDrawer->addVertices(vertices_);
    mDrawer->addTriangles(triangles_);
    mDrawer->addTextureImage(img_);

    mDrawer->drawMesh(0.99);
}

void MeshViewer::draw_current_gt_pc() {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3f> pts;
    db_->getGT(points);
    
    for(int i=0; i < points.size(); i++) {
        pts.push_back(points[i].cast<float>());
    }
    pangolin::glDrawPoints(pts);
}

void MeshViewer::run() {

    pangolin::OpenGlMatrix rotation_180_z;
rotation_180_z.SetIdentity();
rotation_180_z.m[0] = -1; // Invert X-axis
rotation_180_z.m[5] = -1; // Invert Y-axis
    // glewExperimental = GL_TRUE;
    pangolin::CreateWindowAndBind(viewer_name_, width_, height_);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    s_cam_ = std::unique_ptr<pangolin::OpenGlRenderState>(new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1080,720,420,420,320,240,0.2,1000),
        pangolin::ModelViewLookAt(0,30,-10,15,25,30, pangolin::AxisY) * rotation_180_z
    ));

    // Create Interactive View in window
    // pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1080.0f/720.0f)
            .SetHandler(new pangolin::Handler3D(*s_cam_));


    create_menu_panel();

    bool var_pause = false;

    while(!isFinished_) {
        std::chrono::microseconds(1000);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(*s_cam_);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // // Draw the camera pose -> it is fixed 
        draw_current_cam_pose();

        // // draw Mesh!
        if(*menu_show_mesh_) {
            draw_current_mesh();
        }

        if(*menu_show_GT_) {
            draw_current_gt_pc();
        }

        if(*menu_pause_ != var_pause) {
            var_pause = *menu_pause_;
            if(var_pause)
                db_->setPause();
            else
                db_->setUnpause();
        }

        
        pangolin::FinishFrame();

        if (*menu_terminate_ || pangolin::ShouldQuit()) {
            request_terminate();
        }
    }

}

};// Namespace