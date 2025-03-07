#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <boost/format.hpp>

#include <sophus/se3.h>

using namespace std;
using namespace g2o;

/********************************************
 * 本节演示了RGBD上的稀疏直接法 
 ********************************************/

// 相机内参
// float fx = 518.0;
// float fy = 519.0;
// float cx = 325.5;
// float cy = 253.5;

float fx = 637.27803366;
float fy = 637.30526147;
float cx = 636.3285782;
float cy = 377.00039794;
float depth_scale = 1000.0;
Eigen::Matrix3d K;

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect ( const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& intrinsics, Eigen::Isometry3d& Tcw );


// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect: public BaseUnaryEdge< 1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image )
        : x_world_ ( point ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image )
    {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );
        float x = x_local[0]*fx_/x_local[2] + cx_;
        float y = x_local[1]*fy_/x_local[2] + cy_;
        // check x,y is in the image
        if ( x-4<0 || ( x+4 ) >image_->cols || ( y-4 ) <0 || ( y+4 ) >image_->rows )
        {
            _error ( 0,0 ) = 0.0;
            this->setLevel ( 1 );
        }
        else
        {
            _error ( 0,0 ) = getPixelValue ( x,y ) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus( )
    {
        if ( level() == 1 )
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        float u = x*fx_*invz + cx_;
        float v = y*fy_*invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {return true;}
    virtual bool write ( std::ostream& out ) const {return true;}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue ( float x, float y )
    {
        uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                   ( 1-xx ) * ( 1-yy ) * data[0] +
                   xx* ( 1-yy ) * data[1] +
                   ( 1-xx ) *yy*data[ image_->step ] +
                   xx*yy*data[image_->step+1]
               );
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics
    cv::Mat* image_=nullptr;    // reference image
};

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3> { // Use Sophus::SE3 instead of g2o::SE3
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};

class EdgeProjectionDirect : public g2o::BaseUnaryEdge<1, double, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectionDirect(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K, cv::Mat* image) : _pos3d(pos), _K(K), image_(image) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3 T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error(0, 0) = getPixelValue ( pos_pixel[0], pos_pixel[1]) - _measurement;
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3 T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx_ = _K(0, 0);
    double fy_ = _K(1, 1);
    double cx_ = _K(0, 2);
    double cy_ = _K(1, 2);
    double x = pos_cam[0];
    double y = pos_cam[1];
    double z = pos_cam[2];
    double invz = 1.0 / z;
    double invz_2 = invz*invz;
    float pix_u = fx_ * x / z + cx_, pix_v = fy_ * y / z + cy_;
    _jacobianOplusXi << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    for (int x = -half_patch_size_; x <= half_patch_size_; x++) {
        for (int y = -half_patch_size_; y <= half_patch_size_; y++) {
            // _jacobianOplusXi
            //   << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
            //   0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
            Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
            jacobian_uv_ksai ( 0,3 ) = - x*y*invz_2 *fx_;
            jacobian_uv_ksai ( 0,4 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
            jacobian_uv_ksai ( 0,5 ) = - y*invz *fx_;
            jacobian_uv_ksai ( 0,0 ) = invz *fx_;
            jacobian_uv_ksai ( 0,1 ) = 0;
            jacobian_uv_ksai ( 0,2 ) = -x*invz_2 *fx_;

            jacobian_uv_ksai ( 1,3 ) = - ( 1+y*y*invz_2 ) *fy_;
            jacobian_uv_ksai ( 1,4 ) = x*y*invz_2 *fy_;
            jacobian_uv_ksai ( 1,5 ) = x*invz *fy_;
            jacobian_uv_ksai ( 1,0 ) = 0;
            jacobian_uv_ksai ( 1,1 ) = invz *fy_;
            jacobian_uv_ksai ( 1,2 ) = -y*invz_2 *fy_;

            Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

            jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( pix_u+1+x, pix_v+y )-getPixelValue ( pix_u-1+x, pix_v+y ) ) /2;
            jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( pix_u+x, pix_v+1+y )-getPixelValue ( pix_u+x, pix_v-1+y ) ) /2;

            _jacobianOplusXi += jacobian_pixel_uv*jacobian_uv_ksai;
        }
    }
    
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue ( float x, float y )
    {
        if(x<0)x=0; if(y<0)y=0; if(x>image_->cols-1)x=image_->cols-1; if(y>image_->rows-1)y=image_->rows-1;
        uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                   ( 1-xx ) * ( 1-yy ) * data[0] +
                   xx* ( 1-yy ) * data[1] +
                   ( 1-xx ) *yy*data[ image_->step ] +
                   xx*yy*data[image_->step+1]
               );
    }
private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
    cv::Mat* image_=nullptr;
    const int half_patch_size_ = 0;

};

void extractFeatures(cv::Mat& gray, cv::Mat& depth, vector<Measurement>& measurements, int mode) {
    measurements.clear();
    switch (mode)
    {
        case 0: {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect ( gray, keypoints );
            for ( auto kp:keypoints )
            {
                // 去掉邻近边缘处的点
                if ( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) >gray.cols || ( kp.pt.y+20 ) >gray.rows )
                    continue;
                ushort d = depth.ptr<ushort> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ];
                if ( d==0 )
                    continue;
                Eigen::Vector3d p3d = project2Dto3D ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale );
                float grayscale = float ( gray.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );
                measurements.push_back ( Measurement ( p3d, grayscale ) );
            }
            break;
        }

        case 1: {
            int grid_size = 40; // grid size is 40*40
            int rows = gray.rows; // 360
            int cols = gray.cols; // 640
            int rows_grid = rows / grid_size; // 9
            int cols_grid = cols / grid_size; // 16
            int* grad_cnt = new int[100];
            for(int y=0;y<rows_grid;y++) {
                for(int x=0;x<cols_grid;x++) {
                    memset(grad_cnt, 0, sizeof(int)*100);

                    for (int j = 0; j < grid_size; j++) { // row
                        for (int i = 0; i < grid_size; i++) { // col
                            Eigen::Vector2d delta (
                                gray.ptr<uchar>(j+grid_size*y)[i+grid_size*x+1] - gray.ptr<uchar>(j+grid_size*y)[i+grid_size*x-1], 
                                gray.ptr<uchar>(j+grid_size*y+1)[i+grid_size*x] - gray.ptr<uchar>(j+grid_size*y-1)[i+grid_size*x]
                            );
                            int g = delta.norm();
                            if ( g < 30 )
                                continue;
                            if ( g > 98) g = 98;
                            grad_cnt[g+1]++;
                            grad_cnt[0]++;
                        }
                    }

                    int grad_th_cnt = grad_cnt[0]*0.3;
                    int th = 0;
                    int max_num = 100;
                    for (int i = 99; i > 30; i--) {
                        grad_th_cnt -= grad_cnt[i];
                        max_num -= grad_cnt[i];
                        // if (grad_th_cnt < 0 || max_num < 0) {
                        if (max_num < 0) {
                            th = i;
                            break;
                        }
                    }
                    if (th<30) th = 30;
                    if(y==0 && x==0) cout << "th = " << th << " " << grad_cnt[0]*0.5 << " "<< grad_cnt[1] << " "<< grad_cnt[2] << endl;
                    for (int j = 1; j < grid_size-1; j++) { // row
                        for (int i = 1; i < grid_size-1; i++) { // col
                            Eigen::Vector2d delta (
                                gray.ptr<uchar>(j+grid_size*y)[i+grid_size*x+1] - gray.ptr<uchar>(j+grid_size*y)[i+grid_size*x-1], 
                                gray.ptr<uchar>(j+grid_size*y+1)[i+grid_size*x] - gray.ptr<uchar>(j+grid_size*y-1)[i+grid_size*x]
                            );
                            int g = delta.norm();
                            // if ( g < 9 )
                            //     continue;
                            if ( g > th) {
                                ushort d = depth.ptr<ushort> (j+grid_size*y)[i+grid_size*x+1];
                                if ( float ( d ) /depth_scale < 0.3 || float ( d ) /depth_scale > 6)
                                    continue;
                                Eigen::Vector3d p3d = project2Dto3D ( i+grid_size*x, j+grid_size*y, d, fx, fy, cx, cy, depth_scale );
                                float grayscale = float ( gray.ptr<uchar> (j+grid_size*y)[i+grid_size*x+1] );
                                measurements.push_back ( Measurement ( p3d, grayscale ) );
                            }

                        }
                    }
                }
            }
            delete grad_cnt;
            break;
        }

        case 2: {
            const int half_patch_size = 1;
            // select the pixels with high gradiants 
            for ( int x=1; x<gray.cols-1; x+=2 ) {
                for ( int y=1; y<gray.rows-1; y+=2 ) {
                    Eigen::Vector2d delta (
                        gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1], 
                        gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]
                    );

                    if ( delta.norm() < 50 )
                        continue;
                    ushort d = depth.ptr<ushort> (y)[x];
                    if ( float(d)/depth_scale<0.3 || float(d)/depth_scale>8 )
                        continue;
                    Eigen::Vector3d p3d = project2Dto3D ( x, y, d, fx, fy, cx, cy, depth_scale );
                    // float grayscale = 0.0;
                    // for (int u = -half_patch_size; u <= half_patch_size; u++) {
                    //     for (int v = -half_patch_size; v <= half_patch_size; v++) {
                    //         grayscale += float ( gray.ptr<uchar> (y+v) [x+u] );
                    //     }
                    // }
                    float grayscale = float ( gray.ptr<uchar> (y) [x] );
                    measurements.push_back ( Measurement ( p3d, grayscale ) );
                }
            }
            break;
        }

        case 3: {
            cv::Mat grad_x, grad_y;
            cv::Mat abs_grad_x, abs_grad_y, dst;

            //求x方向梯度
            Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
            convertScaleAbs(grad_x, abs_grad_x);
            imshow("x方向soble", abs_grad_x);

            //求y方向梯度
            Sobel(gray, grad_y,CV_16S,0, 1,3, 1, 1, cv::BORDER_DEFAULT);
            convertScaleAbs(grad_y,abs_grad_y);
            imshow("y向soble", abs_grad_y);

            //合并梯度
            addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
            imshow("整体方向soble", dst);
            cv::waitKey(0);
            break;
        }

    }
    cout<<"add total "<<measurements.size()<<" measurements."<<endl;
}

void generateImages(cv::Mat& img_ref, cv::Mat& depth, cv::Mat& gray) {
    Eigen::Matrix3d rot_mat_temp = Eigen::Matrix3d::Identity();
    rot_mat_temp = Eigen::AngleAxisd(0*3.14159/180.0, Eigen::Vector3d::UnitZ()) * 
    Eigen::AngleAxisd(1*3.14159/180.0, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
    Eigen::Vector3d tran_temp = Eigen::Vector3d::Zero();
    tran_temp << 0, 0.03, 0;
    for (int u = 0; u < img_ref.cols; u++) {
        for (int v = 0; v < img_ref.rows; v++) {

            ushort d = depth.ptr<ushort> (v)[u];
            if ( d <= 0 )
                continue;
            Eigen::Vector3d p3d = project2Dto3D ( u, v, d, fx, fy, cx, cy, depth_scale );
            Eigen::Vector3d pnt_cam_rot = rot_mat_temp * p3d + tran_temp;
            Eigen::Vector2d pnt_pixel = Eigen::Vector2d (
                                        fx * pnt_cam_rot ( 0,0 ) / pnt_cam_rot ( 2,0 ) + cx,
                                        fy * pnt_cam_rot ( 1,0 ) / pnt_cam_rot ( 2,0 ) + cy);
            if(pnt_pixel[0] > 0 && pnt_pixel[0] < img_ref.cols && pnt_pixel[1] > 0 && pnt_pixel[1] < img_ref.rows) {
                gray.at<uchar>(pnt_pixel[1], pnt_pixel[0]) = img_ref.at<uchar>(v, u);
            }
        }
    }
}

int main ( int argc, char** argv )
{
    if ( argc != 1 )
    {
        cout<<"usage: useLK path_to_dataset. No need to provide now."<<endl;
        return 1;
    }
    srand ( ( unsigned int ) time ( 0 ) );
    // string path_to_dataset = argv[1];
    string path_to_dataset = "../../data/data";
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin ( associate_file );

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray;
    vector<Measurement> measurements;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    boost::format fmt("%s/image_%d/%06d.png"); // 10 us
    string rgbd_dataset_path_ = "/home/zh/data/img";
    
    cv::Mat prev_color, prev_gray, prev_depth, gray_ori, prev_gray_ori;
    bool first_img = false;
    gray = cv::imread(rgbd_dataset_path_+"/image_0/000699.png", cv::IMREAD_GRAYSCALE);
    depth = cv::imread(rgbd_dataset_path_+"/image_1/000699.png", cv::IMREAD_UNCHANGED);
    cv::Mat gray_new = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1);
    // cv::imshow("graynew", gray_new);
    // cv::waitKey(0);
    extractFeatures(gray, depth, measurements, 2);
    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    for ( int index=700; index<3500; index+=1) {
    // for ( int index=590; index<750; index+=2 ) {
        cout<<"*********** loop "<<index<<" ************"<<endl;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        // color = cv::imread ( path_to_dataset+"/"+rgb_file );
        // depth = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
        // cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
        gray = cv::imread((fmt % rgbd_dataset_path_ % 0 % index).str(), cv::IMREAD_GRAYSCALE);
        // generateImages(gray, depth, gray_new);
        // gray = gray_new.clone();
        
        // gray_ori = gray.clone();
        // cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2);

        if ( index % 3 == 0) {
            depth = cv::imread((fmt % rgbd_dataset_path_ % 1 % index).str(), cv::IMREAD_UNCHANGED);
            extractFeatures(gray, depth, measurements, 2);
            continue;
        }
 

        // if ( gray.data==nullptr || depth.data==nullptr )
        //     continue; 
        // if ( !first_img ) {
        //     first_img = true;
        //     prev_gray = gray.clone();
        //     prev_depth = depth.clone();
        //     prev_gray_ori = gray_ori.clone();
        //     continue;
        // }
            

        // poseEstimationDirect ( measurements, &gray, K, Tcw );
    // }
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolverType;
    // DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    std::unique_ptr<DirectBlock::LinearSolverType> linearSolver (new g2o::LinearSolverDense< DirectBlock::PoseMatrixType>());
    // DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    std::unique_ptr<DirectBlock> solver_ptr (new DirectBlock ( std::move(linearSolver)));
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(solver_ptr) ); // G-N
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr) ); // L-M
    // auto solver = new g2o::OptimizationAlgorithmLevenberg(
    //             g2o::make_unique<DirectBlock>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    // optimizer.setVerbose( true ); // output info






    VertexPose* pose = new VertexPose();
    pose->setEstimate ( Sophus::SE3 ( Tcw.rotation(), Tcw.translation() ) );
    pose->setId ( 0 );
    optimizer.addVertex ( pose );
    std::vector<EdgeProjectionDirect*> edges; // only optimize pose
    // 添加边
    int id=1;
    for ( Measurement m: measurements ) {
        EdgeProjectionDirect* edge = new EdgeProjectionDirect (m.pos_world, K, &gray);
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
        edges.push_back(edge);
    }

    // optimizer.initializeOptimization();
    // optimizer.optimize(10);

    cout << "pose = " << endl << pose->estimate().rotation_matrix() << endl << pose->estimate().translation().transpose() << endl;
    std::vector<bool> features(edges.size(), false);
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        // vertex_pose->setEstimate(vertex_pose->estimate()); // no need to set estimation
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;
        std::cout << "iteration=  " << iteration << std::endl;
        // features stores the pointer to features_left_

            const double chi2_th = 400;
            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (features[i]) {
                    e->computeError();
                }
                if (e->chi2() > chi2_th) {
                    features[i] = true;
                    e->setLevel(1); // do not optimize
                    cnt_outlier++;
                } else {
                    features[i] = false;
                    e->setLevel(0); // optimize
                };

                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
    }
    Tcw = Eigen::Isometry3d::Identity();
    Tcw.rotate ( pose->estimate().rotation_matrix() );
    Tcw.pretranslate ( pose->estimate().translation() );




    // g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    // pose->setEstimate ( g2o::SE3Quat ( Tcw.rotation(), Tcw.translation() ) );
    // pose->setId ( 0 );
    // optimizer.addVertex ( pose );
    // std::vector<EdgeSE3ProjectDirect*> edges_direct; // only optimize pose

    // // 添加边
    // int id=1;
    // for ( Measurement m: measurements )
    // {
    //     EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
    //         m.pos_world,
    //         K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), &gray
    //     );
    //     edge->setVertex ( 0, pose );
    //     edge->setMeasurement ( m.grayscale );
    //     edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
    //     edge->setId ( id++ );
    //     optimizer.addEdge ( edge );
    //     edges_direct.push_back(edge);
    // }
    // cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    // // optimizer.initializeOptimization();
    // // optimizer.optimize ( 30 );
    // // std::cout << "0" << std::endl;
    // // Tcw = pose->estimate();
    // // std::cout << "1" << std::endl;

    //     std::vector<bool> features(edges_direct.size(), false);
    //     int cnt_outlier = 0;
    //     for (int iteration = 0; iteration < 4; ++iteration) {
    //         // vertex_pose->setEstimate(vertex_pose->estimate()); // no need to set estimation
    //         optimizer.initializeOptimization();
    //         optimizer.optimize(10);
    //         cnt_outlier = 0;
    //         std::cout << "iteration=  " << iteration << std::endl;
    //         // features stores the pointer to features_left_

    //             const double chi2_th = 900;
    //             // count the outliers
    //             for (size_t i = 0; i < edges_direct.size(); ++i) {
    //                 auto e = edges_direct[i];
    //                 if (features[i]) {
    //                     e->computeError();
    //                 }
    //                 if (e->chi2() > chi2_th) {
    //                     features[i] = true;
    //                     e->setLevel(1); // do not optimize
    //                     cnt_outlier++;
    //                 } else {
    //                     features[i] = false;
    //                     e->setLevel(0); // optimize
    //                 };

    //                 if (iteration == 2) {
    //                     e->setRobustKernel(nullptr);
    //                 }
    //             }
    //     }
        
    //     for (size_t i = 0; i < edges_direct.size(); ++i) {
    //         auto e = edges_direct[i];
    //         // std::cout << "e->chi2() = " << e->chi2() << std::endl;
    //     }
    //     std::cout << "Outlier/Inlier in frontend pose estimating: " << cnt_outlier << "/"
    //           << features.size() - cnt_outlier << std::endl;
    //     Tcw = pose->estimate();






        std::cout << "finish direct pose estimation" << std::endl;
        // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        // chrono::duration<double> time_used_feature_extraction = chrono::duration_cast<chrono::duration<double>> ( t1-t0 );
        // cout<<"direct method feature extraction costs time: "<<time_used_feature_extraction.count() <<" seconds."<<endl;
        // cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;

        // plot the feature points
        // cv::Mat img_show ( gray.rows*2, gray.cols, CV_8UC3 );
        // cv::cvtColor ( prev_gray, prev_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor ( gray, color, cv::COLOR_GRAY2BGR);
        // prev_gray = gray.clone();
        // prev_gray_ori = gray_ori.clone();
        // prev_color.copyTo ( img_show ( cv::Rect ( 0,0,gray.cols, gray.rows ) ) );
        // color.copyTo ( img_show ( cv::Rect ( 0,gray.rows,gray.cols, gray.rows ) ) );
        for ( Measurement m:measurements ) {
            // if ( rand() > RAND_MAX/5 )
            //     continue;
            // Eigen::Vector3d p = m.pos_world;
            // Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=gray.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=gray.rows )
                continue;

            float b = 0;//255*float ( rand() ) /RAND_MAX;
            float g = 0;//255*float ( rand() ) /RAND_MAX;
            float r = 255;//*float ( rand() ) /RAND_MAX;
            // cv::circle ( color, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 2, cv::Scalar ( b,g,r ), 1 );
            cv::circle ( color, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 )), 2, cv::Scalar ( b,g,r ), 1 );
            // cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), cv::Scalar ( b,g,r ), 1 );
        }
        cv::imshow ( "result", color );
        cv::waitKey ( 0 );

    }
    return 0;
}

bool poseEstimationDirect ( const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolverType;
    // DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    std::unique_ptr<DirectBlock::LinearSolverType> linearSolver (new g2o::LinearSolverDense< DirectBlock::PoseMatrixType>());
    // DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    std::unique_ptr<DirectBlock> solver_ptr (new DirectBlock ( std::move(linearSolver)));
    // // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr) ); // L-M
    // auto solver = new g2o::OptimizationAlgorithmLevenberg(
    //             g2o::make_unique<DirectBlock>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( true );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate ( g2o::SE3Quat ( Tcw.rotation(), Tcw.translation() ) );
    pose->setId ( 0 );
    optimizer.addVertex ( pose );

    // 添加边
    int id=1;
    for ( Measurement m: measurements )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
            m.pos_world,
            K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), gray
        );
        edge->setVertex ( 0, pose );
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );
    }
    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 30 );
    std::cout << "0" << std::endl;
    Tcw = pose->estimate();
    std::cout << "1" << std::endl;
}

