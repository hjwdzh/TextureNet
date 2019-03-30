#include <igl/readOBJ.h>
#include <igl/point_mesh_squared_distance.h>
#include <fstream>
#include <map>
std::string property;
std::string label;
std::string barycentricfile;

float maplabel(Eigen::MatrixXf& V1, Eigen::MatrixXf& V2, Eigen::MatrixXf& TC2, Eigen::MatrixXi& F2, Eigen::MatrixXi& FTC2, std::vector<int>& l, int face_label) {
        Eigen::VectorXf sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXf C;
        igl::point_mesh_squared_distance(V1,V2,F2,sqrD,I,C);
        float dis = 0;
        std::vector<int> labels;
        std::vector<float> properties;
        for (int i = 0; i < sqrD.rows(); ++i) {
                Eigen::MatrixXf weight;
                int find = I[i];
                if (face_label == 0) {
                        igl::barycentric_coordinates(C.row(i), V2.row(F2(find, 0)), V2.row(F2(find, 1)), V2.row(F2(find, 2)), weight);
                        Eigen::Vector3f p0 = V2.row(F2(find, 0)) - C.row(i);
                        Eigen::Vector3f p1 = V2.row(F2(find, 1)) - C.row(i);
                        Eigen::Vector3f p2 = V2.row(F2(find, 2)) - C.row(i);
                        if (p0.norm() < p1.norm() && p0.norm() < p2.norm()) {
                                labels.push_back(l[F2(find, 0)]);
                        }
                        else if (p0.norm() > p1.norm() && p1.norm() < p2.norm()) {
                                labels.push_back(l[F2(find, 1)]);
                        } else {
                                labels.push_back(l[F2(find, 2)]);
                        }
                } else {
                        labels.push_back(l[find]);
                }
        }
        FILE* fp = fopen(label.c_str(), "wb");
        fwrite(labels.data(), sizeof(int), labels.size(), fp);
        fclose(fp);
        if (barycentricfile.size() > 0) {
                FILE* fp = fopen(barycentricfile.c_str(), "wb");
                int num = C.rows();
                fwrite(&num, sizeof(int), 1, fp);
                num = I.size();
                fwrite(&num, sizeof(int), 1, fp);
                fwrite(C.data(), sizeof(float), 3 * C.rows(), fp);
                fwrite(I.data(), sizeof(int), I.size(), fp);
                fclose(fp);
        }
        return dis;
}

int main(int argc, char** argv) {
        Eigen::MatrixXf V1, V2, TC1, TC2, N1, N2;
        Eigen::MatrixXi F1, F2, FTC1, FTC2, FN1, FN2;

        std::vector<Eigen::Vector3f> v;
        float x, y, z;
        std::ifstream is(argv[1]);
        while (is >> x) {
                is >> y >> z;
                v.push_back(Eigen::Vector3f(x, y, z));
        }
        is.close();
        V1.resize(v.size(), 3);
        for (int i = 0; i < v.size(); ++i) {
                V1.row(i) = v[i];
        }
        igl::readOBJ(argv[2], V2, TC2, N2, F2, FTC2, FN2);
        is.open(argv[3]);
        std::vector<int> labels;
        while (is >> x) {
                labels.push_back(x);
        }
        //property = argv[4];
        int face_label = 0;
        label = argv[4];
        if (argc > 5)
                barycentricfile = argv[5];
        if (argc > 6)
                face_label = 1;
        //color = cv::imread(std::string(argv[3]) + "/t000000.jpg");
        //category = cv::imread(std::string(argv[3]) + "/category.png", cv::IMREAD_UNCHANGED);
        float dis0 = maplabel(V1, V2, TC2, F2, FTC2, labels, face_label);
        return 0;
}