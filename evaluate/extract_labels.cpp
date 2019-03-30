#include <igl/readOBJ.h>
#include <igl/point_mesh_squared_distance.h>
#include <fstream>
#include <map>
#include <set>
#include <queue>
std::string property;
std::string label;
std::string barycentricfile;

int labels_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39};
int verify_map[40];

float maplabel(Eigen::MatrixXf& V1, Eigen::MatrixXf& V2, Eigen::MatrixXf& TC2, Eigen::MatrixXi& F2, Eigen::MatrixXi& FTC2, std::vector<int>& labels_gt, std::vector<int>& labels_pred, const char* output, const char* output_res) {
        Eigen::VectorXf sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXf C;
        igl::point_mesh_squared_distance(V1,V2,F2,sqrD,I,C);
        float dis = 0;
        std::vector<int> label_indices;
        for (int i = 0; i < sqrD.rows(); ++i) {
                Eigen::MatrixXf weight;
                int find = I[i];
                igl::barycentric_coordinates(C.row(i), V2.row(F2(find, 0)), V2.row(F2(find, 1)), V2.row(F2(find, 2)), weight);
                Eigen::Vector3f p0 = V2.row(F2(find, 0)) - C.row(i);
                Eigen::Vector3f p1 = V2.row(F2(find, 1)) - C.row(i);
                Eigen::Vector3f p2 = V2.row(F2(find, 2)) - C.row(i);
                if (p0.norm() < p1.norm() && p0.norm() < p2.norm()) {
                        label_indices.push_back(F2(find, 0));
                }
                else if (p0.norm() > p1.norm() && p1.norm() < p2.norm()) {
                        label_indices.push_back(F2(find, 1));
                } else {
                        label_indices.push_back(F2(find, 2));
                }
        }
        std::vector<double> distances(V2.rows(), 1e30);
        std::vector<int> vindices(V2.rows(), -1);
        for (int i = 0; i < label_indices.size(); ++i) {
                if (labels_pred[i] == 0)
                        continue;
                Eigen::Vector3f p = V2.row(label_indices[i]) - V1.row(i);
                double len = p.norm();
                if (len < distances[label_indices[i]]) {
                        vindices[label_indices[i]] = i;
                        distances[label_indices[i]] = len;
                }
        }

        std::vector<int> labels_extracted(labels_gt.size(), -1);
        for (int i = 0; i < vindices.size(); ++i) {
                if (vindices[i] == -1)
                        continue;
                labels_extracted[i] = labels_pred[vindices[i]];
        }
        std::vector<std::set<int> > links(V2.rows()), links_tmp;
        for (int i = 0; i < F2.rows(); ++i) {
                for (int j = 0; j < 3; ++j) {
                        int v1 = F2(i, j);
                        int v2 = F2(i, (j + 1) % 3);
                        links[v1].insert(v2);
                        links[v2].insert(v1);
                }
        }

        std::priority_queue<std::pair<float, std::pair<int, int> > > q;

        for (int i = 0; i < labels_extracted.size(); ++i) {
                if (labels_extracted[i] != -1) {
                        int expand = 0;
                        for (auto l : links[i]) {
                                if (labels_extracted[l] == -1) {
                                        Eigen::Vector3f off = V2.row(i) - V2.row(l);
                                        double dis = off.norm();
                                        if (dis < distances[l]) {
                                                distances[l] = dis;
                                                q.push(std::make_pair(-dis, std::make_pair(l, labels_extracted[i])));
                                        }
                                }
                        }
                }
        }
        while (!q.empty()) {
                auto info = q.top();
                q.pop();
                if (labels_extracted[info.second.first] != -1)
                        continue;
                labels_extracted[info.second.first] = info.second.second;
                for (auto l : links[info.second.first]) {
                        if (labels_extracted[l] == -1) {
                                Eigen::Vector3f off = V2.row(info.second.first) - V2.row(l);
                                double dis = off.norm() - info.first;
                                if (dis < distances[l]) {
                                        distances[l] = dis;
                                        q.push(std::make_pair(-dis, std::make_pair(l, info.second.second)));
                                }
                        }
                }
        }

        std::vector<int> confusion(21 * 21);
        for (int i = 0; i < labels_extracted.size(); ++i) {
                if (labels_gt[i] >= 40) {
                        labels_gt[i] = 0;
                }
                int gt_val = verify_map[labels_gt[i]];
                int pred_val = labels_extracted[i];
                if (gt_val == 0)
                        continue;
                confusion[gt_val * 21 + pred_val] += 1;
        }
        printf("%s\n", output);
        std::ofstream os(output);
        for (int i = 0; i < 21; ++i) {
                for (int j = 0; j < 21; ++j) {
                        os << confusion[i * 21 + j] << " ";
                }
                os << "\n";
        }
        os.close();

        os.open(output_res);
        for (int i = 0; i < labels_extracted.size(); ++i) {
                os << labels_map[labels_extracted[i]] << "\n";
        }
        os.close();
/*
        for i in range(gt_ids.shape[0]):
                gt_val = gt_ids[i]
                pred_val = pred_ids[i]
                if gt_val not in VALID_CLASS_IDS:
                        continue
                if pred_val not in VALID_CLASS_IDS:
                        pred_val = UNKNOWN_ID
                confusion[gt_val][pred_val] += 1
                total_seen_class[gt_val] += 1
                total_union_class[gt_val] += 1
                if gt_val == pred_val:
                        total_correct_class[gt_val] += 1
                else:
                        total_union_class[pred_val] += 1
*/
        return dis;
}

int main(int argc, char** argv) {
        memset(verify_map, 0, sizeof(int) * 40);
        for (int i = 0; i < 21; ++i) {
                verify_map[labels_map[i]] = i;
        }
        Eigen::MatrixXf V1, V2, TC1, TC2, N1, N2;
        Eigen::MatrixXi F1, F2, FTC1, FTC2, FN1, FN2;

        std::vector<Eigen::Vector3f> v;
        std::vector<int> labels_pred;
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
        labels_pred.resize(v.size());
        is.open(argv[2]);
        int label;
        for (int i = 0; i < labels_pred.size(); ++i) {
                is >> labels_pred[i];
        }
        is.close();
        igl::readOBJ(argv[3], V2, TC2, N2, F2, FTC2, FN2);
        is.open(argv[4]);
        std::vector<int> labels;
        while (is >> x) {
                labels.push_back(x);
        }
        maplabel(V1, V2, TC2, F2, FTC2, labels, labels_pred, argv[5], argv[6]);
        return 0;
}