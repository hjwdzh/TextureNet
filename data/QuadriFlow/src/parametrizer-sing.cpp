#include "config.hpp"
#include "field-math.hpp"
#include "parametrizer.hpp"
#include <queue>
#include <iostream>

void Parametrizer::ComputeOrientationSingularities() {
    MatrixXd &N = hierarchy.mN[0], &Q = hierarchy.mQ[0];
    const MatrixXi& F = hierarchy.mF;
    singularities.clear();
    for (int f = 0; f < F.cols(); ++f) {
        int index = 0;
        int abs_index = 0;
        for (int k = 0; k < 3; ++k) {
            int i = F(k, f), j = F(k == 2 ? 0 : (k + 1), f);
            auto value =
                compat_orientation_extrinsic_index_4(Q.col(i), N.col(i), Q.col(j), N.col(j));
            index += value.second - value.first;
            abs_index += std::abs(value.second - value.first);
        }
        int index_mod = modulo(index, 4);
        if (index_mod == 1 || index_mod == 3) {
            if (index >= 4 || index < 0) {
                Q.col(F(0, f)) = -Q.col(F(0, f));
            }
            singularities[f] = index_mod;
        }
    }
}

void Parametrizer::ComputePositionSingularities() {
    const MatrixXd &V = hierarchy.mV[0], &N = hierarchy.mN[0], &Q = hierarchy.mQ[0],
                   &O = hierarchy.mO[0];
    const MatrixXi& F = hierarchy.mF;

    pos_sing.clear();
    pos_rank.resize(F.rows(), F.cols());
    pos_index.resize(6, F.cols());
    for (int f = 0; f < F.cols(); ++f) {
        Vector2i index = Vector2i::Zero();
        uint32_t i0 = F(0, f), i1 = F(1, f), i2 = F(2, f);

        Vector3d q[3] = {Q.col(i0).normalized(), Q.col(i1).normalized(), Q.col(i2).normalized()};
        Vector3d n[3] = {N.col(i0), N.col(i1), N.col(i2)};
        Vector3d o[3] = {O.col(i0), O.col(i1), O.col(i2)};
        Vector3d v[3] = {V.col(i0), V.col(i1), V.col(i2)};

        int best[3];
        double best_dp = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i) {
            Vector3d v0 = rotate90_by(q[0], n[0], i);
            for (int j = 0; j < 4; ++j) {
                Vector3d v1 = rotate90_by(q[1], n[1], j);
                for (int k = 0; k < 4; ++k) {
                    Vector3d v2 = rotate90_by(q[2], n[2], k);
                    double dp = std::min(std::min(v0.dot(v1), v1.dot(v2)), v2.dot(v0));
                    if (dp > best_dp) {
                        best_dp = dp;
                        best[0] = i;
                        best[1] = j;
                        best[2] = k;
                    }
                }
            }
        }
        pos_rank(0, f) = best[0];
        pos_rank(1, f) = best[1];
        pos_rank(2, f) = best[2];
        for (int k = 0; k < 3; ++k) q[k] = rotate90_by(q[k], n[k], best[k]);

        for (int k = 0; k < 3; ++k) {
            int kn = k == 2 ? 0 : (k + 1);
            double scale_x = hierarchy.mScale, scale_y = hierarchy.mScale,
                   scale_x_1 = hierarchy.mScale, scale_y_1 = hierarchy.mScale;
            if (flag_adaptive_scale) {
                scale_x *= hierarchy.mS[0](0, F(k, f));
                scale_y *= hierarchy.mS[0](1, F(k, f));
                scale_x_1 *= hierarchy.mS[0](0, F(kn, f));
                scale_y_1 *= hierarchy.mS[0](1, F(kn, f));
                if (best[k] % 2 != 0) std::swap(scale_x, scale_y);
                if (best[kn] % 2 != 0) std::swap(scale_x_1, scale_y_1);
            }
            double inv_scale_x = 1.0 / scale_x, inv_scale_y = 1.0 / scale_y,
                   inv_scale_x_1 = 1.0 / scale_x_1, inv_scale_y_1 = 1.0 / scale_y_1;
            std::pair<Vector2i, Vector2i> value = compat_position_extrinsic_index_4(
                v[k], n[k], q[k], o[k], v[kn], n[kn], q[kn], o[kn], scale_x, scale_y, inv_scale_x,
                inv_scale_y, scale_x_1, scale_y_1, inv_scale_x_1, inv_scale_y_1, nullptr);
            auto diff = value.first - value.second;
            index += diff;
            pos_index(k * 2, f) = diff[0];
            pos_index(k * 2 + 1, f) = diff[1];
        }

        if (index != Vector2i::Zero()) {
            pos_sing[f] = rshift90(index, best[0]);
        }
    }
}

void Parametrizer::AnalyzeValence() {
    auto& F = hierarchy.mF;
    std::map<int, int> sing;
    for (auto& f : singularities) {
        for (int i = 0; i < 3; ++i) {
            sing[F(i, f.first)] = f.second;
        }
    }
    auto& F2E = face_edgeIds;
    auto& E2E = hierarchy.mE2E;
    auto& FQ = face_edgeOrients;
    std::set<int> sing1, sing2;
    for (int i = 0; i < F2E.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int deid = i * 3 + j;
            int sum_int = 0;
            std::vector<int> edges;
            std::vector<double> angles;
            do {
                int deid1 = deid / 3 * 3 + (deid + 2) % 3;
                deid = E2E[deid1];
                sum_int += (FQ[deid / 3][deid % 3] + 6 - FQ[deid1 / 3][deid1 % 3]) % 4;
            } while (deid != i * 3 + j);
            if (sum_int % 4 == 2) {
                printf("OMG! valence = 2\n");
                exit(0);
            }
            if (sum_int % 4 == 1) sing1.insert(F(j, i));
            if (sum_int % 4 == 3) sing2.insert(F(j, i));
        }
    }
    int count3 = 0, count4 = 0;
    for (auto& s : singularities) {
        if (s.second == 1)
            count3 += 1;
        else
            count4 += 1;
    }
    printf("singularity: <%d %d> <%d %d>\n", (int)sing1.size(), (int)sing2.size(), count3, count4);
}

void Parametrizer::Downsample(std::vector<std::pair<Vector4i, Vector4i> >& entries, std::vector<Vector3d>& textiles_positions,
        std::vector<std::pair<Vector4i, Vector4i> >& new_entries, std::vector<Vector3d>& new_textiles_positions,
        std::vector<int>& groups, std::vector<int>& groups_orient_diff,
        std::vector<int>& seeds) {
    new_entries.clear();
    new_textiles_positions.clear();
    groups.clear();
    groups.resize(entries.size(), -1);
    groups_orient_diff.clear();
    groups_orient_diff.resize(entries.size(), 0);
    seeds.clear();
    std::queue<std::pair<int, int> > q;
    int group_id = 0;

    for (int patch_size = 4; patch_size >= 1; --patch_size) {
        for (int i = 0; i < entries.size(); ++i) {
            if (groups[i] != -1)
                continue;
            int id = i;
            int dir = 0;
            int odir = dir;
            std::map<int, int> ids;
            for (int t = 0; t < 4; ++t) {
                if (groups[id] == -1) {
                    ids[id] = (odir - dir + 4) % 4;
                }
                int id_next = entries[id].first[dir];
                if (id_next == -1)
                    break;
                dir = (dir + entries[id].second[dir] + 1) % 4;
                id = id_next;
            }
            if (ids.size() < patch_size)
                continue;
            for (auto& id : ids) {
                groups[id.first] = group_id;
                groups_orient_diff[id.first] = id.second;
            }
            seeds.push_back(i);
            q.push(std::make_pair(i, 0));
            group_id += 1;
            while (!q.empty()) {
                auto s = q.front();
                q.pop();
                for (int t = 0; t < 4; ++t) {
                    int id = s.first;
                    int dir = s.second;
                    for (int j = 0; j < 2; ++j) {
                        int id_next = entries[id].first[(dir + t) % 4];
                        if (id_next == -1) {
                            id = -1;
                            break;
                        }
                        dir = (dir + entries[id].second[(dir + t) % 4]) % 4;
                        id = id_next;
                    }
                    if (id == -1) {
                        continue;
                    }
                    if (groups[id] == -1) {
                        std::map<int, int> ids;
                        int oid = id;
                        int odir = dir;
                        for (int t = 0; t < 4; ++t) {
                            if (groups[id] == -1) {
                                ids[id] = (odir - dir + 4) % 4;
                            }
                            int id_next = entries[id].first[dir];
                            if (id_next == -1)
                                break;
                            dir = (dir + entries[id].second[dir] + 1) % 4;
                            id = id_next;
                        }
                        if (ids.size() < patch_size) {
                            continue;
                        }
                        for (auto& id : ids) {
                            groups[id.first] = group_id;
                            groups_orient_diff[id.first] = id.second;
                        }
                        seeds.push_back(oid);
                        q.push(std::make_pair(oid, odir));
                        group_id += 1;
                    }
                }
            }
        }
    }

    for (int i = 0; i < seeds.size(); ++i) {
        new_textiles_positions.push_back(textiles_positions[seeds[i]]);
    }
    new_entries.resize(seeds.size());
    for (int i = 0; i < seeds.size(); ++i) {
        new_entries[i] = std::make_pair(Vector4i(-1, -1, -1, -1), Vector4i(-1, -1, -1, -1));
    }
    for (int i = 0; i < seeds.size(); ++i) {
        int vert = seeds[i];
        for (int dir = 0; dir < 4; ++dir) {
            int id = vert;
            int odir = dir;
            int ndir = dir;
            for (int j = 0; j < 2; ++j) {
                int id_next = entries[id].first[ndir];
                if (id_next == -1)
                    break;
                ndir = (ndir + entries[id].second[ndir]) % 4;
                id = id_next;
                if (groups[id] != i) {
                    new_entries[i].first[dir] = groups[id];
                    new_entries[i].second[dir] = (ndir - odir + groups_orient_diff[id] + 4) % 4;
                }
            }
        }
    }
}


void Parametrizer::QuantizeTextiles() {
    MatrixXd &V = hierarchy.mV[0], &N = hierarchy.mN[0], &Q = hierarchy.mQ[0],
                   &O = hierarchy.mO[0];
    const MatrixXi& F = hierarchy.mF;

    pos_sing.clear();
    pos_rank.resize(F.rows(), F.cols());
    pos_index.resize(6, F.cols());

    DisajointTree tree(V.cols());
    for (int f = 0; f < F.cols(); ++f) {
        Vector2i index = Vector2i::Zero();
        uint32_t i0 = F(0, f), i1 = F(1, f), i2 = F(2, f);

        Vector3d q[3] = {Q.col(i0).normalized(), Q.col(i1).normalized(), Q.col(i2).normalized()};
        Vector3d n[3] = {N.col(i0), N.col(i1), N.col(i2)};
        Vector3d o[3] = {O.col(i0), O.col(i1), O.col(i2)};
        Vector3d v[3] = {V.col(i0), V.col(i1), V.col(i2)};

        int best[3];
        double best_dp = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i) {
            Vector3d v0 = rotate90_by(q[0], n[0], i);
            for (int j = 0; j < 4; ++j) {
                Vector3d v1 = rotate90_by(q[1], n[1], j);
                for (int k = 0; k < 4; ++k) {
                    Vector3d v2 = rotate90_by(q[2], n[2], k);
                    double dp = std::min(std::min(v0.dot(v1), v1.dot(v2)), v2.dot(v0));
                    if (dp > best_dp) {
                        best_dp = dp;
                        best[0] = i;
                        best[1] = j;
                        best[2] = k;
                    }
                }
            }
        }
        for (int k = 0; k < 3; ++k) q[k] = rotate90_by(q[k], n[k], best[k]);

        pos_rank(0, f) = best[0];
        pos_rank(1, f) = best[1];
        pos_rank(2, f) = best[2];

        for (int k = 0; k < 3; ++k) {
            int kn = k == 2 ? 0 : (k + 1);
            double scale_x = hierarchy.mScale, scale_y = hierarchy.mScale,
                   scale_x_1 = hierarchy.mScale, scale_y_1 = hierarchy.mScale;
            if (flag_adaptive_scale) {
                scale_x *= hierarchy.mS[0](0, F(k, f));
                scale_y *= hierarchy.mS[0](1, F(k, f));
                scale_x_1 *= hierarchy.mS[0](0, F(kn, f));
                scale_y_1 *= hierarchy.mS[0](1, F(kn, f));
                if (best[k] % 2 != 0) std::swap(scale_x, scale_y);
                if (best[kn] % 2 != 0) std::swap(scale_x_1, scale_y_1);
            }
            double inv_scale_x = 1.0 / scale_x, inv_scale_y = 1.0 / scale_y,
                   inv_scale_x_1 = 1.0 / scale_x_1, inv_scale_y_1 = 1.0 / scale_y_1;
            std::pair<Vector2i, Vector2i> value = compat_position_extrinsic_index_4(
                v[k], n[k], q[k], o[k], v[kn], n[kn], q[kn], o[kn], scale_x, scale_y, inv_scale_x,
                inv_scale_y, scale_x_1, scale_y_1, inv_scale_x_1, inv_scale_y_1, nullptr);
            auto diff = value.first - value.second;
            index += diff;
            pos_index(k * 2, f) = diff[0];
            pos_index(k * 2 + 1, f) = diff[1];
        }

        for (int k = 0; k < 3; ++k) {
            if (pos_index(k * 2, f) == 0 && pos_index(k * 2 + 1, f) == 0) {
                tree.Merge(F(k, f), F((k + 1) % 3, f));
            }
        }
    }

    tree.BuildCompactParent();

    std::vector<int> occupied_grids(tree.CompactNum(), -1);

    std::vector<Eigen::Vector3d> positions(tree.CompactNum());
    for (int i = 0; i < positions.size(); ++i) {
        positions[i] = Eigen::Vector3d(0, 0, 0);
    }

    std::vector<int> counts(positions.size(), 0);
    std::set<std::pair<int, int> > lines;
    std::vector<Eigen::Vector3d> orientations(tree.CompactNum());
    std::vector<Eigen::Vector3d> normals(tree.CompactNum());
    std::vector<double> distance(tree.CompactNum(), 1e30);
    for (int i = 0; i < V.cols(); ++i) {
        int index = tree.Index(i);
        positions[index] += O.col(i);
        counts[index] += 1;
    }

    for (int i = 0; i < positions.size(); ++i) {
        positions[i] /= counts[i];
    }

    for (int i = 0; i < V.cols(); ++i) {
        int index = tree.Index(i);
        Eigen::Vector3d dis = O.col(i) - positions[index];
        double d = dis.norm();
        if (d < distance[index]) {
            distance[index] = d;
            orientations[index] = Q.col(i);
            normals[index] = N.col(i);
        }
    }

    std::vector<Vector3d> textiles_positions, new_textiles_positions, textiles_orientations, textiles_normals;
    std::vector<int> textiles_faces;
    std::vector<std::pair<Vector4i, Vector4i> > entries, new_entries;
    for (int f = 0; f < F.cols(); ++f) {
        int grid_index = tree.Index(F(0, f));
        if (occupied_grids[grid_index] != -1) {
            continue;
        }

        uint32_t i0 = F(0, f), i1 = F(1, f), i2 = F(2, f);

        Vector3d q[3] = {Q.col(i0).normalized(), Q.col(i1).normalized(), Q.col(i2).normalized()};
        Vector3d n[3] = {N.col(i0), N.col(i1), N.col(i2)};
        Vector3d o[3] = {O.col(i0), O.col(i1), O.col(i2)};
        Vector3d v[3] = {V.col(i0), V.col(i1), V.col(i2)};

        int best[3];
        double best_dp = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i) {
            Vector3d v0 = rotate90_by(q[0], n[0], i);
            for (int j = 0; j < 4; ++j) {
                Vector3d v1 = rotate90_by(q[1], n[1], j);
                for (int k = 0; k < 4; ++k) {
                    Vector3d v2 = rotate90_by(q[2], n[2], k);
                    double dp = std::min(std::min(v0.dot(v1), v1.dot(v2)), v2.dot(v0));
                    if (dp > best_dp) {
                        best_dp = dp;
                        best[0] = i;
                        best[1] = j;
                        best[2] = k;
                    }
                }
            }
        }
        for (int k = 0; k < 3; ++k) q[k] = rotate90_by(q[k], n[k], best[k]);

        Vector2i diff[3];
        diff[0] = Vector2i(0, 0);
        diff[1] = Vector2i(pos_index(0, f), pos_index(1, f));
        diff[2] = Vector2i(-pos_index(4, f), -pos_index(5, f));
        Vector2d tex[3];
        tex[0] = Vector2d(0, 0);
        tex[1] = Vector2d(pos_index(0, f), pos_index(1, f));
        tex[2] = Vector2d(-pos_index(4, f), -pos_index(5, f));

        double tx_min = 1e30, tx_max = -1e30, ty_min = 1e30, ty_max = -1e30;
        for (int k = 0; k < 3; ++k) {
            tex[k][0] += (v[k] - o[k]).dot(q[k]);
            tex[k][1] += (v[k] - o[k]).dot(rotate90_by(q[k], n[k], 1));
            tx_min = std::min(tx_min, tex[k][0]);
            tx_max = std::max(tx_max, tex[k][0]);
            ty_min = std::min(ty_min, tex[k][1]);
            ty_max = std::max(ty_max, tex[k][1]);
        }

        Eigen::MatrixXd invt(2, 2);
        invt(0, 0) = tex[1][0] - tex[0][0];
        invt(1, 0) = tex[1][1] - tex[0][1];
        invt(0, 1) = tex[2][0] - tex[0][0];
        invt(1, 1) = tex[2][1] - tex[0][1];
        invt = invt.inverse().eval();

        Vector2d t(0 - tex[0][0], 0 - tex[0][1]);
        t = invt * t;
        if (t[0] >= 0 && t[1] >= 0 && t[0] + t[1] <= 1) {
            // make sure there is no conflict
            int grid_num = textiles_positions.size();
            Vector3d p = o[0];
            textiles_positions.push_back(positions[tree.Index(i0)]);
            textiles_orientations.push_back(orientations[tree.Index(i0)]);
            textiles_normals.push_back(normals[tree.Index(i0)]);
            entries.push_back(std::make_pair(Vector4i(-1, -1, -1, -1),Vector4i(-1, -1, -1, -1)));
            textiles_faces.push_back(f);
            occupied_grids[grid_index] = grid_num;
        }
    }

    int consistent_edges = 0;
    int total_edges = 0;
    int boundary_edges = 0;

    for (int i = 0; i < F.cols(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int vid1 = F(j, i);
            int vid2 = F((j + 1) % 3, i);
            int index = tree.Index(vid1);
            int index1 = tree.Index(vid2);
            if (index == index1)
                continue;
            if (occupied_grids[index] == -1 || occupied_grids[index1] == -1)
                continue;

            Vector2i diff(pos_index(j * 2, i), pos_index(j * 2 + 1, i));
            int vref = F(0, textiles_faces[occupied_grids[index]]);
            auto orients = compat_orientation_extrinsic_index_4(Q.col(vid1), N.col(vid1), Q.col(vref), N.col(vref));
            vref = F(0, textiles_faces[occupied_grids[index1]]);
            auto orients1 = compat_orientation_extrinsic_index_4(Q.col(vid2), N.col(vid2), Q.col(vref), N.col(vref));

            diff = rshift90(diff, (pos_rank(j, i) + orients.second - orients.first + 4) % 4);
            int orient_diff = (pos_rank((j + 1) % 3, i) - pos_rank(j, i) + 12 + orients1.second - orients.second + orients.first - orients1.first) % 4;
            if (diff[0] == 1 && diff[1] == 0) {
                if (entries[occupied_grids[index]].first[0] == -1) {
                    entries[occupied_grids[index]].first[0] = occupied_grids[index1];
                    entries[occupied_grids[index]].second[0] = orient_diff;
                }
                else {
                    if (entries[occupied_grids[index]].first[0] == occupied_grids[index1]) {
                        consistent_edges += 1;
                    }
                    total_edges += 1;
                }
            }
            if (diff[0] == 0 && diff[1] == 1) {
                if (entries[occupied_grids[index]].first[1] == -1) {
                    entries[occupied_grids[index]].first[1] = occupied_grids[index1];
                    entries[occupied_grids[index]].second[1] = orient_diff;
                }
                else {
                    if (entries[occupied_grids[index]].first[1] == occupied_grids[index1]) {
                        consistent_edges += 1;
                    }
                    total_edges += 1;
                }
            }
            if (diff[0] == -1 && diff[1] == 0) {
                if (entries[occupied_grids[index]].first[2] == -1) {
                    entries[occupied_grids[index]].first[2] = occupied_grids[index1];
                    entries[occupied_grids[index]].second[2] = orient_diff;
                }
                else {
                    if (entries[occupied_grids[index]].first[2] == occupied_grids[index1]) {
                        consistent_edges += 1;
                    }
                    total_edges += 1;
                }
            }
            if (diff[0] == 0 && diff[1] == -1) {
                if (entries[occupied_grids[index]].first[3] == -1) {
                    entries[occupied_grids[index]].first[3] = occupied_grids[index1];
                    entries[occupied_grids[index]].second[3] = orient_diff;
                }
                else {
                    if (entries[occupied_grids[index]].first[3] == occupied_grids[index1]) {
                        consistent_edges += 1;
                    }
                    total_edges += 1;
                }
            }
        }
    }

    for (int i = 0; i < entries.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            int nv = entries[i].first[j];
            if (nv == -1) {
                boundary_edges += 1;
                continue;
            }
        }
    }

    int num_levels = this->num_levels;
    int kernel_size = this->kernel_size;
    int rosy = this->rosy;
    printf("num_levels = %d, kernel_size = %d, rosy = %d, gen_textiles = %d, gen_pool = %d, gen_dialation = %d\n", num_levels, kernel_size, rosy, (int)gen_textiles, (int)gen_pool, (int)gen_dialation);
    if (this->gen_textiles) {
    std::ofstream os(position_file);
    for (int i = 0; i < textiles_positions.size(); ++i) {
        auto t = this->normalize_scale * textiles_positions[i] + this->normalize_offset;
        os << t[0] << " " << t[1] << " " << t[2] << "\n";
    }
    os.close();
    }
    if (this->gen_frames) {
    printf("%d %d %d\n", textiles_positions.size(), orientations.size(), normals.size());
    std::ofstream os(frames_file);
    for (int i = 0; i < textiles_positions.size(); ++i) {
        os << textiles_orientations[i][0] << " " << textiles_orientations[i][1] << " " << textiles_orientations[i][2] << " "
        << textiles_normals[i][0] << " " << textiles_normals[i][1] << " " << textiles_normals[i][2] << "\n";
    }
    os.close();
    printf("Finish...\n");
    }        
    
    if (this->gen_entry) {
    printf("Genentry... file %s\n", entry_file.c_str());
    FILE* fp = fopen(entry_file.c_str(), "wb");
    int num = entries.size();
    fwrite(&num, sizeof(int), 1, fp);
    for (int i = 0; i < entries.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            fwrite(&(entries[i].first[j]), sizeof(int), 1, fp);
        }
        for (int j = 0; j < 4; ++j) {
            fwrite(&(entries[i].second[j]), sizeof(int), 1, fp);
        }
    }
    fclose(fp);
    }
    FILE* fp = fopen(indices_file.c_str(), "wb");
    fwrite(&num_levels, 1, sizeof(int), fp);
    if (this->gen_dialation) { 
    std::vector<std::vector<int> > level_conv_indices(num_levels);
    for (int i = 0; i < num_levels; ++i) {
        level_conv_indices[i].resize(kernel_size * kernel_size * rosy * entries.size(), -1);
    }
    for (int i = 0; i < entries.size(); ++i) {
	   struct VisitInfo
        {
            VisitInfo()
            {}
            VisitInfo(int x, int y, int index, int start_dir)
            : x(x), y(y), index(index), start_dir(start_dir)
            {}
            int x;
            int y;
            int index;
            int start_dir;
        };
        for (int dir = 0; dir < rosy; ++dir) {
            int stride = 1 << (num_levels - 1);
            int ns = stride * (kernel_size - 1) + 1;
            std::queue<VisitInfo> q;
	        std::vector<int> hash(ns*ns, 0);
            q.push(VisitInfo(ns / 2, ns / 2, i, dir));
            hash[ns / 2 * ns + ns / 2] = 1;
            std::vector<int> conv_indices(ns * ns, -1);
            conv_indices[ns / 2 * ns + ns / 2] = i * rosy + dir % rosy;
            int dx[] = {1, 0, -1, 0};
            int dy[] = {0, 1, 0, -1};
            while (!q.empty()) {
                auto info = q.front();
                q.pop();
                int index = info.index;
                for (int j = 0; j < 4; ++j) {
                    int next_index = entries[index].first[(info.start_dir + j) % 4];
                    if (next_index == -1) {
                        continue;
                    }
                    int tx = info.x + dx[j];
                    int ty = info.y + dy[j];
                    int potential_dis = hash[info.x * ns + info.y] + 1;
                    int ideal_dis = std::abs(tx - ns / 2) + std::abs(ty - ns / 2);
                    if (tx < 0 || ty < 0 || tx >= ns || ty >= ns || hash[tx * ns + ty] || potential_dis > ideal_dis + 2)
                        continue;
                    int next_dir = (info.start_dir + entries[index].second[(info.start_dir + j) % 4]) % 4;
                    q.push(VisitInfo(tx, ty, next_index, next_dir));
                    hash[tx * ns + ty] = hash[info.x * ns + info.y] + 1;
                    conv_indices[tx * ns + ty] = next_index * rosy + next_dir % rosy;
                }
            }
            
            memset(hash.data(), 0, sizeof(int) * ns * ns);
            hash[ns/2 * ns + ns / 2] = 1;
            q.push(VisitInfo(ns/2, ns/2, 0, 0));
            while (!q.empty()) {
                auto info = q.front();
                q.pop();
                for (int j = 0; j < 4; ++j) {
                    int tx = info.x + dx[j];
                    int ty = info.y + dy[j];
                    if (tx < 0 || ty < 0 || tx >= ns || ty >= ns || hash[tx * ns + ty])
                        continue;
                    q.push(VisitInfo(tx, ty, 0, 0));
                    hash[tx * ns + ty] = 1;
                    if (conv_indices[tx * ns + ty] == -1)
                        conv_indices[tx * ns + ty] = conv_indices[info.x * ns + info.y];
                }
            }
            
            int step = 1;
            for (int level = 0; level < num_levels; ++level) {
                int offset = (i * rosy + dir) * kernel_size * kernel_size;
		        for (int x = -kernel_size / 2; x <= kernel_size / 2; ++x) {
                    for (int y = -kernel_size / 2; y <= kernel_size / 2; ++y) {
                        int loc = (x + kernel_size / 2) * kernel_size + y + kernel_size / 2;
                        int convloc = (ns / 2 + x * step) * ns + (ns / 2 + y * step);
                        level_conv_indices[level][offset + loc] = conv_indices[convloc];
                    }
                }
                step *= 2;
            }
	        continue;
        }
    }
    for (int i = 0; i < num_levels; ++i) {
        fwrite(level_conv_indices[i].data(), sizeof(int), level_conv_indices[i].size(), fp);
    }
    fclose(fp);
    }
    if (this->gen_pool) {
    for (int level = 0; level < num_levels; ++level) {
        printf("level %d %d...\n", level, num_levels);
        struct VisitInfo
        {
            VisitInfo()
            {}
            VisitInfo(int x, int y, int index, int start_dir)
            : x(x), y(y), index(index), start_dir(start_dir)
            {}
            int x;
            int y;
            int index;
            int start_dir;
        };
        printf("Conv...\n");
        std::vector<int> conv_indices(kernel_size * kernel_size * rosy * entries.size(), -1);
        int offset = 0;
        for (int i = 0; i < entries.size(); ++i) {
            for (int dir = 0; dir < rosy; ++dir) {
                std::queue<VisitInfo> q;
        		std::vector<int> hash(kernel_size * kernel_size, 0);
                q.push(VisitInfo(kernel_size/2, kernel_size/2, i, dir));
                hash[kernel_size/2 * kernel_size + kernel_size/2] = 1;
                conv_indices[offset + kernel_size/2 * kernel_size + kernel_size/2] = i * rosy + dir;
                int dx[] = {1, 0, -1, 0};
                int dy[] = {0, 1, 0, -1};
                while (!q.empty()) {
                    auto info = q.front();
                    q.pop();
                    int index = info.index;
                    for (int j = 0; j < rosy; ++j) {
                        int next_index = entries[index].first[(info.start_dir + j) % 4];
                        if (next_index == -1) {
                            continue;
                        }
                        int tx = info.x + dx[j];
                        int ty = info.y + dy[j];
                        if (tx < 0 || ty < 0 || tx >= kernel_size || ty >= kernel_size || hash[tx * kernel_size + ty])
                            continue;
                        int next_dir = (info.start_dir + entries[index].second[(info.start_dir + j) % 4]) % 4;
                        q.push(VisitInfo(tx, ty, next_index, next_dir));
                        hash[tx * kernel_size + ty] = 1;
                        conv_indices[offset + tx * kernel_size + ty] = next_index * rosy + next_dir % rosy;
                    }
                }
                hash.clear();
        		hash.resize(kernel_size * kernel_size, 0);
                q.push(VisitInfo(kernel_size/2, kernel_size/2, i, dir));
                hash[kernel_size/2 * kernel_size + kernel_size/2] = 1;
                while (!q.empty()) {
                    auto info = q.front();
                    q.pop();
                    for (int j = 0; j < rosy; ++j) {
                        int tx = info.x + dx[j];
                        int ty = info.y + dy[j];
                        if (tx < 0 || ty < 0 || tx >= kernel_size || ty >= kernel_size || hash[tx * kernel_size + ty])
                            continue;
                        q.push(VisitInfo(tx, ty, 0, 0));
                        hash[tx * kernel_size + ty] = 1;
                        if (conv_indices[offset + tx * kernel_size + ty] == -1)
                            conv_indices[offset + tx * kernel_size + ty] = conv_indices[offset + info.x * kernel_size + info.y];
                    }
                }
                offset += kernel_size * kernel_size;
            }
        }
        /*
        printf("Out...\n");
        char buffer[256];
        sprintf(buffer, "result%d.obj", level);
        std::ofstream os(buffer);
        for (int i = 0; i < textiles_positions.size(); ++i) {
            Eigen::Vector3d pos = this->normalize_scale * textiles_positions[i] + this->normalize_offset;
            os << "v " << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
        }
        for (int i = 0; i < entries.size(); ++i) {
            for (int j = 0; j < 4; ++j) {
                if (entries[i].first[j] == -1)
                    continue;
                os << "l " << i + 1 << " " << entries[i].first[j] + 1 << "\n";
            }
        }
        os.close();
        */

        std::vector<int> groups, groups_orient_diff;
        std::vector<int> seeds;
        printf("Downsample...\n");
        Downsample(entries, textiles_positions, new_entries, new_textiles_positions, groups, groups_orient_diff, seeds);
        std::vector<int> pool_indices(entries.size() * rosy);
        for (int i = 0; i < groups.size(); ++i) {
            for (int dir = 0; dir < rosy; ++dir) {
                int next_index = groups[i] * rosy + (groups_orient_diff[i] + dir) % rosy;
                pool_indices[i * rosy + dir] = next_index;
            }
        }

        int num = entries.size() * rosy;
        fwrite(&num, sizeof(int), 1, fp);
        num = new_entries.size() * rosy;
        fwrite(&num, sizeof(int), 1, fp);
        fwrite(conv_indices.data(), sizeof(int), conv_indices.size(), fp);
        fwrite(pool_indices.data(), sizeof(int), pool_indices.size(), fp);

        std::swap(entries, new_entries);
        std::swap(textiles_positions, new_textiles_positions);
        printf("Finish...\n");
    }
    fclose(fp);
    }
}

void Parametrizer::ExtractGroups(std::vector<std::pair<std::pair<int, double>, Eigen::Vector2d> >& groups,
    std::vector<int>& min_id,
    std::vector<Eigen::Vector3d>& positions,
    std::vector<std::pair<int, int> >& conv_indices,
    int approx_dis) {
    MatrixXd &V = hierarchy.mV[0], &N = hierarchy.mN[0], &Q = hierarchy.mQ[0],
                   &O = hierarchy.mO[0];
    const MatrixXi& F = hierarchy.mF;
    pos_sing.clear();
    pos_rank.resize(F.rows(), F.cols());
    pos_index.resize(6, F.cols());
    DisajointTree tree(V.cols());
    for (int f = 0; f < F.cols(); ++f) {
        Vector2i index = Vector2i::Zero();
        uint32_t i0 = F(0, f), i1 = F(1, f), i2 = F(2, f);

        Vector3d q[3] = {Q.col(i0).normalized(), Q.col(i1).normalized(), Q.col(i2).normalized()};
        Vector3d n[3] = {N.col(i0), N.col(i1), N.col(i2)};
        Vector3d o[3] = {O.col(i0), O.col(i1), O.col(i2)};
        Vector3d v[3] = {V.col(i0), V.col(i1), V.col(i2)};

        int best[3];
        double best_dp = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < 4; ++i) {
            Vector3d v0 = rotate90_by(q[0], n[0], i);
            for (int j = 0; j < 4; ++j) {
                Vector3d v1 = rotate90_by(q[1], n[1], j);
                for (int k = 0; k < 4; ++k) {
                    Vector3d v2 = rotate90_by(q[2], n[2], k);
                    double dp = std::min(std::min(v0.dot(v1), v1.dot(v2)), v2.dot(v0));
                    if (dp > best_dp) {
                        best_dp = dp;
                        best[0] = i;
                        best[1] = j;
                        best[2] = k;
                    }
                }
            }
        }
        for (int k = 0; k < 3; ++k) q[k] = rotate90_by(q[k], n[k], best[k]);

        pos_rank(0, f) = best[0];
        pos_rank(1, f) = best[1];
        pos_rank(2, f) = best[2];

        for (int k = 0; k < 3; ++k) {
            int kn = k == 2 ? 0 : (k + 1);
            double scale_x = hierarchy.mScale, scale_y = hierarchy.mScale,
                   scale_x_1 = hierarchy.mScale, scale_y_1 = hierarchy.mScale;
            if (flag_adaptive_scale) {
                scale_x *= hierarchy.mS[0](0, F(k, f));
                scale_y *= hierarchy.mS[0](1, F(k, f));
                scale_x_1 *= hierarchy.mS[0](0, F(kn, f));
                scale_y_1 *= hierarchy.mS[0](1, F(kn, f));
                if (best[k] % 2 != 0) std::swap(scale_x, scale_y);
                if (best[kn] % 2 != 0) std::swap(scale_x_1, scale_y_1);
            }
            double inv_scale_x = 1.0 / scale_x, inv_scale_y = 1.0 / scale_y,
                   inv_scale_x_1 = 1.0 / scale_x_1, inv_scale_y_1 = 1.0 / scale_y_1;
            std::pair<Vector2i, Vector2i> value = compat_position_extrinsic_index_4(
                v[k], n[k], q[k], o[k], v[kn], n[kn], q[kn], o[kn], scale_x, scale_y, inv_scale_x,
                inv_scale_y, scale_x_1, scale_y_1, inv_scale_x_1, inv_scale_y_1, nullptr);
            auto diff = value.first - value.second;
            index += diff;
            pos_index(k * 2, f) = diff[0];
            pos_index(k * 2 + 1, f) = diff[1];
        }

        for (int k = 0; k < 3; ++k) {
            if (pos_index(k * 2, f) == 0 && pos_index(k * 2 + 1, f) == 0) {
                tree.Merge(F(k, f), F((k + 1) % 3, f));
            }
        }
    }

    tree.BuildCompactParent();

    positions.resize(tree.CompactNum());
    
    struct Distance
    {
        Distance() {x = 10000, y = 10000, xy = 1e30, index = -1;}
        Distance(double _x, double _y, int ind, const Eigen::Vector3d& q, const Eigen::Vector3d& n)
        : x(_x), y(_y), index(ind), q(q), n(n)
        {
            vx = x;
            vy = y;
            xy = _x * _x + _y * _y;//std::max(std::abs(_x), std::abs(_y));
        }
        double x, y, xy, vx, vy;
        int index;
        Eigen::Vector3d q;
        Eigen::Vector3d n;
        bool operator<(const Distance& nd) const {
            return xy > nd.xy + 1e-6;
        }
        bool operator==(const Distance& nd) const {
            return xy > nd.xy - 1e-6 && xy <= nd.xy + 1e-6;
        }
        int rot() const {
            if (x >= 0 && y >= 0)
                return 0;
            if (x < 0 && y >= 0)
                return 1;
            if (x < 0 && y < 0)
                return 2;
            return 3;
        }
    };

    struct DistanceIndex
    {
        DistanceIndex(int i, Distance& v)
        : i(i), v(v) {
        }
        bool operator<(const DistanceIndex& nd) const {
            return v < nd.v;
        }
        bool operator==(const DistanceIndex& nd) const {
            return v == nd.v;
        }
        int i;
        Distance v;
    };

    std::vector<std::set<int> > links1(V.cols()), links(V.cols());
    for (int i = 0; i < F.cols(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int ind1 = F(j, i);
            int ind2 = F((j + 1) % 3, i);
            links1[ind1].insert(ind2);
            links1[ind2].insert(ind1);
        }
    }

    for (int i = 0; i < links1.size(); ++i) {
        for (auto& p : links1[i]) {
            links[i].insert(p);
            for (auto& q : links1[p]) {
                links[i].insert(q);
            }
        }
    }

    std::vector<Distance> distances(V.cols() * 4);
    std::vector<int> count(tree.CompactNum(), 0);
    min_id.resize(tree.CompactNum());
    std::vector<double> min_dis(tree.CompactNum(), 1e30);

    for (int i = 0; i < positions.size(); ++i) {
        positions[i] = Eigen::Vector3d(0, 0, 0);
    }
    for (int i = 0; i < V.cols(); ++i) {
        int index = tree.Index(i);
        count[index] += 1;
        positions[index] += O.col(i);
    }

    for (int i = 0; i < positions.size(); ++i) {
        positions[i] /= count[i];
    }

    for (int i = 0; i < V.cols(); ++i) {
        int index = tree.Index(i);        
        double dis = (V.col(i) - positions[index]).norm();
        if (dis < min_dis[index]) {
            min_dis[index] = dis;
            min_id[index] = i;
        }
    }

    groups.resize(V.cols() * 4);

    for (auto& g : groups) {
        g.first.first = -1;
    }

    for (int i = 0; i < min_id.size(); ++i) {
        int id = min_id[i];
        Eigen::Vector3d q = Q.col(id);
        Eigen::Vector3d n = N.col(id);
        Eigen::Vector3d qy = n.cross(q);
        double x = ((V.col(id) - positions[i]).dot(q));
        double y = ((V.col(id) - positions[i]).dot(qy));
        Distance d(x, y, i, q, n);
        int expand_id = id * 4 + d.rot();
        if (distances[expand_id] < d) {
            distances[expand_id] = d;
        }
    }
    std::priority_queue<DistanceIndex> instances;
    for (int i = 0; i < distances.size(); ++i) {
        if (distances[i].xy < 10000) {
            instances.push(DistanceIndex(i, distances[i]));
        }
    }
    std::vector<int> visited(distances.size(), 0);
    while (!instances.empty()) {
        DistanceIndex info = instances.top();
        instances.pop();
        int index = info.i;
        if (visited[index] == 1)
            continue;
        visited[index] = 1;
        groups[index] = std::make_pair(std::make_pair(distances[index].index, distances[index].xy), Eigen::Vector2d(distances[index].x, distances[index].y));
        int vid = index / 4;
        for (auto& ni : links[vid]) {
            Eigen::Vector3d n = N.col(ni);
            Eigen::Vector3d q = Q.col(ni);

            auto diff = compat_orientation_extrinsic_index_4(q, n, info.v.q, info.v.n);
            Eigen::Vector3d q0 = q;
            q = rotate90_by(q, n, (diff.first - diff.second + 4) % 4);

            double new_x = (q.dot(V.col(ni) - V.col(vid)));
            Eigen::Vector3d qy = n.cross(q);
            double new_y = (qy.dot(V.col(ni) - V.col(vid)));

            double abs_new_x = std::abs(new_x) + std::abs(info.v.vx);
            double abs_new_y = std::abs(new_y) + std::abs(info.v.vy);
            if (approx_dis == 0) {
                abs_new_x = new_x + info.v.x;
                abs_new_y = new_y + info.v.y;
            }
            Distance d = Distance(abs_new_x, abs_new_y, info.v.index, q, n);
            d.x = new_x + info.v.x;
            d.y = new_y + info.v.y;
            int ind = ni * 4 + (diff.first - diff.second + 4 + d.rot()) % 4;
            if (visited[ind] == 1)
                continue;

            if (d.xy < distances[ind].xy) {
                distances[ind] = d;
                instances.push(DistanceIndex(ind, distances[ind]));
            }
        }
    }

    int num_groups = positions.size();

    std::vector<std::pair<Eigen::Vector4i, Eigen::Vector4i> > entries;
    for (int i = 0; i < num_groups; ++i) {
        entries.push_back(std::make_pair(Vector4i(-1, -1, -1, -1),Vector4i(-1, -1, -1, -1)));
    }
    // Build Conv Indices
    for (int i = 0; i < F.cols(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int vid1 = F(j, i);
            int vid2 = F((j + 1) % 3, i);
            int index = tree.Index(vid1);
            int index1 = tree.Index(vid2);
            if (index == index1)
                continue;

            Vector2i diff(pos_index(j * 2, i), pos_index(j * 2 + 1, i));
            int vref = min_id[index];
            auto orients = compat_orientation_extrinsic_index_4(Q.col(vid1), N.col(vid1), Q.col(vref), N.col(vref));
            vref = min_id[index1];
            auto orients1 = compat_orientation_extrinsic_index_4(Q.col(vid2), N.col(vid2), Q.col(vref), N.col(vref));

            diff = rshift90(diff, (pos_rank(j, i) + orients.second - orients.first + 4) % 4);
            int orient_diff = (pos_rank((j + 1) % 3, i) - pos_rank(j, i) + 12 + orients1.second - orients.second + orients.first - orients1.first) % 4;
            if (diff[0] == 1 && diff[1] == 0) {
                if (entries[index].first[0] == -1) {
                    entries[index].first[0] = index1;
                    entries[index].second[0] = orient_diff;
                }
            }
            if (diff[0] == 0 && diff[1] == 1) {
                if (entries[index].first[1] == -1) {
                    entries[index].first[1] = index1;
                    entries[index].second[1] = orient_diff;
                }
            }
            if (diff[0] == -1 && diff[1] == 0) {
                if (entries[index].first[2] == -1) {
                    entries[index].first[2] = index1;
                    entries[index].second[2] = orient_diff;
                }
            }
            if (diff[0] == 0 && diff[1] == -1) {
                if (entries[index].first[3] == -1) {
                    entries[index].first[3] = index1;
                    entries[index].second[3] = orient_diff;
                }
            }
        }
    }
    
    int kernel_size = this->kernel_size;
    conv_indices.resize(kernel_size * kernel_size * entries.size(), std::make_pair(-1, -1));
    struct VisitInfo
    {
        VisitInfo()
        {}
        VisitInfo(int x, int y, int index, int start_dir)
        : x(x), y(y), index(index), start_dir(start_dir)
        {}
        int x;
        int y;
        int index;
        int start_dir;
    };
    int offset = 0;
    for (int i = 0; i < entries.size(); ++i) {
        std::queue<VisitInfo> q;
        std::vector<int> hash(kernel_size * kernel_size, 0);
        q.push(VisitInfo(kernel_size/2, kernel_size/2, i, 0));
        hash[kernel_size/2 * kernel_size + kernel_size/2] = 1;
        conv_indices[offset + kernel_size/2 * kernel_size + kernel_size/2] = std::make_pair(i, 0);
        int dx[] = {1, 0, -1, 0};
        int dy[] = {0, 1, 0, -1};
        while (!q.empty()) {
            auto info = q.front();
            q.pop();
            int index = info.index;
            for (int j = 0; j < 4; ++j) {
                int next_index = entries[index].first[(info.start_dir + j) % 4];
                if (next_index == -1) {
                    continue;
                }
                int tx = info.x + dx[j];
                int ty = info.y + dy[j];
                if (tx < 0 || ty < 0 || tx >= kernel_size || ty >= kernel_size || hash[tx * kernel_size + ty])
                    continue;
                int next_dir = (info.start_dir + entries[index].second[(info.start_dir + j) % 4]) % 4;
                q.push(VisitInfo(tx, ty, next_index, next_dir));
                hash[tx * kernel_size + ty] = 1;
                conv_indices[offset + tx * kernel_size + ty] = std::make_pair(next_index, next_dir);
            }
        }
        hash.clear();
        hash.resize(kernel_size * kernel_size, 0);
        q.push(VisitInfo(kernel_size/2, kernel_size/2, i, 0));
        hash[kernel_size/2 * kernel_size + kernel_size/2] = 1;
        while (!q.empty()) {
            auto info = q.front();
            q.pop();
            for (int j = 0; j < 4; ++j) {
                int tx = info.x + dx[j];
                int ty = info.y + dy[j];
                if (tx < 0 || ty < 0 || tx >= kernel_size || ty >= kernel_size || hash[tx * kernel_size + ty])
                    continue;
                q.push(VisitInfo(tx, ty, 0, 0));
                hash[tx * kernel_size + ty] = 1;
                if (conv_indices[offset + tx * kernel_size + ty].first == -1)
                    conv_indices[offset + tx * kernel_size + ty] = conv_indices[offset + info.x * kernel_size + info.y];
            }
        }
        offset += kernel_size * kernel_size;
    }

    for (int i = 0; i < positions.size(); ++i) {
        positions[i] = this->normalize_scale * positions[i] + this->normalize_offset;
    }
    /*
    std::vector<Eigen::Vector3i> colors(positions.size());
    for (int i = 0; i < colors.size(); ++i) {
        colors[i] = Eigen::Vector3i(rand() % 256, rand() % 256, rand() % 256);
    }
    static int frame = 0;
    char filename[1024];
    frame += 1;
    sprintf(filename, "result%d.obj", frame);
    std::ofstream os(filename);
    for (int i = 0; i < V.cols(); ++i) {
        int c = -1;
        double min_dis = 1e30;
        for (int j = 0; j < 4; ++j) {
            double dis = groups[i * 4 + j].first.second;
            if (dis < min_dis) {
                c = groups[i * 4 + j].first.first;
                min_dis = dis;
            }
        }
        //int c = groups[i * 4].first;
        os << "v " << V(0, i) << " " << V(1, i) << " " << V(2, i) << " ";
        os << colors[c][0] << " " << colors[c][1] << " " << colors[c][2] << "\n";
    }
    for (int i = 0; i < F.cols(); ++i) {
        os << "f " << F(0, i) + 1 << " " << F(1, i) + 1 << " " << F(2, i) + 1 << "\n";
    }
    os.close();

    os.open(std::string(filename) + "_model.obj");
    for (int i = 0; i < positions.size(); ++i) {
        os << "v " << positions[i][0] << " " << positions[i][1] << " " << positions[i][2] << "\n";
    }
    os.close();
    */
}