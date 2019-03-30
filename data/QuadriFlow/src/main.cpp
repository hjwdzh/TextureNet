#include "config.hpp"
#include "field-math.hpp"
#include "gldraw.hpp"
#include "optimizer.hpp"
#include "parametrizer.hpp"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

Parametrizer field, level1, level2, level3, level4;

void BuildPoolIndices(std::vector<std::pair<std::pair<int, double>, Eigen::Vector2d> >& groups1, std::vector<std::pair<std::pair<int, double>, Eigen::Vector2d> >& groups2, 
    std::vector<int>& min_id1, std::vector<int>& min_id2,
    Parametrizer& field1, Parametrizer& field2,
    std::vector<std::pair<int, int> >& indices) {

    indices.resize(min_id1.size(), std::make_pair(-1, -1));
    auto& q1 = field1.hierarchy.mQ[0];
    auto& n1 = field1.hierarchy.mN[0];
    auto& q2 = field2.hierarchy.mQ[0];
    auto& n2 = field2.hierarchy.mN[0];

    for (int i = 0; i < min_id1.size(); ++i) {
        int mid = min_id1[i];
        double min_dis = 1e30;
        int nid = -1;
        for (int j = mid * 4; j < mid * 4 + 4; ++j) {
            if (groups2[j].first.first == -1)
                continue;
            if (groups2[j].first.second < min_dis) {
                nid = groups2[j].first.first;
                min_dis = groups2[j].first.second;
            }
        }
        auto diff = compat_orientation_extrinsic_index_4(q1.col(mid), n1.col(mid), q2.col(min_id2[nid]), n2.col(min_id2[nid]));
        indices[i] = std::make_pair(nid, (diff.second - diff.first + 4) % 4);
    }
}

void BuildDeconvIndices(std::vector<std::pair<std::pair<int, double>, Eigen::Vector2d> >& groups1, std::vector<std::pair<std::pair<int, double>, Eigen::Vector2d> >& groups2, 
    std::vector<int>& min_id1, std::vector<int>& min_id2,
    Parametrizer& field1, Parametrizer& field2,
    std::vector<std::pair<std::pair<int, int>, double> >& indices) {

    indices.resize(min_id1.size() * 4, std::make_pair(std::make_pair(-1, -1), 1e30));
    auto& q1 = field1.hierarchy.mQ[0];
    auto& n1 = field1.hierarchy.mN[0];
    auto& q2 = field2.hierarchy.mQ[0];
    auto& n2 = field2.hierarchy.mN[0];

    for (int i = 0; i < min_id1.size(); ++i) {
        int mid = min_id1[i];
        int nid = -1;
        for (int j = mid * 4; j < mid * 4 + 4; ++j) {
            if (groups2[j].first.first == -1)
                continue;
            nid = groups2[j].first.first;
            auto diff = compat_orientation_extrinsic_index_4(q1.col(mid), n1.col(mid), q2.col(min_id2[nid]), n2.col(min_id2[nid]));
            indices[(i - mid) * 4 + j] = std::make_pair(std::make_pair(nid, (diff.second - diff.first + 4) % 4), std::sqrt(groups2[j].first.second));
        }
    }
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

#ifdef WITH_CUDA
    cudaFree(0);
#endif
    int t1, t2;
    field.num_levels = 5;
    field.rosy = 4;
    field.kernel_size = 3;
    field.gen_textiles = true;
    field.gen_pool = false;
    field.gen_dialation = false;
    field.gen_normalize = false;
    field.gen_entry = false;
    field.gen_frames = false;
    field.unit = 0.1;
    std::string input_obj, output_obj;
    int faces = -1;

    int gen_field = 0;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-f") == 0) {
            sscanf(argv[i + 1], "%d", &faces);
        }
        if (strcmp(argv[i], "-i") == 0) {
            input_obj = argv[i + 1];
        }
        if (strcmp(argv[i], "-o") == 0) {
            output_obj = argv[i + 1];
        }
        if (strcmp(argv[i], "-sharp") == 0) {
            field.flag_preserve_sharp = 1;
        }
        if (strcmp(argv[i], "-adaptive") == 0) {
            field.flag_adaptive_scale = 1;
        }
        if (strcmp(argv[i], "-indices") == 0) {
            field.indices_file = argv[i + 1];
        }
        if (strcmp(argv[i], "-position") == 0) {
            field.position_file = argv[i + 1];
        }
        if (strcmp(argv[i], "-frame") == 0) {
            field.frames_file = argv[i + 1];
            field.gen_frames = true;
        }
        if (strcmp(argv[i], "-entry") == 0) {
            field.entry_file = argv[i + 1];
            field.gen_entry = true;
        }
        if (strcmp(argv[i], "-num_levels") == 0) {
            sscanf(argv[i + 1], "%d", &field.num_levels);
        }
        if (strcmp(argv[i], "-rosy") == 0) {
            sscanf(argv[i + 1], "%d", &field.rosy);
        }
        if (strcmp(argv[i], "-ks") == 0) {
            sscanf(argv[i + 1], "%d", &field.kernel_size);
        }
        if (strcmp(argv[i], "-normalize") == 0) {
            field.gen_normalize = true;
        }
        if (strcmp(argv[i], "-unit") == 0) {
            sscanf(argv[i + 1], "%f", &field.unit);
        }
        if (strcmp(argv[i], "-gen_textiles") == 0) field.gen_textiles = true;
        if (strcmp(argv[i], "-gen_pool") == 0) field.gen_pool = true;
        if (strcmp(argv[i], "-gen_dialation") == 0) field.gen_dialation = true;
        if (strcmp(argv[i], "-gen_field") == 0) gen_field = 1;
    }
    printf("%d %s %s\n", faces, input_obj.c_str(), output_obj.c_str());
    fflush(stdout);
    if (input_obj.size() >= 1) {
        field.Load(input_obj.c_str());
    } else {
        field.Load((std::string(DATA_PATH) + "/fertility.obj").c_str());
    }

    printf("Initialize...\n");
    fflush(stdout);
    field.Initialize(faces);

    printf("Solve Orientation Field...\n");
    fflush(stdout);
    t1 = GetCurrentTime64();

    Optimizer::optimize_orientations(field.hierarchy);
    field.ComputeOrientationSingularities();
    t2 = GetCurrentTime64();
    printf("Use %lf seconds\n", (t2 - t1) * 1e-3);
    fflush(stdout);

    if (field.flag_adaptive_scale == 1) {
        printf("Estimate Slop...\n");
        t1 = GetCurrentTime64();
        field.EstimateSlope();
        t2 = GetCurrentTime64();
        printf("Use %lf seconds\n", (t2 - t1) * 1e-3);
    }
    printf("Solve for scale...\n");
    t1 = GetCurrentTime64();
    Optimizer::optimize_scale(field.hierarchy, field.rho, field.flag_adaptive_scale);
    field.flag_adaptive_scale = 1;
    t2 = GetCurrentTime64();
    printf("Use %lf seconds\n", (t2 - t1) * 1e-3);

    printf("Solve for position field...\n");
    fflush(stdout);
    t1 = GetCurrentTime64();

    if (gen_field) {
        if (field.gen_textiles) {
            std::ofstream os(field.position_file);
            auto& V = field.hierarchy.mV[0];
            auto& F = field.hierarchy.mF;
            for (int i = 0; i < field.num_original_v; ++i) {
                Eigen::Vector3d t = V.col(i);
                t = field.normalize_scale * t + field.normalize_offset;
                os << t[0] << " " << t[1] << " " << t[2] << "\n";
                if (i == field.num_original_v - 1)
                    printf("%f %f %f\n", t[0], t[1], t[2]);
            }
            os.close();
        }
        if (field.gen_frames) {
            std::ofstream os(field.frames_file);
            auto& O = field.hierarchy.mQ[0];
            auto& N = field.hierarchy.mN[0];
            auto& F = field.hierarchy.mF;
            for (int i = 0; i < field.num_original_v; ++i) {
                int ii = i;
                os << O(0, ii) << " " << O(1, ii) << " " << O(2, ii) << " "
                << N(0, ii) << " " << N(1, ii) << " " << N(2, ii) << "\n";
            }
            os.close();
            printf("Finish...\n");
        }        
        return 0;
    }

    if (field.gen_dialation || field.gen_entry) {
        Optimizer::optimize_positions(field.hierarchy, field.flag_adaptive_scale);
        printf("Quantize...\n");
        field.QuantizeTextiles();
        return 0;
    }

    level1 = field;
    level1.hierarchy.mScale = 0.1;
    level2 = field;
    level2.hierarchy.mScale = 0.2;
    level3 = field;
    level3.hierarchy.mScale = 0.4;
    level4 = field;
    level4.hierarchy.mScale = 0.8;
    printf("Optimize Position0\n");
    Optimizer::optimize_positions(field.hierarchy, field.flag_adaptive_scale);
    printf("Optimize Position1\n");
    Optimizer::optimize_positions(level1.hierarchy, field.flag_adaptive_scale);
    printf("Optimize Position2\n");
    Optimizer::optimize_positions(level2.hierarchy, field.flag_adaptive_scale);
    printf("Optimize Position3\n");
    Optimizer::optimize_positions(level3.hierarchy, field.flag_adaptive_scale);
    printf("Optimize Position4\n");
    Optimizer::optimize_positions(level4.hierarchy, field.flag_adaptive_scale);

    std::vector<std::pair<std::pair<int, double>, Eigen::Vector2d> > groups_level0, groups_level1, groups_level2, groups_level3, groups_level4;
    std::vector<int> mind_level0, mind_level1, mind_level2, mind_level3, mind_level4;
    std::vector<Eigen::Vector3d> positions0, positions1, positions2, positions3, positions4;
    std::vector<std::pair<int, int> > conv_indices1, conv_indices2, conv_indices3, conv_indices4, conv_indices5, pool_indices01, pool_indices12, pool_indices23, pool_indices34;
    std::vector<std::pair<std::pair<int, int>, double> > deconv_indices01, deconv_indices12, deconv_indices23, deconv_indices34;

    printf("Extract Group0\n");
    field.ExtractGroups(groups_level0, mind_level0, positions0, conv_indices1, 0);
    printf("Extract Group1\n");
    level1.ExtractGroups(groups_level1, mind_level1, positions1, conv_indices2, 0);
    printf("Extract Group2\n");
    level2.ExtractGroups(groups_level2, mind_level2, positions2, conv_indices3, 1);
    printf("Extract Group3\n");
    level3.ExtractGroups(groups_level3, mind_level3, positions3, conv_indices4, 1);
    printf("Extract Group4\n");
    level4.ExtractGroups(groups_level4, mind_level4, positions4, conv_indices5, 1);

    printf("%d %d %d %d %d\n", conv_indices1.size(), conv_indices2.size(), conv_indices3.size(), conv_indices4.size(), conv_indices5.size());

    printf("Build Pool0\n");
    BuildPoolIndices(groups_level0, groups_level1, mind_level0, mind_level1, field, level1, pool_indices01);
    printf("Build Pool1\n");
    BuildPoolIndices(groups_level1, groups_level2, mind_level1, mind_level2, level1, level2, pool_indices12);
    printf("Build Pool2\n");
    BuildPoolIndices(groups_level2, groups_level3, mind_level2, mind_level3, level2, level3, pool_indices23);
    printf("Build Pool3\n");
    BuildPoolIndices(groups_level3, groups_level4, mind_level3, mind_level4, level3, level4, pool_indices34);

    printf("Build Deconv0\n");
    BuildDeconvIndices(groups_level0, groups_level1, mind_level0, mind_level1, field, level1, deconv_indices01);
    printf("Build Deconv1\n");
    BuildDeconvIndices(groups_level1, groups_level2, mind_level1, mind_level2, level1, level2, deconv_indices12);
    printf("Build Deconv2\n");
    BuildDeconvIndices(groups_level2, groups_level3, mind_level2, mind_level3, level2, level3, deconv_indices23);
    printf("Build Deconv3\n");
    BuildDeconvIndices(groups_level3, groups_level4, mind_level3, mind_level4, level3, level4, deconv_indices34);

    std::ofstream os;
    os.open(field.position_file);
    for (int i = 0; i < positions0.size(); ++i) {
        os << positions0[i][0] << " " << positions0[i][1] << " " << positions0[i][2] << "\n";
    }
    os.close();
    FILE* fp = fopen(field.indices_file.c_str(), "wb");
    int num = 4;
    fwrite(&num, sizeof(int), 1, fp);
    num = conv_indices1.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(conv_indices1.data(), sizeof(std::pair<int, int>), conv_indices1.size(), fp);
    num = conv_indices2.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(conv_indices2.data(), sizeof(std::pair<int, int>), conv_indices2.size(), fp);
    num = conv_indices3.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(conv_indices3.data(), sizeof(std::pair<int, int>), conv_indices3.size(), fp);
    num = conv_indices4.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(conv_indices4.data(), sizeof(std::pair<int, int>), conv_indices4.size(), fp);
    num = conv_indices5.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(conv_indices5.data(), sizeof(std::pair<int, int>), conv_indices5.size(), fp);

    num = pool_indices01.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(pool_indices01.data(), sizeof(std::pair<int, int>), pool_indices01.size(), fp);
    num = pool_indices12.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(pool_indices12.data(), sizeof(std::pair<int, int>), pool_indices12.size(), fp);
    num = pool_indices23.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(pool_indices23.data(), sizeof(std::pair<int, int>), pool_indices23.size(), fp);
    num = pool_indices34.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(pool_indices34.data(), sizeof(std::pair<int, int>), pool_indices34.size(), fp);

    num = deconv_indices01.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(deconv_indices01.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv_indices01.size(), fp);
    num = deconv_indices12.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(deconv_indices12.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv_indices12.size(), fp);
    num = deconv_indices23.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(deconv_indices23.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv_indices23.size(), fp);
    num = deconv_indices34.size();
    fwrite(&num, sizeof(int), 1, fp);
    fwrite(deconv_indices34.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv_indices34.size(), fp);

    fclose(fp);
    return 0;
}
