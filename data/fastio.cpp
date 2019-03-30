#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <map>
std::vector<std::pair<int, int> > conv0, conv1, conv2, conv3, conv4;
std::vector<std::pair<int, int> > pool01, pool12, pool23, pool34;
std::vector<std::pair<std::pair<int, int>, double> > deconv01, deconv12, deconv23, deconv34;
std::vector<float> positions;

void InitializeFullIndicesCPP(const char* filename) {
	FILE* fp = fopen(filename, "rb");
    int num = 4;
    fread(&num, sizeof(int), 1, fp);
    fread(&num, sizeof(int), 1, fp);
    printf("%d\n", num);
    conv0.resize(num);
    fread(conv0.data(), sizeof(std::pair<int, int>), conv0.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    printf("%d\n", num);
    conv1.resize(num);
    fread(conv1.data(), sizeof(std::pair<int, int>), conv1.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    printf("%d\n", num);
    conv2.resize(num);
    fread(conv2.data(), sizeof(std::pair<int, int>), conv2.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    printf("%d\n", num);
    conv3.resize(num);
    fread(conv3.data(), sizeof(std::pair<int, int>), conv3.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    printf("%d\n", num);
    conv4.resize(num);
    fread(conv4.data(), sizeof(std::pair<int, int>), conv4.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    pool01.resize(num);
    fread(pool01.data(), sizeof(std::pair<int, int>), pool01.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    pool12.resize(num);
    fread(pool12.data(), sizeof(std::pair<int, int>), pool12.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    pool23.resize(num);
    fread(pool23.data(), sizeof(std::pair<int, int>), pool23.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    pool34.resize(num);
    fread(pool34.data(), sizeof(std::pair<int, int>), pool34.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    deconv01.resize(num);
    fread(deconv01.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv01.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    deconv12.resize(num);
    fread(deconv12.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv12.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    deconv23.resize(num);
    fread(deconv23.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv23.size(), fp);
    fread(&num, sizeof(int), 1, fp);
    deconv34.resize(num);
    fread(deconv34.data(), sizeof(std::pair<std::pair<int, int>, double>), deconv34.size(), fp);
    fclose(fp);

    for (int i = 0; i < 4; ++i) {
    	auto& conv = (i == 0) ? deconv01 : ((i == 1) ? deconv12 : (i == 2 ? deconv23 : deconv34));
    	for (int j = 0; j < (int)conv.size(); j += 4) {
    		int min_id = -1;
    		for (int k = 0; k < 4; ++k) {
    			if (conv[j + k].first.first != -1 && (min_id == -1 || conv[j + k].second < conv[min_id].second)) {
    				min_id = j + k;
    			}
    		}
    		if (min_id != -1) {
    			for (int k = 0; k < 4; ++k) {
    				if (conv[j + k].first.first == -1) {
    					conv[j + k] = conv[min_id];
    				}
    			}
    		}
    	}
    }
}

int GetNumConvIndicesCPP(int level) {
	auto& conv = (level == 0) ? conv0 : ((level == 1) ? conv1 : ((level == 2) ? conv2 : (level == 3 ? conv3 : conv4)));
	return (int)conv.size();
}

int GetNumPoolIndicesCPP(int level) {
	level += 1;
	auto& conv = ((level == 1) ? conv1 : ((level == 2) ? conv2 : ((level == 3) ? conv3 : conv4)));
	return (int)conv.size() / 9;
}

int GetNumDeconvIndicesCPP(int level) {
	auto& conv = (level == 0) ? deconv01 : ((level == 1) ? deconv12 : (level == 2 ? deconv23 : deconv34));
	return (int)conv.size();
}

void GetConvIndicesCPP(void* indatav, int level, int rosy) {
	int* indices = (int*)indatav;
	auto& conv = (level == 0) ? conv0 : ((level == 1) ? conv1 : ((level == 2) ? conv2 : (level == 3 ? conv3 : conv4)));
	if (rosy == 1) {
		for (int i = 0; i < (int)conv.size(); ++i) {
			indices[i] = conv[i].first;
		}
	} else {
		for (int i = 0; i < (int)conv.size(); ++i) {
			for (int dir = 0; dir < 4; ++dir) {
				indices[i * 4 + dir] = conv[i].first * 4 + (dir + conv[i].second) % 4;
			}
		}
	}
}

void GetPoolIndicesCPP(void* indatav, int level, int rosy) {
	int num_pool = GetNumPoolIndicesCPP(level);
	int* indices = (int*)indatav;
	auto& pool = (level == 0) ? pool01 : ((level == 1) ? pool12 : (level == 2 ? pool23 : pool34));
	if (rosy == 1) {
		for (int i = 0; i < (int)pool.size(); ++i) {
			int index = pool[i].first;
			indices[i] = index;
		}
	} else {
		for (int i = 0; i < (int)pool.size(); ++i) {
			int offset = i * 4;
			for (int j = 0; j < 4; ++j) {
				int index = pool[i].first * 4 + (j + pool[i].second) % 4;
				indices[i * 4 + j] = index;
			}
		}
	}
}

void GetPoolIndicesReverseCPP(void* indatav, int level, int rosy, int stride) {
	int num_pool = GetNumPoolIndicesCPP(level);
	std::vector<int> count(num_pool, 0);
	int* indices = (int*)indatav;

	auto& pool = (level == 0) ? pool01 : ((level == 1) ? pool12 : pool23);
	if (rosy == 1) {
		for (int i = 0; i < (int)pool.size(); ++i) {
			int index = pool[i].first;
			int& offset = count[pool[i].first];
			if (offset < stride) {
				indices[index * stride + offset] = i;
				offset += 1;
			}
		}
		for (int i = 0; i < num_pool; ++i) {
			int offset = count[i];
			int t = (offset == 0) ? (int)pool.size() : indices[i * stride + offset - 1];
			for (int j = offset; j < stride; ++j) {
				indices[i * stride + j] = t;
			}
		}
	} else {
		for (int i = 0; i < (int)pool.size(); ++i) {
			int& offset = count[pool[i].first];
			for (int j = 0; j < 4; ++j) {
				int index = pool[i].first * 4 + (j + pool[i].second) % 4;
				if (offset < stride) {
					indices[index * stride + offset] = i * 4 + j;
				}
			}
			offset += 1;
		}
		for (int i = 0; i < num_pool * 4; ++i) {
			int offset = count[i / 4];
			int t = (offset == 0) ? (int)pool.size() * 4 : indices[i * stride + offset - 1];
			for (int j = offset; j < stride; ++j) {
				indices[i * stride + j] = t;
			}
		}
	}
}

void GetDeconvIndicesCPP(void* indatav, int level, int rosy) {
	int* indices = (int*)indatav;

	auto& conv = (level == 0) ? deconv01 : ((level == 1) ? deconv12 : (level == 2 ? deconv23 : deconv34));
	int num_pool = GetNumPoolIndicesCPP(level);
	if (rosy == 1) {
		for (int i = 0; i < (int)conv.size(); ++i) {
			int index = conv[i].first.first;
			indices[i] = index;
		}
	} else {
		for (int i = 0; i < (int)conv.size(); ++i) {
			for (int j = 0; j < 4; ++j) {
				int index = conv[i].first.first * 4 + (j + conv[i].first.second) % 4;
				indices[i] = index;
			}
		}
	}
}

extern "C" {

void InitializeTangentSpace(void* vertices, void* faces, void* nf, int num_v, int num_f, void* tangent_space) {
	auto V = Eigen::Map<Eigen::MatrixXf>((float*)vertices, 3, num_v);
	auto F = Eigen::Map<Eigen::MatrixXi>((int*)faces, 3, num_f);
	auto FN = Eigen::Map<Eigen::MatrixXf>((float*)nf, 3, num_f);
	for (int i = 0; i < num_f; ++i) {
		Eigen::Matrix3d p, q;
		Eigen::Vector3f v1 = V.col(F(1, i)) - V.col(F(0, i));
		Eigen::Vector3f v2 = V.col(F(2, i)) - V.col(F(0, i));
		Eigen::Vector3f v3 = FN.col(i);
		p.col(0) = Eigen::Vector3d(v1[0], v1[1], v1[2]);
		p.col(1) = Eigen::Vector3d(v2[0], v2[1], v2[2]);
		p.col(2) = Eigen::Vector3d(v3[0], v3[1], v3[2]);
		q = p.inverse();
		double* triangle_space_f = ((double*)tangent_space) + 6 * i;
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 3; ++k) {
				triangle_space_f[k * 2 + j] = q(j, k);
			}
		}
	}

}

void ReadBaryCentry(void* bary_coords, void* bary_indices, const char* filename) {
	FILE* fp = fopen(filename, "rb");
	int num_C, num_I;
	fread(&num_C, sizeof(int), 1, fp);
	fread(&num_I, sizeof(int), 1, fp);
	fread(bary_coords, sizeof(float), 3 * num_C, fp);
	fread(bary_indices, sizeof(int), num_I, fp);
	fclose(fp);
}

void ComputeFacesInfo(void* fv, void* fn, void* v, void* f, int num_v, int num_f) {
	auto V = Eigen::Map<Eigen::MatrixXf>((float*)v, 3, num_v);
	auto F = Eigen::Map<Eigen::MatrixXi>((int*)f, 3, num_f);
#pragma omp parallel for
	for (int i = 0; i < num_f; ++i) {
		float* fv_f = ((float*)fv) + 3 * i;
		float* fn_f = ((float*)fn) + 3 * i;
		Eigen::Vector3f FV = (V.col(F(0, i)) + V.col(F(1, i)) + V.col(F(2, i))) / 3.0f;
		Eigen::Vector3f FN = (Eigen::Vector3f(V.col(F(1, i)) - V.col(F(0, i))).cross(Eigen::Vector3f(V.col(F(2, i)) - V.col(F(0, i))))).normalized();
		fv_f[0] = FV[0];
		fv_f[1] = FV[1];
		fv_f[2] = FV[2];
		fn_f[0] = FN[0];
		fn_f[1] = FN[1];
		fn_f[2] = FN[2];
	}
}

void InitializeE2E(void* vertices, void* faces, int num_v, int num_f, void* pE2E) {
	int* iE2E = (int*)pE2E;
	auto V = Eigen::Map<Eigen::MatrixXf>((float*)vertices, 3, num_v);
	auto F = Eigen::Map<Eigen::MatrixXi>((int*)faces, 3, num_f);
	//E2E = Eigen::VectorXi(num_f * 3);
	std::map<std::pair<int, int>, int > dedgeid;
	for (int i = 0; i < num_f; ++i) {
		for (int j = 0; j < 3; ++j) {
			iE2E[i * 3 + j] = -1;
			int x = F(j, i);
			int y = F((j + 1) % 3, i);
			auto key = std::make_pair(x, y);
			auto reverse_key = std::make_pair(y, x);
			if (dedgeid.count(reverse_key)) {
				int deid = dedgeid[reverse_key];
				iE2E[i * 3 + j] = deid;
				iE2E[deid] = i * 3 + j;
				dedgeid.erase(reverse_key);
			} else {
				dedgeid[key] = i * 3 + j;
			}
		}
	}
}

void IndexPool(const void * indatav, int inlen, void * outdatav, int outlen, int invalid_idx) {
	int* input = (int*)indatav;
	int* output = (int*)outdatav;
	for (int i = 0; i < inlen; ++i) {
		int next_index = input[i] * 4;
		while (output[next_index] != -1)
			next_index += 1;
		output[next_index] = i;
	}

	for (int i = 0; i < outlen; i += 4) {
		if (output[i] == -1)
			output[i] = invalid_idx;
		for (int j = 0; j < 3; ++j) {
			if (output[i + j + 1] == -1)
				output[i + j + 1] = output[i + j];
		}
	}
}

void IndexLabel(const void* indatav, int inlen, void* outdatav, int outlen, int invalid_idx) {
	int* input = (int*)indatav;
	int* output = (int*)outdatav;
	int top = 0;
	for (int i = 0; i < inlen; ++i) {
		if (input[i] != invalid_idx) {
			output[top++] = i;
		}
	}
}

void ReadDialateIndices(void* indatav, const char* filename, int size) {
	int t;
	FILE* fp = fopen(filename, "rb");
	fread(&t, sizeof(int), 1, fp);
	fread(indatav, sizeof(int), size, fp);
	fclose(fp);
}

void ReadArray(void* indatav, const char* filename, int size, int stride) {
	FILE* fp = fopen(filename, "rb");
	fread(indatav, stride, size, fp);
	fclose(fp);
}

int ReadOBJVertices(const char* filename) {
	positions.clear();
	positions.reserve(65536);
	std::ifstream is(filename);
	char c;
	while (is >> c) {
		if (c != 'v')
			break;
		float x, y, z;
		is >> x >> y >> z;
		positions.push_back(x);
		positions.push_back(y);
		positions.push_back(z);
	}
	return positions.size() / 3;
}

void CopyOBJVertices(void* pvertices) {
	memcpy(pvertices, positions.data(), sizeof(float) * positions.size());
}

void InitializeFullIndices(const char* filename) {
	InitializeFullIndicesCPP(filename);
}

int GetLevels() {
	return 4;
}

int GetNumConvIndices(int level) {
	return GetNumConvIndicesCPP(level);
}

int GetNumPoolIndices(int level) {
	return GetNumPoolIndicesCPP(level);
}

int GetNumDeconvIndices(int level) {
	return GetNumDeconvIndicesCPP(level);
}

void GetConvIndices(void* indatav, int level, int rosy) {
	GetConvIndicesCPP(indatav, level, rosy);
}

void GetPoolIndices(void* indatav, int level, int rosy) {
	GetPoolIndicesCPP(indatav, level, rosy);
}

void GetDeconvIndices(void* indatav, int level, int rosy) {
	GetDeconvIndicesCPP(indatav, level, rosy);
}


}
