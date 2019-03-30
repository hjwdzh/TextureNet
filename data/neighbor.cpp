#include <fstream>
#include <Eigen/Core>
#include <igl/point_mesh_squared_distance.h>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

Eigen::MatrixXf V;
Eigen::MatrixXi F;
Eigen::MatrixXf FN;
Eigen::MatrixXf FV;

int* E2E;
glm::vec3 *pts1, *pts2;
std::pair<glm::vec3, glm::vec3> *pframes;
int *face_indices1, *face_indices2, *group_indices;
glm::vec2* group_tex;

int* find_queue;
glm::vec3 *ux_queue, *uy_queue;
glm::vec2 *coord_queue;
int* neighbor_hash;
glm::vec3 *ux_hash, *uy_hash;
glm::vec2 *coord_hash;

glm::vec3 *cudaV;
glm::ivec3 *cudaF;
glm::vec3 *cudaFV;
glm::vec3 *cudaFN;
int *cudaE2E;
int num_v, num_f;

glm::vec3 *p8192, *p1024, *p256, *p64, *p16;
glm::ivec3 *i1024, *i256, *i64, *i16;

FILE* fps;

extern "C" {


double GetTickCount(void) 
{
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now))
    return 0;
  return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

void ReadArray(void* indatav, const char* filename, int size, int stride) {
	FILE* fp = fopen(filename, "rb");
	fread(indatav, stride, size, fp);
	fclose(fp);
}

void InitializeE2E(void* vertices, void* faces, int num_v, int num_f, void* pE2E) {
	int* iE2E = (int*)pE2E;
	V = Eigen::Map<Eigen::MatrixXf>((float*)vertices, 3, num_v);
	F = Eigen::Map<Eigen::MatrixXi>((int*)faces, 3, num_f);
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

void InitializeMesh(void* vertices, void* faces, void* faceV, void* faceN, void* iE2E, int _num_v, int _num_f) {
	num_v = _num_v;
	num_f = _num_f;

	V = Eigen::Map<Eigen::MatrixXf>((float*)vertices, 3, num_v);
	F = Eigen::Map<Eigen::MatrixXi>((int*)faces, 3, num_f);
	FV = Eigen::Map<Eigen::MatrixXf>((float*)faceV, 3, num_f);
	FN = Eigen::Map<Eigen::MatrixXf>((float*)faceN, 3, num_f);
	E2E = (int*)iE2E;
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

void FurthestSampling(void* pointcloud, void* inds, int num_input, int num_indices) {
	int* indices = (int*)inds;
	float* pts = (float*)pointcloud;
	std::vector<double> distance(num_input, 1e30);
	for (int i = 0; i < num_indices; ++i) {
		if (i == 0) {
			indices[i] = rand() % num_input;
			continue;
		}
		int offset_p = indices[i - 1] * 3;
		float px = pts[offset_p];
		float py = pts[offset_p + 1];
		float pz = pts[offset_p + 2];
		for (int j = 0; j < num_input; ++j) {
			int offset_j = j * 3;
			float x = pts[offset_j] - px;
			float y = pts[offset_j + 1] - py;
			float z = pts[offset_j + 2] - pz;
			float dis = x * x + y * y + z * z;
			if (dis < distance[j]) {
				distance[j] = dis;
			}
		}

		int max_index = 0;
		for(int j = 0; j < num_input; ++j)
		{
			if(distance[j] > distance[max_index])
			{
				max_index = j;
			}
		}
		indices[i] = max_index;
	}
}

void ComputeFacesInfo(void* fv, void* fn, void* v, void* f, int num_v, int num_f) {
	V = Eigen::Map<Eigen::MatrixXf>((float*)v, 3, num_v);
	F = Eigen::Map<Eigen::MatrixXi>((int*)f, 3, num_f);
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

inline Eigen::Vector3d rotate_vector_into_plane(Eigen::Vector3d q, const Eigen::Vector3d &source_normal,
                                         const Eigen::Vector3d &target_normal) {
    const double cosTheta = source_normal.dot(target_normal);
    if (cosTheta < 0.9999f) {
        if (cosTheta < -0.9999f) return -q;
        Eigen::Vector3d axis = source_normal.cross(target_normal);
        q = q * cosTheta + axis.cross(q) +
            axis * (axis.dot(q) * (1.0 - cosTheta) / axis.dot(axis));
    }
    return q;
}


inline Eigen::Vector3d Travel(Eigen::Vector3d p, const Eigen::Vector3d& dir, double len, int f, int* E2E, const Eigen::MatrixXf& V, const Eigen::MatrixXi& F, const Eigen::MatrixXf& NF, const double* triangle_space) {
	Eigen::Vector3f Nf = NF.col(f);
	Eigen::Vector3d N(Nf[0], Nf[1], Nf[2]);
	Eigen::Vector3d pt = (dir - dir.dot(N) * N).normalized();

	int prev_id = -1;
	int count = 0;

	while (len > 0) {
		count += 1;
		//printf("%d %f\n", count, len);
		Eigen::Vector3f t0f = V.col(F(0, f));
		Eigen::Vector3f t1f = V.col(F(1, f));
		Eigen::Vector3f t2f = V.col(F(2, f));
		Eigen::Vector3d t0(t0f[0], t0f[1], t0f[2]);
		Eigen::Vector3d t1(t1f[0], t1f[1], t1f[2]);
		Eigen::Vector3d t2(t2f[0], t2f[1], t2f[2]);
		t1 -= t0;
		t2 -= t0;

		int edge_id = f * 3;
		double max_len = 1e30;
		bool found = false;
		int next_id, next_f;
		Eigen::Vector3d next_q;

		double* triangle_space_f = ((double*)triangle_space) + f * 6;
		Eigen::Matrix<double, 2, 3> T = Eigen::Map<Eigen::Matrix<double, 2, 3> >(triangle_space_f);

		//const Eigen::MatrixXd& T = triangle_space[f];
		Eigen::VectorXd coord = T * (p - t0);
		Eigen::VectorXd dirs = (T * pt);

		double lens[3];
		lens[0] = -coord.y() / dirs.y();
		lens[1] = (1 - coord.x() - coord.y()) / (dirs.x() + dirs.y());
		lens[2] = -coord.x() / dirs.x();

		for (int fid = 0; fid < 3; ++fid) {
			if (fid + edge_id == prev_id)
				continue;

			if (lens[fid] >= 0 && lens[fid] < max_len) {
				max_len = lens[fid];
				next_id = E2E[edge_id + fid];
				next_f = next_id;
				if (next_f != -1)
					next_f /= 3;
				found = true;
			}
		}

		Eigen::Vector3f Nf_f = NF.col(f);
//		printf("status: %f %f %d\n", len, max_len, f);
		if (max_len >= len) {
			p = p + len * pt;
			len = 0;
			break;
		}
		p = t0 + t1 * (coord.x() + dirs.x() * max_len) + t2 * (coord.y() + dirs.y() * max_len);
		if (!found) {
			break;			
		}
		len -= max_len;
		if (next_f == -1) {
			break;
		}
		
		Eigen::Vector3f Nf_nf = NF.col(next_f);
		pt = rotate_vector_into_plane(pt, Eigen::Vector3d(Nf_f[0], Nf_f[1], Nf_f[2]), Eigen::Vector3d(Nf_nf[0], Nf_nf[1], Nf_nf[2]));
		f = next_f;
		prev_id = next_id;
	}
	return p;
}

void InitializeTangentSpace(void* vertices, void* faces, void* nf, int num_v, int num_f, void* tangent_space) {
	V = Eigen::Map<Eigen::MatrixXf>((float*)vertices, 3, num_v);
	F = Eigen::Map<Eigen::MatrixXi>((int*)faces, 3, num_f);
	FN = Eigen::Map<Eigen::MatrixXf>((float*)nf, 3, num_f);
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

void GetTangentNeighbors(float radius, void* positions, void* frames, void* finds, void* triangle_space, int num_v, void* out_p) {
#pragma omp parallel for
	for (int v = 0; v < num_v; ++v) {
		float* p_pos = ((float*)positions) + v * 3;
		float* p_frame = ((float*)frames) + v * 6;
		float* p_normal = p_frame + 3;
		float* out = ((float*)out_p) + v * 27;
		Eigen::Vector3d p(p_pos[0], p_pos[1], p_pos[2]);
		Eigen::Vector3d orient(p_frame[0], p_frame[1], p_frame[2]);
		Eigen::Vector3d normal(p_normal[0], p_normal[1], p_normal[2]);
		Eigen::Vector3d orient_y = normal.cross(orient);
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				Eigen::Vector3d dir = orient * i + orient_y * j; 
				double len = dir.norm();
				dir = dir / len;
				Eigen::Vector3d np = Travel(p, dir, len * radius, ((int*)finds)[v], E2E, V, F, FN, (double*)triangle_space);
				float* outp = out + ((i + 1) * 3 + (j + 1)) * 3;
				outp[0] = np[0];
				outp[1] = np[1];
				outp[2] = np[2];
			}
		}
	}
}


struct Coordinate {
	Coordinate()
	{}
	Coordinate(int f, const float* end, const float* from, const float* tex, const float* _ux) {
		float* n = ((float*)FN.data()) + f * 3;
		uy[0] = n[1] * _ux[2] - n[2] * _ux[1];
		uy[1] = n[2] * _ux[0] - n[0] * _ux[2];
		uy[2] = n[0] * _ux[1] - n[1] * _ux[0];
		float len = 1.0 / sqrt(uy[0] * uy[0] + uy[1] * uy[1] + uy[2] * uy[2]);
		uy[0] *= len;
		uy[1] *= len;
		uy[2] *= len;

		ux[0] = uy[1] * n[2] - uy[2] * n[1];
		ux[1] = uy[2] * n[0] - uy[0] * n[2];
		ux[2] = uy[0] * n[1] - uy[1] * n[0];

		float dx = ux[0] * (end[0] - from[0]) + ux[1] * (end[1] - from[1]) + ux[2] * (end[2] - from[2]);
		float dy = uy[0] * (end[0] - from[0]) + uy[1] * (end[1] - from[1]) + uy[2] * (end[2] - from[2]);
		coord[0] = tex[0] + dx;
		coord[1] = tex[1] + dy;
		find = f;
	}
	int find;
	float ux[3];
	float uy[3];
	float coord[2];
};
void GetNeighborhood(float radius,
	void* positions, void* frames, void* finds, int num_group_pts,
	void* pointcloud, void* pointcloud_finds, int num_pts,
	void* neighbor_inds, void* neighbor_tex, int num_indices)
{
	float radius2 = radius * radius;
	std::vector<int> indices(num_pts);
	std::iota(indices.begin(), indices.end(), 0);
	std::random_shuffle(indices.begin(), indices.end());
#pragma omp parallel for
	for (int pt_ind = 0; pt_ind < num_group_pts; ++pt_ind) {
		Eigen::MatrixXf pts = Eigen::Map<Eigen::MatrixXf>((float*)pointcloud, 3, num_pts);
		Eigen::Vector3f p = Eigen::Map<Eigen::Vector3f>(((float*)positions) + 3 * pt_ind);
		int find = ((int*)finds)[pt_ind];

		std::unordered_map<int, int> neighbor_faces;
		std::vector<Coordinate> q;
		q.reserve(1024);
		int front = 0;
		float* fv = (float*)FV.data();
		//q.push(std::make_pair(find, c));
		float tex[] = {0, 0};
		q.emplace_back(Coordinate(find, fv + find * 3, (float*)&p, tex, (float*)frames + 6 * pt_ind));
		neighbor_faces[find] = q.size() - 1;
		while (front < q.size()) {
			auto& info = q[front];
			int f = info.find;
			int* nf = E2E + f * 3;
			for (int i = 0; i < 3; ++i) {
				int next_f = *nf++;
				if (next_f == -1)
					continue;
				next_f /= 3;
				bool flag = false;
				for (int j = 0; j < 3; ++j) {
					auto diff = V.col(F(j, next_f)) - p;
					if (diff.squaredNorm() <= radius2 * 2) {
						flag = true;
						break;
					}
				}
				if (flag && !neighbor_faces.count(next_f)) {
					q.emplace_back(Coordinate(next_f, fv + next_f * 3, fv + f * 3, info.coord, info.ux));
					neighbor_faces[next_f] = q.size() - 1;
				}
			}
			front += 1;
		}
		int offset_i = rand() % indices.size();
		int offset = 0;
		int* neighbor_ind = ((int*)neighbor_inds) + num_indices * pt_ind;
		float* neighbor_texs = ((float*)neighbor_tex) + num_indices * pt_ind * 2;
		auto it = indices.begin() + offset_i;
		bool found = false;
		for (int i = 0; i < num_pts; ++i) {
			int seed = *it++;
			if (it == indices.end())
				it = indices.begin();
			Eigen::Vector3f qs = pts.col(seed) - p;
			if (qs.squaredNorm() <= radius2 * 2) {
				int find = ((int*)pointcloud_finds)[seed];
				auto it = neighbor_faces.find(find);
				if (it != neighbor_faces.end()) {
					auto diff = pts.col(seed) - FV.col(find);
					auto& coord = q[it->second];
					Eigen::Vector3f ux(coord.ux[0],coord.ux[1],coord.ux[2]);
					Eigen::Vector3f uy(coord.uy[0],coord.uy[1],coord.uy[2]);
					float x = coord.coord[0] + diff.dot(ux);
					float y = coord.coord[1] + diff.dot(uy);
					if (std::abs(x) <= radius && std::abs(y) <= radius) {
						if (offset == num_indices) {
							if (x < 1e-5 && y < 1e-5) {
								neighbor_texs[0] = x;
								neighbor_texs[1] = y;
								neighbor_ind[0] = seed;
								break;
							}
						}
						else {
							if (x < 1e-5 && y < 1e-5) {
								found = true;
							}
							neighbor_texs[offset * 2] = x;
							neighbor_texs[offset * 2 + 1] = y;
							neighbor_ind[offset++] = seed;
							if (offset == num_indices && found)
								break;
						}
					}
				}
			}
		}
		for (; offset < num_indices; ++offset) {
			neighbor_ind[offset] = neighbor_ind[0];
			neighbor_texs[offset * 2] = 1e30;
			neighbor_texs[offset * 2 + 1] = 1e30;
		}
	}
}



void ComputeMask(void* maskp, void* indicesp, int num) {
	std::unordered_set<int> uset;
	int* mask = (int*)maskp;
	int* indices = (int*)indicesp;
	for (int i = 0; i < num; ++i) {
		int d = indices[i];
		if (uset.count(d)) {
			mask[i] = 0;
		} else {
			mask[i] = 1;
			uset.insert(d);
		}
	}
}
}