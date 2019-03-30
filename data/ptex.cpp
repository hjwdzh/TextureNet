#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

int colorWidth, colorHeight, depthWidth, depthHeight;
Eigen::Matrix4f colorIntrinsic;
Eigen::Matrix4f depthIntrinsic;
int frame_size;

void ParseInfo(const char* filename) {
	std::ifstream is(filename);
	char buffer[1024];
	is.getline(buffer, 1024);
	is.getline(buffer, 1024);
	is >> buffer >> buffer >> colorWidth;
	is >> buffer >> buffer >> colorHeight;
	is >> buffer >> buffer >> depthWidth;
	is >> buffer >> buffer >> depthHeight;
	is.getline(buffer, 1024);
	is.getline(buffer, 1024);
	is >> buffer >> buffer;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			is >> colorIntrinsic(i, j);
		}
	}
	is.getline(buffer, 1024);
	is.getline(buffer, 1024);
	is >> buffer >> buffer;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			is >> depthIntrinsic(i, j);
		}
	}
	is.getline(buffer, 1024);
	is.getline(buffer, 1024);
	is >> buffer >> buffer >> frame_size;

}

void LoadFrames(const char* folder, int i, cv::Mat& color, cv::Mat& depth, Eigen::Matrix4f& pose) {
	char buffer[1024];
	sprintf(buffer, "%s/image_data/frame-%06d.color.jpg", folder, i);
	color = cv::imread(buffer);
	sprintf(buffer, "%s/image_data/frame-%06d.depth.pgm", folder, i);
	depth = cv::imread(buffer, cv::IMREAD_UNCHANGED);
	sprintf(buffer, "%s/image_data/frame-%06d.pose.txt", folder, i);
	std::ifstream is(buffer);
	for (int j = 0; j < 4; ++j) {
		for (int k = 0; k < 4; ++k) {
			is >> pose(j, k);
		}
	}
	is.close();
	pose = pose.inverse().eval();
}
int main(int argc, char** argv)
{
	char buffer[1024];
	sprintf(buffer, "%s/image_data/_info.txt", argv[1]);
	printf("%s\n", buffer);
	ParseInfo(buffer);
	sprintf(buffer, "%s/%s", argv[1], argv[2]);
	std::ifstream is(buffer);
	float x, y, z;
	std::vector<Eigen::Vector3f> vertices;
	while (is >> x) {
		is >> y >> z;
		vertices.push_back(Eigen::Vector3f(x, y, z));
	}
	is.close();
	sprintf(buffer, "%s/%s", argv[1], argv[3]);
	is.open(buffer);
	float tx, ty, tz, nx, ny, nz;
	std::vector<Eigen::Vector3f> orients(vertices.size()), normals(vertices.size());
	for (int i = 0; i < vertices.size(); ++i) {
		is >> orients[i][0] >> orients[i][1] >> orients[i][2] >> normals[i][0] >> normals[i][1] >> normals[i][2];
	}
	is.close();
	for (int i = 0; i < vertices.size(); ++i) {
		Eigen::Vector3f qx = orients[i];
		Eigen::Vector3f n = normals[i];
		Eigen::Vector3f qy = n.cross(qx);
		if (qx[1] > 0)
			qx = -qx;
		if (qy[1] > 0)
			qy = -qy;
		if (qx[1] > qy[1])
			qx = qy;
		orients[i] = qx;
	}

	int half_kernel = 5;
	int kernel = 2 * half_kernel;
	float pixel_width = 0.004;
	int kernel2 = kernel * kernel;
	std::vector<Eigen::Vector3f> textiles(vertices.size() * kernel2, Eigen::Vector3f(0, 0, 0));
	std::vector<cv::Vec3f> textile_colors(vertices.size() * kernel2, cv::Vec3f(0, 0, 0));
	std::vector<int> textile_count(vertices.size() * kernel2, 0);

	for (int i = 0; i < vertices.size(); ++i) {
		Eigen::Vector3f p = vertices[i];
		Eigen::Vector3f tx = orients[i];
		Eigen::Vector3f ty = normals[i].cross(orients[i]);
		for (int j = -half_kernel; j < half_kernel; ++j) {
			for (int k = -half_kernel; k < half_kernel; ++k) {
				Eigen::Vector3f pt = p + tx * (j + 0.5f) * pixel_width + ty * (k + 0.5f) * pixel_width;
				textiles[i * kernel2 + (j + half_kernel) * kernel + k + half_kernel] = pt;
			}
		}		
	}

	std::vector<cv::Vec3b> vertices_colors(vertices.size(), cv::Vec3b(0, 0, 0));
	std::vector<cv::Vec3b> temp_colors(kernel2);
	std::vector<int> temp_mask(kernel2);
	std::vector<double> variance(vertices.size(), 1e30);

	for (int j = 0; j < frame_size; ++j) {
		printf("%d of %d\n", j, frame_size);
		cv::Mat color, depth;
		Eigen::Matrix4f pose;
		LoadFrames(argv[1], j, color, depth, pose);
		for (int i = 0; i < vertices.size(); ++i) {
			Eigen::Vector4f p(vertices[i][0], vertices[i][1], vertices[i][2], 1);			
			p = pose * p;
			float z = p[2];
			if (z <= 0.3)
				continue;
			float visual = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
			p = p / p[2];
			Eigen::Vector4f pd = depthIntrinsic * p;
			int px = pd[0] + 0.5;
			int py = pd[1] + 0.5;
			if (px >= 0 && px < depthWidth && py >= 0 && py < depthHeight) {
				double d = depth.at<unsigned short>(py, px) * 1e-3;
				double dis = std::abs(d - z);
				if (dis < 0.1) {
					float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
					for (int k = 0; k < kernel2; ++k) {
						Eigen::Vector3f pc3 = textiles[i * kernel2 + k];
						Eigen::Vector4f pc(pc3[0], pc3[1], pc3[2], 1);
						pc = pose * pc;
						pc = pc / pc[2];
						pc = colorIntrinsic * pc;
						px = pc[0] + 0.5;
						py = pc[1] + 0.5;
						if (px < colorWidth - 100 && py < colorHeight - 100 && px >= 100 && py >= 100) {
							temp_colors[k] = color.at<cv::Vec3b>(py, px);
							temp_mask[k] = 1;
							r += temp_colors[k].val[0];
							g += temp_colors[k].val[1];
							b += temp_colors[k].val[2];
							a += 1;
						} else {
							temp_mask[k] = 0;
						}
					}
					if (a == 0.0)
						continue;

					r /= a;
					g /= a;
					b /= a;
					for (int k = 0; k < kernel2; ++k) {
						if (temp_mask[k] == 0) {
							temp_colors[k] = cv::Vec3b(r, g, b);
						}
					}
					if (visual < variance[i]) {
						variance[i] = visual;
						for (int k = 0; k < kernel2; ++k) {
							auto p = temp_colors[k];
							textile_colors[i * kernel2 + k] = cv::Vec3f(p.val[0], p.val[1], p.val[2]);
							//textile_count[i * kernel2 + k] += 1;
						}
					}

				}
			}
		}
	}
	
	for (int i = 0; i < textile_colors.size(); ++i) {
		if (textile_count[i] != 0) {
			//textile_colors[i] /= (float)textile_count[i];
		}
	}
	for (int i = 0; i < textile_colors.size(); ++i) {
		textile_colors[i] /= 255.0f;
	}
	sprintf(buffer, "%s/%s", argv[1], argv[4]);
	FILE* fp = fopen(buffer, "wb");
	fwrite(textile_colors.data(), sizeof(cv::Vec3f), textile_colors.size(), fp);
	fclose(fp);
	return 0;
}
