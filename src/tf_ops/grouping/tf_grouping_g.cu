#include <iostream>

// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
                float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx1 (b,m,nsample), idx2 (b,m,nsample), idx3 (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_level_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx1, int *idx2, int *idx3, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx1 += m*nsample*batch_index;
    idx2 += m*nsample*batch_index;
    idx3 += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt1 = 0;
        int cnt2 = 0;
        int cnt3 = 0;
        int idxs = -1;
        for (int k=0;k<n;++k) {
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
            float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d < 1e-5)
                idxs = k;
            //0.577
            if (d<radius*0.577) {
                if (cnt1 < nsample) {
                    if (cnt1==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx1[j*nsample+l] = k;
                    }
                    idx1[j*nsample+cnt1] = k;
                    cnt1+=1;
                }
            }
            //0.816
            if (d<radius*0.816 && d >= radius * 0.577) {
                if (cnt2 < nsample) {
                    if (cnt2==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx2[j*nsample+l] = k;
                    }
                    idx2[j*nsample+cnt2] = k;
                    cnt2+=1;
                }
            }
            if (d<radius && d >= radius * 0.816) {
                if (cnt3 < nsample) {
                    if (cnt3==0) {
                        for (int l=0;l<nsample;++l)
                            idx3[j*nsample+l] = k;
                    }
                    idx3[j*nsample+cnt3] = k;
                    cnt3+=1;
                }
            }
        }
        idxs = -(idxs + 1);
        if (cnt1 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx1[j*nsample+l] = idxs;
        }
        if (cnt2 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx2[j*nsample+l] = idxs;
        }
        if (cnt3 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx3[j*nsample+l] = idxs;
        }
        pts_cnt[j] = cnt1+cnt2+cnt3;
    }
}

// input: radius (1), nsample (1), tangent (b,n,2)
// output: idx1 (b,n,nsample), idx2 (b,m,nsample), idx3 (b,n,nsample), pts_cnt (b,n)
__global__ void query_tangent_point_level_gpu(int b, int n, int m, float radius, int nsample, const float *tangent, const int* group, int *idx1, int *idx2, int *idx3, int *pts_cnt) {
    int batch_index = blockIdx.x;
    tangent += n*m*2*batch_index;
    group += n*m*batch_index;

    idx1 += n*nsample*batch_index;
    idx2 += n*nsample*batch_index;
    idx3 += n*nsample*batch_index;

    pts_cnt += n*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<n;j+=stride) {
        int cnt1 = 0;
        int cnt2 = 0;
        int cnt3 = 0;
        int idxs = -1;
        for (int k=0;k<m;++k) {
            float tx = std::abs(tangent[(j*m+k)*2]);
            float ty = std::abs(tangent[(j*m+k)*2+1]);
            if (tx < 1e-5 && ty < 1e-5)
                idxs = group[j*m+k];
            //0.577
            tx /= radius;
            ty /= radius;
            if (tx <= 0.5 && ty <= 0.5) {
                if (cnt1 < nsample) {
                    if (cnt1==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx1[j*nsample+l] = group[j*m+k];
                    }
                    idx1[j*nsample+cnt1] = group[j*m+k];
                    cnt1+=1;
                }
            }
            //0.816
            else if (tx > 0.5 && ty > 0.5) {
                if (cnt2 < nsample) {
                    if (cnt2==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx2[j*nsample+l] = group[j*m+k];
                    }
                    idx2[j*nsample+cnt2] = group[j*m+k];
                    cnt2+=1;
                }
            }
            else {
                if (cnt3 < nsample) {
                    if (cnt3==0) {
                        for (int l=0;l<nsample;++l)
                            idx3[j*nsample+l] = group[j*m+k];
                    }
                    idx3[j*nsample+cnt3] = group[j*m+k];
                    cnt3+=1;
                }
            }
        }
        idxs = -(idxs + 1);
        if (cnt1 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx1[j*nsample+l] = idxs;
        }
        if (cnt2 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx2[j*nsample+l] = idxs;
        }
        if (cnt3 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx3[j*nsample+l] = idxs;
        }
        pts_cnt[j] = cnt1+cnt2+cnt3;
    }
}

// input: radius (1), nsample (1), tangent (b,n,2)
// output: idx1 (b,n,nsample), idx2 (b,m,nsample), idx3 (b,n,nsample), pts_cnt (b,n)
__global__ void query_radius_point_level_gpu(int b, int n, int m, float radius, int nsample, const float *tangent, const int* group, int *idx1, int *idx2, int *idx3, int *pts_cnt) {
    int batch_index = blockIdx.x;
    tangent += n*m*2*batch_index;
    group += n*m*batch_index;

    idx1 += n*nsample*batch_index;
    idx2 += n*nsample*batch_index;
    idx3 += n*nsample*batch_index;

    pts_cnt += n*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<n;j+=stride) {
        int cnt1 = 0;
        int cnt2 = 0;
        int cnt3 = 0;
        int idxs = -1;
        for (int k=0;k<m;++k) {
            float tx = std::abs(tangent[(j*m+k)*2]);
            float ty = std::abs(tangent[(j*m+k)*2+1]);
            if (tx < 1e-5 && ty < 1e-5)
                idxs = group[j*m+k];
            //0.577
            tx /= radius;
            ty /= radius;
            float sum_r = sqrt(tx * tx + ty * ty);
            if (sum_r < 0.577) {
                if (cnt1 < nsample) {
                    if (cnt1==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx1[j*nsample+l] = group[j*m+k];
                    }
                    idx1[j*nsample+cnt1] = group[j*m+k];
                    cnt1+=1;
                }
            }
            //0.816
            else if (tx > 0.816) {
                if (cnt2 < nsample) {
                    if (cnt2==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                        for (int l=0;l<nsample;++l)
                            idx2[j*nsample+l] = group[j*m+k];
                    }
                    idx2[j*nsample+cnt2] = group[j*m+k];
                    cnt2+=1;
                }
            }
            else {
                if (cnt3 < nsample) {
                    if (cnt3==0) {
                        for (int l=0;l<nsample;++l)
                            idx3[j*nsample+l] = group[j*m+k];
                    }
                    idx3[j*nsample+cnt3] = group[j*m+k];
                    cnt3+=1;
                }
            }
        }
        idxs = -(idxs + 1);
        if (cnt1 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx1[j*nsample+l] = idxs;
        }
        if (cnt2 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx2[j*nsample+l] = idxs;
        }
        if (cnt3 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx3[j*nsample+l] = idxs;
        }
        pts_cnt[j] = cnt1+cnt2+cnt3;
    }
}

// input: radius (1), nsample (1), tangent (b,n,2)
// output: idx1 (b,n,nsample), idx2 (b,m,nsample), idx3 (b,n,nsample), pts_cnt (b,n)
__global__ void query_radius_angle_point_level_gpu(int b, int n, int m, float start_angle, float radius, int nsample, const float *tangent, const int* group, int *idx1, int *idx2, int *idx3, int *idx4, int *idx5, int *idx6, int *idx7, int *idx8, int *idx9, int *pts_cnt) {
    int batch_index = blockIdx.x;
    tangent += n*m*2*batch_index;
    group += n*m*batch_index;

    idx1 += n*nsample*batch_index;
    idx2 += n*nsample*batch_index;
    idx3 += n*nsample*batch_index;
    idx4 += n*nsample*batch_index;
    idx5 += n*nsample*batch_index;
    idx6 += n*nsample*batch_index;
    idx7 += n*nsample*batch_index;
    idx8 += n*nsample*batch_index;
    idx9 += n*nsample*batch_index;

    pts_cnt += n*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<n;j+=stride) {
        int cnt1 = 0;
        int cnt2 = 0;
        int cnt3 = 0;
        int cnt4 = 0;
        int cnt5 = 0;
        int cnt6 = 0;
        int cnt7 = 0;
        int cnt8 = 0;
        int cnt9 = 0;
        int idxs = -1;
        for (int k=0;k<m;++k) {
            float angle = atan2(tangent[(j*m+k)*2],tangent[(j*m+k)*2+1]) / 3.141592654 * 180.0 + start_angle;
            int angle_diff = (int)angle % 360 / 120;

            float tx = std::abs(tangent[(j*m+k)*2]);
            float ty = std::abs(tangent[(j*m+k)*2+1]);
            if (tx < 1e-5 && ty < 1e-5)
                idxs = group[j*m+k];
            //0.577
            tx /= radius;
            ty /= radius;
            float sum_r = sqrt(tx * tx + ty * ty);
            if (sum_r < 0.577) {
                if (angle_diff == 0) {
                    if (cnt1 < nsample) {
                        if (cnt1==0) {
                            for (int l=0;l<nsample;++l)
                                idx1[j*nsample+l] = group[j*m+k];
                        }
                        idx1[j*nsample+cnt1] = group[j*m+k], cnt1 += 1;
                    }
                }
                else if (angle_diff == 1) {
                    if (cnt2 < nsample) {
                        if (cnt2==0) {
                            for (int l=0;l<nsample;++l)
                                idx2[j*nsample+l] = group[j*m+k];
                        }
                        idx2[j*nsample+cnt2] = group[j*m+k], cnt2 += 1;
                    }
                }
                else {
                    if (cnt3 < nsample) {
                        if (cnt3==0) {
                            for (int l=0;l<nsample;++l)
                                idx3[j*nsample+l] = group[j*m+k];
                        }
                        idx3[j*nsample+cnt3] = group[j*m+k], cnt3 += 1;
                    }
                }
            }
            //0.816
            else if (tx > 0.816) {
                if (angle_diff == 0) {
                    if (cnt4 < nsample) {
                        if (cnt4==0) {
                            for (int l=0;l<nsample;++l)
                                idx4[j*nsample+l] = group[j*m+k];
                        }
                        idx4[j*nsample+cnt4] = group[j*m+k], cnt4 += 1;
                    }
                }
                else if (angle_diff == 1) {
                    if (cnt5 < nsample) {
                        if (cnt5==0) {
                            for (int l=0;l<nsample;++l)
                                idx5[j*nsample+l] = group[j*m+k];
                        }
                        idx5[j*nsample+cnt5] = group[j*m+k], cnt5 += 1;
                    }
                }
                else {
                    if (cnt6 < nsample) {
                        if (cnt6==0) {
                            for (int l=0;l<nsample;++l)
                                idx6[j*nsample+l] = group[j*m+k];
                        }
                        idx6[j*nsample+cnt6] = group[j*m+k], cnt6 += 1;
                    }
                }
            }
            else {
                if (angle_diff == 0) {
                    if (cnt7 < nsample) {
                        if (cnt7==0) {
                            for (int l=0;l<nsample;++l)
                                idx7[j*nsample+l] = group[j*m+k];
                        }
                        idx7[j*nsample+cnt7] = group[j*m+k], cnt7 += 1;
                    }
                }
                else if (angle_diff == 1) {
                    if (cnt8 < nsample) {
                        if (cnt8==0) {
                            for (int l=0;l<nsample;++l)
                                idx8[j*nsample+l] = group[j*m+k];
                        }
                        idx8[j*nsample+cnt8] = group[j*m+k], cnt8 += 1;
                    }
                }
                else {
                    if (cnt9 < nsample) {
                        if (cnt9==0) {
                            for (int l=0;l<nsample;++l)
                                idx9[j*nsample+l] = group[j*m+k];
                        }
                        idx9[j*nsample+cnt9] = group[j*m+k], cnt9 += 1;
                    }
                }
            }
        }
        idxs = -(idxs + 1);
        if (cnt1 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx1[j*nsample+l] = idxs;
        }
        if (cnt2 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx2[j*nsample+l] = idxs;
        }
        if (cnt3 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx3[j*nsample+l] = idxs;
        }
        if (cnt4 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx4[j*nsample+l] = idxs;
        }
        if (cnt5 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx5[j*nsample+l] = idxs;
        }
        if (cnt6 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx6[j*nsample+l] = idxs;
        }
        if (cnt7 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx7[j*nsample+l] = idxs;
        }
        if (cnt8 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx8[j*nsample+l] = idxs;
        }
        if (cnt9 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx9[j*nsample+l] = idxs;
        }
        pts_cnt[j] = cnt1+cnt2+cnt3+cnt4+cnt5+cnt6+cnt7+cnt8+cnt9;
    }
}

// input: radius (1), nsample (1), tangent (b,n,2)
// output: idx1 (b,n,nsample), idx2 (b,m,nsample), idx3 (b,n,nsample), pts_cnt (b,n)
__global__ void query_tangent9_point_level_gpu(int b, int n, int m, float start_angle, float radius, int nsample, const float *tangent, const int* group, int *idx1, int *idx2, int *idx3, int *idx4, int *idx5, int *idx6, int *idx7, int *idx8, int *idx9, int *pts_cnt) {
    int batch_index = blockIdx.x;
    tangent += n*m*2*batch_index;
    group += n*m*batch_index;

    idx1 += n*nsample*batch_index;
    idx2 += n*nsample*batch_index;
    idx3 += n*nsample*batch_index;
    idx4 += n*nsample*batch_index;
    idx5 += n*nsample*batch_index;
    idx6 += n*nsample*batch_index;
    idx7 += n*nsample*batch_index;
    idx8 += n*nsample*batch_index;
    idx9 += n*nsample*batch_index;

    pts_cnt += n*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<n;j+=stride) {
        int cnt1 = 0;
        int cnt2 = 0;
        int cnt3 = 0;
        int cnt4 = 0;
        int cnt5 = 0;
        int cnt6 = 0;
        int cnt7 = 0;
        int cnt8 = 0;
        int cnt9 = 0;
        int idxs = -1;
        for (int k=0;k<m;++k) {

            float tx = tangent[(j*m+k)*2];
            float ty = tangent[(j*m+k)*2+1];

            if (abs(tx) < 1e-5 && abs(ty) < 1e-5)
                idxs = group[j*m+k];
            //0.577
            tx /= radius;
            ty /= radius;

            int angle_diff = 0;
            if (tx > -0.5)
                angle_diff = 1;
            if (tx > 0.5)
                angle_diff = 2;
            
            if (ty < -0.5) {
                if (angle_diff == 0) {
                    if (cnt1 < nsample) {
                        if (cnt1==0) {
                            for (int l=0;l<nsample;++l)
                                idx1[j*nsample+l] = group[j*m+k];
                        }
                        idx1[j*nsample+cnt1] = group[j*m+k], cnt1 += 1;
                    }
                }
                else if (angle_diff == 1) {
                    if (cnt2 < nsample) {
                        if (cnt2==0) {
                            for (int l=0;l<nsample;++l)
                                idx2[j*nsample+l] = group[j*m+k];
                        }
                        idx2[j*nsample+cnt2] = group[j*m+k], cnt2 += 1;
                    }
                }
                else {
                    if (cnt3 < nsample) {
                        if (cnt3==0) {
                            for (int l=0;l<nsample;++l)
                                idx3[j*nsample+l] = group[j*m+k];
                        }
                        idx3[j*nsample+cnt3] = group[j*m+k], cnt3 += 1;
                    }
                }
            }
            //0.816
            else if (ty > 0.5) {
                if (angle_diff == 0) {
                    if (cnt4 < nsample) {
                        if (cnt4==0) {
                            for (int l=0;l<nsample;++l)
                                idx4[j*nsample+l] = group[j*m+k];
                        }
                        idx4[j*nsample+cnt4] = group[j*m+k], cnt4 += 1;
                    }
                }
                else if (angle_diff == 1) {
                    if (cnt5 < nsample) {
                        if (cnt5==0) {
                            for (int l=0;l<nsample;++l)
                                idx5[j*nsample+l] = group[j*m+k];
                        }
                        idx5[j*nsample+cnt5] = group[j*m+k], cnt5 += 1;
                    }
                }
                else {
                    if (cnt6 < nsample) {
                        if (cnt6==0) {
                            for (int l=0;l<nsample;++l)
                                idx6[j*nsample+l] = group[j*m+k];
                        }
                        idx6[j*nsample+cnt6] = group[j*m+k], cnt6 += 1;
                    }
                }
            }
            else {
                if (angle_diff == 0) {
                    if (cnt7 < nsample) {
                        if (cnt7==0) {
                            for (int l=0;l<nsample;++l)
                                idx7[j*nsample+l] = group[j*m+k];
                        }
                        idx7[j*nsample+cnt7] = group[j*m+k], cnt7 += 1;
                    }
                }
                else if (angle_diff == 1) {
                    if (cnt8 < nsample) {
                        if (cnt8==0) {
                            for (int l=0;l<nsample;++l)
                                idx8[j*nsample+l] = group[j*m+k];
                        }
                        idx8[j*nsample+cnt8] = group[j*m+k], cnt8 += 1;
                    }
                }
                else {
                    if (cnt9 < nsample) {
                        if (cnt9==0) {
                            for (int l=0;l<nsample;++l)
                                idx9[j*nsample+l] = group[j*m+k];
                        }
                        idx9[j*nsample+cnt9] = group[j*m+k], cnt9 += 1;
                    }
                }
            }
        }
        idxs = -(idxs + 1);
        if (cnt1 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx1[j*nsample+l] = idxs;
        }
        if (cnt2 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx2[j*nsample+l] = idxs;
        }
        if (cnt3 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx3[j*nsample+l] = idxs;
        }
        if (cnt4 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx4[j*nsample+l] = idxs;
        }
        if (cnt5 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx5[j*nsample+l] = idxs;
        }
        if (cnt6 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx6[j*nsample+l] = idxs;
        }
        if (cnt7 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx7[j*nsample+l] = idxs;
        }
        if (cnt8 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx8[j*nsample+l] = idxs;
        }
        if (cnt9 == 0) {
            for (int l = 0; l < nsample; ++l)
                idx9[j*nsample+l] = idxs;
        }
        pts_cnt[j] = cnt1+cnt2+cnt3+cnt4+cnt5+cnt6+cnt7+cnt8+cnt9;
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out, int relative) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                if (ii < 0) {
                    if (relative == 0)
                        out[j * nsample * c + k * c + l] = 0;
                    else
                        out[j * nsample * c + k * c + l] = points[(-ii-1)*c+l];
                } else {
                    out[j*nsample*c+k*c+l] = points[ii*c+l];
                }
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points, int relative) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                if (ii < 0) {
                    if (relative == 1)
                        atomicAdd(&grad_points[(-ii-1)*c+l], grad_out[j*nsample*c+k*c+l]);
                } else {
                    atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
                }
            }
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<b,256>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryBallPointLevelLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx1, int* idx2, int* idx3, int *pts_cnt) {
    query_ball_point_level_gpu<<<b,256>>>(b,n,m,radius,nsample,xyz1,xyz2,idx1,idx2,idx3,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryTangentPointLevelLauncher(int b, int n, int m, float radius, int nsample, const float *tangent, const int* group, int *idx1, int* idx2, int* idx3, int *pts_cnt) {
    query_tangent_point_level_gpu<<<b,256>>>(b,n,m,radius,nsample,tangent,group,idx1,idx2,idx3,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryRadiusPointLevelLauncher(int b, int n, int m, float radius, int nsample, const float *tangent, const int* group, int *idx1, int* idx2, int* idx3, int *pts_cnt) {
    query_radius_point_level_gpu<<<b,256>>>(b,n,m,radius,nsample,tangent,group,idx1,idx2,idx3,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryRadiusAnglePointLevelLauncher(int b, int n, int m, float start_angle, float radius, int nsample, const float *tangent, const int* group,
    int *idx1, int *idx2, int *idx3, int* idx4, int* idx5, int* idx6, int* idx7, int* idx8, int* idx9, int *pts_cnt)
{
    query_radius_angle_point_level_gpu<<<b,256>>>(b,n,m,start_angle,radius,nsample,tangent,group,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9,pts_cnt);
}
void queryTangent9PointLevelLauncher(int b, int n, int m, float start_angle, float radius, int nsample, const float *tangent, const int* group,
    int *idx1, int *idx2, int *idx3, int* idx4, int* idx5, int* idx6, int* idx7, int* idx8, int* idx9, int *pts_cnt)
{
    query_tangent9_point_level_gpu<<<b,256>>>(b,n,m,start_angle,radius,nsample,tangent,group,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9,pts_cnt);
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out, int relative){
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out,relative);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points, int relative){
    group_point_grad_gpu<<<b,256>>>(b,n,c,m,nsample,grad_out,idx,grad_points,relative);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}
