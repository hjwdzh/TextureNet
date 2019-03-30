#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("QueryBallPoint")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("QueryBallPointLevel")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx1: int32")
    .Output("idx2: int32")
    .Output("idx3: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(3, output4);
        return Status::OK();
    });
REGISTER_OP("QueryTangentPointLevel")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("tangent: float32")
    .Input("group: int32")
    .Output("idx1: int32")
    .Output("idx2: int32")
    .Output("idx3: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(0), 4, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(3, output4);
        return Status::OK();
    });
REGISTER_OP("QueryRadiusPointLevel")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("tangent: float32")
    .Input("group: int32")
    .Output("idx1: int32")
    .Output("idx2: int32")
    .Output("idx3: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(0), 4, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(3, output4);
        return Status::OK();
    });
REGISTER_OP("QueryRadiusAnglePointLevel")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("tangent: float32")
    .Input("group: int32")
    .Output("idx1: int32")
    .Output("idx2: int32")
    .Output("idx3: int32")
    .Output("idx4: int32")
    .Output("idx5: int32")
    .Output("idx6: int32")
    .Output("idx7: int32")
    .Output("idx8: int32")
    .Output("idx9: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(0), 4, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(3, output4);
        ::tensorflow::shape_inference::ShapeHandle output5 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(4, output5);
        ::tensorflow::shape_inference::ShapeHandle output6 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(5, output6);
        ::tensorflow::shape_inference::ShapeHandle output7 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(6, output7);
        ::tensorflow::shape_inference::ShapeHandle output8 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(7, output8);
        ::tensorflow::shape_inference::ShapeHandle output9 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(8, output9);
        ::tensorflow::shape_inference::ShapeHandle output10 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(10, output10);
        return Status::OK();
    });
REGISTER_OP("QueryTangent9PointLevel")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("tangent: float32")
    .Input("group: int32")
    .Output("idx1: int32")
    .Output("idx2: int32")
    .Output("idx3: int32")
    .Output("idx4: int32")
    .Output("idx5: int32")
    .Output("idx6: int32")
    .Output("idx7: int32")
    .Output("idx8: int32")
    .Output("idx9: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(0), 4, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(1, output2);
        ::tensorflow::shape_inference::ShapeHandle output3 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(2, output3);
        ::tensorflow::shape_inference::ShapeHandle output4 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(3, output4);
        ::tensorflow::shape_inference::ShapeHandle output5 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(4, output5);
        ::tensorflow::shape_inference::ShapeHandle output6 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(5, output6);
        ::tensorflow::shape_inference::ShapeHandle output7 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(6, output7);
        ::tensorflow::shape_inference::ShapeHandle output8 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(7, output8);
        ::tensorflow::shape_inference::ShapeHandle output9 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(8, output9);
        ::tensorflow::shape_inference::ShapeHandle output10 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(10, output10);
        return Status::OK();
    });

REGISTER_OP("SelectionSort")
    .Attr("k: int")
    .Input("dist: float32")
    .Output("outi: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("GroupPoint")
    .Attr("relative: int")
    .Input("points: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims2, 2), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("GroupPointGrad")
    .Attr("relative: int")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
class QueryBallPointGpuOp : public OpKernel {
    public:
        explicit QueryBallPointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU), QueryBallPointGpuOp);

void queryBallPointLevelLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx1, int *idx2, int *idx3, int *pts_cnt);
class QueryBallPointLevelGpuOp : public OpKernel {
    public:
        explicit QueryBallPointLevelGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor1 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor1));
            Tensor *idx_tensor2 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,nsample_}, &idx_tensor2));
            Tensor *idx_tensor3 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,m,nsample_}, &idx_tensor3));

            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat1 = idx_tensor1->flat<int>();
            int *idx1 = &(idx_flat1(0));
            auto idx_flat2 = idx_tensor2->flat<int>();
            int *idx2 = &(idx_flat2(0));
            auto idx_flat3 = idx_tensor3->flat<int>();
            int *idx3 = &(idx_flat3(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointLevelLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx1,idx2,idx3,pts_cnt);
            //queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx1,pts_cnt);
            //queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx2,pts_cnt);
            //queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx3,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPointLevel").Device(DEVICE_GPU), QueryBallPointLevelGpuOp);

void queryTangentPointLevelLauncher(int b, int n, int m, float radius, int nsample, const float *tangent, const int* group, int *idx1, int *idx2, int *idx3, int *pts_cnt);
class QueryTangentPointLevelGpuOp : public OpKernel {
    public:
        explicit QueryTangentPointLevelGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& tangent_tensor = context->input(0);
            OP_REQUIRES(context, tangent_tensor.dims()==4 && tangent_tensor.shape().dim_size(3)==2, errors::InvalidArgument("QueryTangentPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = tangent_tensor.shape().dim_size(0);
            int n = tangent_tensor.shape().dim_size(1);
            int m = tangent_tensor.shape().dim_size(2);
            const Tensor& group_tensor = context->input(1);
            Tensor *idx_tensor1 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,nsample_}, &idx_tensor1));
            Tensor *idx_tensor2 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,nsample_}, &idx_tensor2));
            Tensor *idx_tensor3 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,n,nsample_}, &idx_tensor3));

            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,n}, &pts_cnt_tensor));

            auto tangent_flat = tangent_tensor.flat<float>();
            const float *tangent = &(tangent_flat(0));
            auto group_flat = group_tensor.flat<int>();
            const int* group = &(group_flat(0));
            auto idx_flat1 = idx_tensor1->flat<int>();
            int *idx1 = &(idx_flat1(0));
            auto idx_flat2 = idx_tensor2->flat<int>();
            int *idx2 = &(idx_flat2(0));
            auto idx_flat3 = idx_tensor3->flat<int>();
            int *idx3 = &(idx_flat3(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryTangentPointLevelLauncher(b,n,m,radius_,nsample_,tangent,group,idx1,idx2,idx3,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx1,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx2,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx3,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryTangentPointLevel").Device(DEVICE_GPU), QueryTangentPointLevelGpuOp);

void queryRadiusPointLevelLauncher(int b, int n, int m, float radius, int nsample, const float *tangent, const int* group, int *idx1, int *idx2, int *idx3, int *pts_cnt);
class QueryRadiusPointLevelGpuOp : public OpKernel {
    public:
        explicit QueryRadiusPointLevelGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& tangent_tensor = context->input(0);
            OP_REQUIRES(context, tangent_tensor.dims()==4 && tangent_tensor.shape().dim_size(3)==2, errors::InvalidArgument("QueryTangentPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = tangent_tensor.shape().dim_size(0);
            int n = tangent_tensor.shape().dim_size(1);
            int m = tangent_tensor.shape().dim_size(2);
            const Tensor& group_tensor = context->input(1);
            Tensor *idx_tensor1 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,nsample_}, &idx_tensor1));
            Tensor *idx_tensor2 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,nsample_}, &idx_tensor2));
            Tensor *idx_tensor3 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,n,nsample_}, &idx_tensor3));

            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,n}, &pts_cnt_tensor));

            auto tangent_flat = tangent_tensor.flat<float>();
            const float *tangent = &(tangent_flat(0));
            auto group_flat = group_tensor.flat<int>();
            const int* group = &(group_flat(0));
            auto idx_flat1 = idx_tensor1->flat<int>();
            int *idx1 = &(idx_flat1(0));
            auto idx_flat2 = idx_tensor2->flat<int>();
            int *idx2 = &(idx_flat2(0));
            auto idx_flat3 = idx_tensor3->flat<int>();
            int *idx3 = &(idx_flat3(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryRadiusPointLevelLauncher(b,n,m,radius_,nsample_,tangent,group,idx1,idx2,idx3,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx1,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx2,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx3,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryRadiusPointLevel").Device(DEVICE_GPU), QueryRadiusPointLevelGpuOp);

void queryRadiusAnglePointLevelLauncher(int b, int n, int m, float start_angle, float radius, int nsample, const float *tangent, const int* group,
    int *idx1, int *idx2, int *idx3, int* idx4, int* idx5, int* idx6, int* idx7, int* idx8, int* idx9, int *pts_cnt);
class QueryRadiusAnglePointLevelGpuOp : public OpKernel {
    public:
        explicit QueryRadiusAnglePointLevelGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& tangent_tensor = context->input(0);
            OP_REQUIRES(context, tangent_tensor.dims()==4 && tangent_tensor.shape().dim_size(3)==2, errors::InvalidArgument("QueryTangentPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = tangent_tensor.shape().dim_size(0);
            int n = tangent_tensor.shape().dim_size(1);
            int m = tangent_tensor.shape().dim_size(2);
            const Tensor& group_tensor = context->input(1);
            Tensor *idx_tensor1 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,nsample_}, &idx_tensor1));
            Tensor *idx_tensor2 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,nsample_}, &idx_tensor2));
            Tensor *idx_tensor3 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,n,nsample_}, &idx_tensor3));

            Tensor *idx_tensor4 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,n,nsample_}, &idx_tensor4));
            Tensor *idx_tensor5 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{b,n,nsample_}, &idx_tensor5));
            Tensor *idx_tensor6 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape{b,n,nsample_}, &idx_tensor6));
            Tensor *idx_tensor7 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(6, TensorShape{b,n,nsample_}, &idx_tensor7));
            Tensor *idx_tensor8 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(7, TensorShape{b,n,nsample_}, &idx_tensor8));
            Tensor *idx_tensor9 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(8, TensorShape{b,n,nsample_}, &idx_tensor9));

            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(9, TensorShape{b,n}, &pts_cnt_tensor));

            auto tangent_flat = tangent_tensor.flat<float>();
            const float *tangent = &(tangent_flat(0));
            auto group_flat = group_tensor.flat<int>();
            const int* group = &(group_flat(0));
            auto idx_flat1 = idx_tensor1->flat<int>();
            int *idx1 = &(idx_flat1(0));
            auto idx_flat2 = idx_tensor2->flat<int>();
            int *idx2 = &(idx_flat2(0));
            auto idx_flat3 = idx_tensor3->flat<int>();
            int *idx3 = &(idx_flat3(0));

            auto idx_flat4 = idx_tensor4->flat<int>();
            int *idx4 = &(idx_flat4(0));
            auto idx_flat5 = idx_tensor5->flat<int>();
            int *idx5 = &(idx_flat5(0));
            auto idx_flat6 = idx_tensor6->flat<int>();
            int *idx6 = &(idx_flat6(0));
            auto idx_flat7 = idx_tensor7->flat<int>();
            int *idx7 = &(idx_flat7(0));
            auto idx_flat8 = idx_tensor8->flat<int>();
            int *idx8 = &(idx_flat8(0));
            auto idx_flat9 = idx_tensor9->flat<int>();
            int *idx9 = &(idx_flat9(0));

            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryRadiusAnglePointLevelLauncher(b,n,m,0,radius_,nsample_,tangent,group,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx1,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx2,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx3,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryRadiusAnglePointLevel").Device(DEVICE_GPU), QueryRadiusAnglePointLevelGpuOp);


void queryTangent9PointLevelLauncher(int b, int n, int m, float start_angle, float radius, int nsample, const float *tangent, const int* group,
    int *idx1, int *idx2, int *idx3, int* idx4, int* idx5, int* idx6, int* idx7, int* idx8, int* idx9, int *pts_cnt);
class QueryTangent9PointLevelGpuOp : public OpKernel {
    public:
        explicit QueryTangent9PointLevelGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryTangentPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& tangent_tensor = context->input(0);
            OP_REQUIRES(context, tangent_tensor.dims()==4 && tangent_tensor.shape().dim_size(3)==2, errors::InvalidArgument("QueryTangentPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = tangent_tensor.shape().dim_size(0);
            int n = tangent_tensor.shape().dim_size(1);
            int m = tangent_tensor.shape().dim_size(2);
            const Tensor& group_tensor = context->input(1);
            Tensor *idx_tensor1 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,nsample_}, &idx_tensor1));
            Tensor *idx_tensor2 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,n,nsample_}, &idx_tensor2));
            Tensor *idx_tensor3 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{b,n,nsample_}, &idx_tensor3));

            Tensor *idx_tensor4 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{b,n,nsample_}, &idx_tensor4));
            Tensor *idx_tensor5 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{b,n,nsample_}, &idx_tensor5));
            Tensor *idx_tensor6 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape{b,n,nsample_}, &idx_tensor6));
            Tensor *idx_tensor7 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(6, TensorShape{b,n,nsample_}, &idx_tensor7));
            Tensor *idx_tensor8 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(7, TensorShape{b,n,nsample_}, &idx_tensor8));
            Tensor *idx_tensor9 = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(8, TensorShape{b,n,nsample_}, &idx_tensor9));

            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(9, TensorShape{b,n}, &pts_cnt_tensor));

            auto tangent_flat = tangent_tensor.flat<float>();
            const float *tangent = &(tangent_flat(0));
            auto group_flat = group_tensor.flat<int>();
            const int* group = &(group_flat(0));
            auto idx_flat1 = idx_tensor1->flat<int>();
            int *idx1 = &(idx_flat1(0));
            auto idx_flat2 = idx_tensor2->flat<int>();
            int *idx2 = &(idx_flat2(0));
            auto idx_flat3 = idx_tensor3->flat<int>();
            int *idx3 = &(idx_flat3(0));

            auto idx_flat4 = idx_tensor4->flat<int>();
            int *idx4 = &(idx_flat4(0));
            auto idx_flat5 = idx_tensor5->flat<int>();
            int *idx5 = &(idx_flat5(0));
            auto idx_flat6 = idx_tensor6->flat<int>();
            int *idx6 = &(idx_flat6(0));
            auto idx_flat7 = idx_tensor7->flat<int>();
            int *idx7 = &(idx_flat7(0));
            auto idx_flat8 = idx_tensor8->flat<int>();
            int *idx8 = &(idx_flat8(0));
            auto idx_flat9 = idx_tensor9->flat<int>();
            int *idx9 = &(idx_flat9(0));

            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryTangent9PointLevelLauncher(b,n,m,0,radius_,nsample_,tangent,group,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,idx9,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx1,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx2,pts_cnt);
            //queryTangentPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx3,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryTangent9PointLevel").Device(DEVICE_GPU), QueryRadiusAnglePointLevelGpuOp);

void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out);
class SelectionSortGpuOp : public OpKernel {
    public:
        explicit SelectionSortGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("SelectionSort expects positive k"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& dist_tensor = context->input(0);
            OP_REQUIRES(context, dist_tensor.dims()==3, errors::InvalidArgument("SelectionSort expects (b,m,n) dist shape."));
            int b = dist_tensor.shape().dim_size(0);
            int m = dist_tensor.shape().dim_size(1);
            int n = dist_tensor.shape().dim_size(2);

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,n}, &outi_tensor));
            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,n}, &out_tensor));

            auto dist_flat = dist_tensor.flat<float>();
            const float *dist = &(dist_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            selectionSortLauncher(b,n,m,k_,dist,outi,out);
        }
    private:
        int k_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionSort").Device(DEVICE_GPU), SelectionSortGpuOp);


void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out, int relative);
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("relative", &relative));
        }

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            groupPointLauncher(b,n,c,m,nsample,points,idx,out,relative);
        }
        int relative;
};
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU),GroupPointGpuOp);

void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points, int relative);
class GroupPointGradGpuOp: public OpKernel{
    public:
        explicit GroupPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("relative", &relative));
        }

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPointGrad expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            const Tensor& grad_out_tensor=context->input(2);
            OP_REQUIRES(context,grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==m && grad_out_tensor.shape().dim_size(2)==nsample && grad_out_tensor.shape().dim_size(3)==c, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample, channel) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,c}, &grad_points_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            cudaMemset(grad_points, 0, sizeof(float)*b*n*c);
            groupPointGradLauncher(b,n,c,m,nsample,grad_out,idx,grad_points,relative);
        }
        int relative;
};
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_GPU),GroupPointGradGpuOp);


