//  Copyright © 2023 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UpSample.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/upsample_bicubic2d_backward_native.h>
#include <ATen/ops/upsample_bicubic2d_native.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#include <ATen/ops/upsample_linear1d.h>
#include <ATen/ops/upsample_linear1d_backward.h>
#include <ATen/ops/upsample_linear1d_backward_native.h>
#include <ATen/ops/upsample_linear1d_native.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest1d_backward.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest2d_backward.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#include <ATen/ops/upsample_nearest3d.h>
#include <ATen/ops/upsample_nearest3d_native.h>
#include <ATen/ops/upsample_nearest3d_backward.h>
#include <ATen/ops/upsample_nearest3d_backward_native.h>
#endif
namespace at::native {
namespace mps {

static char const* UPSAMPLE_NEAREST3D_KERNEL = R"UPSAMPLE3D_KERN(
#include <metal_stdlib>
using namespace metal;

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
int nn_compute_source_index(const float scale, int dst_index, int input_size) {
    const int src_index = min(static_cast<int>(floor((dst_index) * scale)), input_size - 1);
    return src_index;
}

//NCDHW
struct Upsample3DParams {
    size_t dstNumEl;
    size_t batchSz;
    size_t numChannels;
    size_t srcDepth;
    size_t srcHeight;
    size_t srcWidth;
    size_t dstDepth;
    size_t dstHeight;
    size_t dstWidth;
    float depthScale;
    float heightScale;
    float widthScale;
};

inline void _upsample_nearest3d_kernel_impl(device float const* src [[buffer(0)]],
                                      device float* dst [[buffer(1)]],
                                      constant Upsample3DParams& upsample3DParams,
                                      size_t threadPosInGrid) {

    size_t threadIdx = threadPosInGrid;
    size_t dstIdx = threadPosInGrid;

    size_t dst_h_stride = upsample3DParams.dstWidth;
    size_t dst_d_stride = upsample3DParams.dstHeight * upsample3DParams.dstWidth;
    size_t dst_c_stride = upsample3DParams.dstDepth * upsample3DParams.dstHeight * upsample3DParams.dstWidth;
    size_t dst_n_stride = upsample3DParams.numChannels * upsample3DParams.dstDepth * upsample3DParams.dstHeight * upsample3DParams.dstWidth;

    size_t src_h_stride = upsample3DParams.srcWidth;
    size_t src_d_stride = upsample3DParams.srcHeight * upsample3DParams.srcWidth;
    size_t src_c_stride = upsample3DParams.srcDepth * upsample3DParams.srcHeight * upsample3DParams.srcWidth;
    size_t src_n_stride = upsample3DParams.numChannels * upsample3DParams.srcDepth * upsample3DParams.srcHeight * upsample3DParams.srcWidth;

    size_t batchSz = threadIdx / dst_n_stride;
    threadIdx = threadIdx % dst_n_stride;
    size_t chSz = threadIdx / dst_c_stride;
    threadIdx = threadIdx % dst_c_stride;
    size_t dstD = threadIdx / dst_d_stride;
    threadIdx = threadIdx % dst_d_stride;
    size_t dstH = threadIdx / dst_h_stride;
    threadIdx = threadIdx % dst_h_stride;
    size_t dstW = threadIdx;

    auto srcD = nn_compute_source_index(upsample3DParams.depthScale, dstD, upsample3DParams.srcDepth);
    auto srcH = nn_compute_source_index(upsample3DParams.heightScale, dstH, upsample3DParams.srcHeight);
    auto srcW = nn_compute_source_index(upsample3DParams.widthScale, dstW, upsample3DParams.srcWidth);
    auto srcIdx = (batchSz * src_n_stride) + (chSz * src_c_stride) + (srcD * src_d_stride) + (srcH * src_h_stride) + srcW;

    dst[dstIdx] = src[srcIdx];

}

//_2d_dispatch_threadgroups
kernel void upsample_nearest3d_kernel(device float const* src [[buffer(0)]],
                                      device float* dst [[buffer(1)]],
                                      constant Upsample3DParams& upsample3DParams [[buffer(2)]],
                                      uint3 thread_position_in_grid [[thread_position_in_grid]],
                                      uint3 threadgroup_position_in_grid   [[ threadgroup_position_in_grid ]],
                                      uint3 thread_position_in_threadgroup [[ thread_position_in_threadgroup ]],
                                      uint3 threads_per_threadgroup        [[ threads_per_threadgroup ]]) {

    auto threadId = thread_position_in_grid.x +
                        thread_position_in_grid.y * threads_per_threadgroup.y +
                        thread_position_in_grid.z * threads_per_threadgroup.y * threads_per_threadgroup.x;
    if(threadId > upsample3DParams.dstNumEl) {
        return;
    }

    _upsample_nearest3d_kernel_impl(src, dst, upsample3DParams, threadId);
}

// --------- upsample3d backward kernel -----------
// see NOTE [ Nearest neighbor upsampling kernel implementation ]
inline static int nn_bw_compute_source_index(const float scale, int dst_index, int output_size) {
  // Equivalent to buggy OpenCV INTER_NEAREST
  // We keep this method for BC and consider as deprecated.
  // See nearest_neighbor_exact_bw_compute_source_index as replacement
  const int src_index = min(static_cast<int>(ceil(dst_index * scale)), output_size);
  return src_index;
}

//dst <-- Grad being calculated
//src <-- 3DUpsample-ed Tensor
struct Upsample3DBackwardParams {
    size_t dstNumEl;
    size_t batchSz;
    size_t numChans;

    size_t dstDepth; //dst => Grad
    size_t dstHeight;
    size_t dstWidth;
    size_t dstBatchStride;
    size_t dstChannelStride;
    size_t dstDepthStride;
    size_t dstHeightStride;

    size_t srcDepth; //src => 3DUpsampled-Ouput
    size_t srcHeight;
    size_t srcWidth;
    size_t srcBatchStride;
    size_t srcChannelStride;
    size_t srcDepthStride;
    size_t srcHeightStride;

    float upToDownDepthScale;
    float upToDownHeightScale;
    float upToDownWidthScale;
};
inline void _upsample_nearest3d_backward_kernel_impl(device float* dst[[buffer(0)]],
                                      constant Upsample3DBackwardParams& upsample3DParams [[buffer(2)]],
                                      size_t threadPosInGrid) {
    //dst <-- Grad being calculated
    //src <-- 3DUpsample-ed Tensor

    size_t threadIdx = threadPosInGrid;
    size_t dstIdx = threadPosInGrid;

    size_t src_h_stride = upsample3DParams.srcHeightStride;
    size_t src_d_stride = upsample3DParams.srcDepthStride;
    size_t src_c_stride = upsample3DParams.srcChannelStride;
    size_t src_n_stride = upsample3DParams.srcBatchStride;

    size_t batchSz = threadIdx / upsample3DParams.dstBatchStride;
    threadIdx = threadIdx % upsample3DParams.dstBatchStride;
    size_t chSz = threadIdx / upsample3DParams.dstChannelStride;
    threadIdx = threadIdx % upsample3DParams.dstChannelStride;
    size_t dstD = threadIdx / upsample3DParams.dstDepthStride;
    threadIdx = threadIdx % upsample3DParams.dstDepthStride;
    size_t dstH = threadIdx / upsample3DParams.dstHeightStride;
    threadIdx = threadIdx % upsample3DParams.dstHeightStride;
    size_t dstW = threadIdx;

    auto srcD = nn_bw_compute_source_index(upsample3DParams.upToDownDepthScale, dstD, upsample3DParams.srcDepth);
    if(upsample3DParams.upToDownDepthScale < 1 && srcD >= upsample3DParams.srcDepth) {
        //special handling for possible downsampling on any of the dims
        dst[dstIdx] = 0;
        return;
    }
    auto srcDPlus1 = nn_bw_compute_source_index(upsample3DParams.upToDownDepthScale, dstD+1, upsample3DParams.srcDepth);

    auto srcH = nn_bw_compute_source_index(upsample3DParams.upToDownHeightScale, dstH, upsample3DParams.srcHeight);
    if(upsample3DParams.upToDownHeightScale < 1 && srcH >= upsample3DParams.srcHeight) {
        //special handling for possible downsampling on any of the dims
        dst[dstIdx] = 0;
        return;
    }
    auto srcHPlus1 = nn_bw_compute_source_index(upsample3DParams.upToDownHeightScale, dstH+1, upsample3DParams.srcHeight);

    auto srcW = nn_bw_compute_source_index(upsample3DParams.upToDownWidthScale, dstW, upsample3DParams.srcWidth);
    if(upsample3DParams.upToDownWidthScale < 1 && srcW >= upsample3DParams.srcWidth) {
        //special handling for possible downsampling on any of the dims
        dst[dstIdx] = 0;
        return;
    }
    auto srcWPlus1 = nn_bw_compute_source_index(upsample3DParams.upToDownWidthScale, dstW+1, upsample3DParams.srcWidth);

    dst[dstIdx] = max(srcDPlus1-srcD, 1) * max(srcHPlus1-srcH, 1) * max(srcWPlus1-srcW, 1);
}

kernel void upsample_nearest3d_backward_kernel(device float* dst[[buffer(0)]],
                                      constant Upsample3DBackwardParams& upsample3DParams [[buffer(1)]],
                                      uint3 thread_position_in_grid [[thread_position_in_grid]],
                                      uint3 threads_per_threadgroup [[ threads_per_threadgroup ]]) {

    auto threadId = thread_position_in_grid.x +
                        thread_position_in_grid.y * threads_per_threadgroup.y +
                        thread_position_in_grid.z * threads_per_threadgroup.y * threads_per_threadgroup.x;

    if(threadId > upsample3DParams.dstNumEl) {
        return;
    }

    _upsample_nearest3d_backward_kernel_impl(dst, upsample3DParams, threadId);
}
)UPSAMPLE3D_KERN";

static id<MTLLibrary> compileUpsample3dKernelLibrary(id<MTLDevice> device) {
	static id<MTLLibrary> upsampleMtlLibrary = nil;
	if (upsampleMtlLibrary) {
		return upsampleMtlLibrary;
	}

	NSError* error = nil;
	MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
	[options setLanguageVersion:MTLLanguageVersion2_3];
	upsampleMtlLibrary = [device newLibraryWithSource:[NSString stringWithCString:UPSAMPLE_NEAREST3D_KERNEL
		encoding:NSASCIIStringEncoding]
		options:options
		error:&error];
	TORCH_CHECK(
			upsampleMtlLibrary, "Failed to create metal UPSAMPLE_3D_NN_KERNEL, error: ", [[error description] UTF8String]);
	return upsampleMtlLibrary;
}

static id<MTLComputePipelineState> upsample3DNearestNeighborPSO(id<MTLDevice> device) {
	std::string kernel = "upsample_nearest3d_kernel";
	static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
	id<MTLComputePipelineState> pso = psoCache[kernel];
	if (pso) {
		return pso;
	}

	NSError* error = nil;
	id<MTLLibrary> upsample3dLib = compileUpsample3dKernelLibrary(device);
	id<MTLFunction> upsample3dNNFunc = [upsample3dLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
	TORCH_CHECK(upsample3dNNFunc, "Failed to create function state object for: ", kernel);
	pso = [device newComputePipelineStateWithFunction:upsample3dNNFunc error:&error];
	TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

	psoCache[kernel] = pso;
	return pso;
}

static id<MTLComputePipelineState> upsample3DNearestNeighborBackwardPSO(id<MTLDevice> device) {
	std::string kernel = "upsample_nearest3d_backward_kernel";
	static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
	id<MTLComputePipelineState> pso = psoCache[kernel];
	if (pso) {
		return pso;
	}

	NSError* error = nil;
	id<MTLLibrary> upsample3dLib = compileUpsample3dKernelLibrary(device);
	id<MTLFunction> upsample3dNNFunc = [upsample3dLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
	TORCH_CHECK(upsample3dNNFunc, "Failed to create function state object for: ", kernel);
	pso = [device newComputePipelineStateWithFunction:upsample3dNNFunc error:&error];
	TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

	psoCache[kernel] = pso;
	return pso;
}

// Upsampling operations (1D/2D forward and backward)
// supported resize_mode: 'nearest' | 'bilinear' | 'nearest-exact'
static void upsample_out_template(const Tensor& input,
                                  IntArrayRef output_size,
                                  std::optional<IntArrayRef> input_size_opt, // only used for backward pass
                                  std::optional<double> scale_h_opt,
                                  std::optional<double> scale_w_opt,
                                  const Tensor& output,
                                  bool align_corners,
                                  const std::string_view resize_mode_str) {
  if (input.numel() == 0) {
    return;
  }
  const auto input_dim = input.sizes();
  if (input_dim.size() <= 3) {
    native::upsample_1d_common_check(input.sizes(), output_size);
  } else {
    native::upsample_2d_common_check(input.sizes(), output_size);
  }
  Tensor out;
  if (needsGather(output)) {
    out = at::empty_like(output, MemoryFormat::Contiguous);
  }

  bool centerResults = false;
  MPSGraphResizeMode resizeMode = MPSGraphResizeNearest;
  MPSGraphResizeNearestRoundingMode nearestRoundingMode = MPSGraphResizeNearestRoundingModeFloor;
  MPSGraphTensorNamedDataLayout dataLayout =
      input_dim.size() > 3 ? MPSGraphTensorNamedDataLayoutNCHW : MPSGraphTensorNamedDataLayoutCHW;
  if (resize_mode_str == "nearest") {
    resizeMode = MPSGraphResizeNearest;
  } else if (resize_mode_str == "bilinear") {
    resizeMode = MPSGraphResizeBilinear;
    centerResults = true;
  } else if (resize_mode_str == "nearest-exact") {
    centerResults = true;
    nearestRoundingMode = MPSGraphResizeNearestRoundingModeRoundPreferCeil;
  } else {
    TORCH_CHECK(false, "Unsupported resize mode ", resize_mode_str);
  }

  const int64_t output_width = output_size.size() > 1 ? output_size[1] : output_size[0];
  const int64_t output_height = output_size.size() > 1 ? output_size[0] : (output.dim() > 2 ? output.size(-2) : 1);
  const float scale_w = (scale_w_opt.value_or(0.) > 0.) ? static_cast<float>(scale_w_opt.value()) : 0.;
  const float scale_h = (scale_h_opt.value_or(0.) > 0.) ? static_cast<float>(scale_h_opt.value()) : 1.;
  const float offset_y = centerResults ? (scale_h - 1.0f) / 2.0f : 0.0f;
  const float offset_x = centerResults ? (scale_w - 1.0f) / 2.0f : 0.0f;

  IntArrayRef input_size;
  const bool is_backward_pass = input_size_opt.has_value();
  if (is_backward_pass) {
    input_size = input_size_opt.value();
  }
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor = nil, *outputTensor = nil;
    MPSGraphTensor* outputSizeTensor = nil;
  };
  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "upsample_" + std::string(resize_mode_str) + (align_corners ? "_aligned_corners" : "") +
        getTensorsStringKey({input}) + ":[" + std::to_string(scale_h) + "," + std::to_string(scale_w) + "]:[" +
        (is_backward_pass ? getArrayRefString(input_size) : "Undefined") + "]";

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      newCachedGraph->outputSizeTensor = mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeInt32, @[ @(2) ]);

      MPSGraphTensor* scaleOffsetTensor = nullptr;
      MPSGraphTensor* inputSizeTensor = nullptr;

      if (scale_w > 0.0) {
        const float outScales[4] = {scale_h, scale_w, offset_y, offset_x};
        scaleOffsetTensor = [mpsGraph constantWithData:[NSData dataWithBytes:outScales length:sizeof(outScales)]
                                                 shape:@[ @4 ]
                                              dataType:MPSDataTypeFloat32];
      }
      if (is_backward_pass) {
        std::vector<NSNumber*> inputSizeVec(4);
        inputSizeVec[0] = @(input_size[0]);
        inputSizeVec[1] = @(input_size[1]);
        inputSizeVec[2] = @(input_size[2]);
        inputSizeVec[3] = @(input_dim.size() > 3 ? input_size[3] : 1);
        inputSizeTensor = [mpsGraph constantWithScalar:0
                                                 shape:[NSArray arrayWithObjects:inputSizeVec.data()
                                                                           count:input_dim.size()]
                                              dataType:getMPSDataType(input)];
      }
      if (!is_backward_pass) {
        if (scaleOffsetTensor && !align_corners) {
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithTensor:newCachedGraph->inputTensor
                                                                  sizeTensor:newCachedGraph->outputSizeTensor
                                                           scaleOffsetTensor:scaleOffsetTensor
                                                         nearestRoundingMode:nearestRoundingMode
                                                                      layout:dataLayout
                                                                        name:nil];
          } else { // bilinear forward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithTensor:newCachedGraph->inputTensor
                                                                   sizeTensor:newCachedGraph->outputSizeTensor
                                                            scaleOffsetTensor:scaleOffsetTensor
                                                                       layout:dataLayout
                                                                         name:nil];
          }
        } else { // scaleOffsetTensor == nil || align_corners
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithTensor:newCachedGraph->inputTensor
                                                                  sizeTensor:newCachedGraph->outputSizeTensor
                                                         nearestRoundingMode:nearestRoundingMode
                                                                centerResult:centerResults
                                                                alignCorners:align_corners
                                                                      layout:dataLayout
                                                                        name:nil];
          } else { // bilinear forward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithTensor:newCachedGraph->inputTensor
                                                                   sizeTensor:newCachedGraph->outputSizeTensor
                                                                 centerResult:centerResults
                                                                 alignCorners:align_corners
                                                                       layout:dataLayout
                                                                         name:nil];
          }
        }
      } else { // is_backward_pass == true
        if (scaleOffsetTensor && !align_corners) {
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithGradientTensor:newCachedGraph->inputTensor
                                                                               input:inputSizeTensor
                                                                   scaleOffsetTensor:scaleOffsetTensor
                                                                 nearestRoundingMode:nearestRoundingMode
                                                                              layout:dataLayout
                                                                                name:nil];
          } else { // bilinear backward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithGradientTensor:newCachedGraph->inputTensor
                                                                                input:inputSizeTensor
                                                                    scaleOffsetTensor:scaleOffsetTensor
                                                                               layout:dataLayout
                                                                                 name:nil];
          }
        } else { // scaleOffsetTensor == nil || align_corners
          if (resizeMode == MPSGraphResizeNearest) {
            newCachedGraph->outputTensor = [mpsGraph resizeNearestWithGradientTensor:newCachedGraph->inputTensor
                                                                               input:inputSizeTensor
                                                                 nearestRoundingMode:nearestRoundingMode
                                                                        centerResult:centerResults
                                                                        alignCorners:align_corners
                                                                              layout:dataLayout
                                                                                name:nil];
          } else { // bilinear backward
            newCachedGraph->outputTensor = [mpsGraph resizeBilinearWithGradientTensor:newCachedGraph->inputTensor
                                                                                input:inputSizeTensor
                                                                         centerResult:centerResults
                                                                         alignCorners:align_corners
                                                                               layout:dataLayout
                                                                                 name:nil];
          }
        }
      }
    });
    MPSNDArrayDescriptor* sizeDesc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32 shape:@[ @(2) ]];
    MPSNDArray* sizeNDArray = [[[MPSNDArray alloc] initWithDevice:stream->device() descriptor:sizeDesc] autorelease];
    [sizeNDArray writeBytes:(int32_t[]){(int32_t)output_height, (int32_t)output_width} strideBytes:nil];
    MPSGraphTensorData* sizeTensorData = [[[MPSGraphTensorData alloc] initWithMPSNDArray:sizeNDArray] autorelease];

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor, input);
    Placeholder outputPlaceholder =
        Placeholder(cachedGraph->outputTensor, out.has_storage() ? out : output, nil, false);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
      cachedGraph->outputSizeTensor : sizeTensorData,
    };
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);

    if (out.has_storage()) {
      output.copy_(out);
    }
  }
}

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UpSample_metallib.h>
#endif

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename accscalar_t>
static accscalar_t compute_scales_value_backwards(const std::optional<double> scale,
                                                  int64_t src_size,
                                                  int64_t dst_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
  return (scale.value_or(0.) > 0.) ? (accscalar_t)scale.value() : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
static accscalar_t area_pixel_compute_scale(int input_size,
                                            int output_size,
                                            bool align_corners,
                                            const std::optional<double> scale) {
  if (align_corners) {
    if (output_size > 1) {
      return (accscalar_t)(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<accscalar_t>(0);
    }
  } else {
    return compute_scales_value<accscalar_t>(scale, input_size, output_size);
  }
}

static void upsample_kernel_out_template(const Tensor& input,
                                         IntArrayRef output_size,
                                         bool align_corners,
                                         std::optional<double> scale_h_opt,
                                         std::optional<double> scale_w_opt,
                                         const Tensor& output,
                                         const std::string name) {
  if (output.numel() == 0) {
    return;
  }
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(input.size(3), output.size(3), align_corners, scale_w_opt),
      area_pixel_compute_scale<float>(input.size(2), output.size(2), align_corners, scale_h_opt)};
  auto upsamplePSO = lib.getPipelineStateForFunc(fmt::format("upsample_{}_{}", name, scalarToMetalTypeString(input)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      std::array<int64_t, 4> output_strides = {output.stride(3), output.stride(2), output.stride(1), output.stride(0)};
      std::array<int64_t, 4> output_sizes = {output.size(3), output.size(2), output.size(1), output.size(0)};
      std::array<int64_t, 4> input_sizes = {input.size(3), input.size(2), input.size(1), input.size(0)};
      std::array<int64_t, 4> input_strides = {input.stride(3), input.stride(2), input.stride(1), input.stride(0)};
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder,
                  input,
                  output,
                  input_strides,
                  output_strides,
                  input_sizes,
                  output_sizes,
                  scales,
                  align_corners);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1]);
    }
  });
}

static void upsample_kernel_backward_out_template(const Tensor& grad_input,
                                                  const Tensor& grad_output,
                                                  IntArrayRef output_size,
                                                  IntArrayRef input_size,
                                                  bool align_corners,
                                                  std::optional<double> scale_h_opt,
                                                  std::optional<double> scale_w_opt,
                                                  const std::string& name) {
  grad_input.zero_();
  if (grad_output.numel() == 0) {
    return;
  }
  std::array<float, 2> scales = {
      area_pixel_compute_scale<float>(grad_input.size(3), grad_output.size(3), align_corners, scale_w_opt),
      area_pixel_compute_scale<float>(grad_input.size(2), grad_output.size(2), align_corners, scale_h_opt)};
  auto upsamplePSO = lib.getPipelineStateForFunc(
      fmt::format("upsample_{}_backward_{}", name, mps::scalarToMetalTypeString(grad_input)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      std::array<int64_t, 4> output_strides = {
          grad_output.stride(3), grad_output.stride(2), grad_output.stride(1), grad_output.stride(0)};
      std::array<int64_t, 4> output_sizes = {
          grad_output.size(3), grad_output.size(2), grad_output.size(1), grad_output.size(0)};
      std::array<int64_t, 4> input_sizes = {
          grad_input.size(3), grad_input.size(2), grad_input.size(1), grad_input.size(0)};
      std::array<int64_t, 4> input_strides = {
          grad_input.stride(3), grad_input.stride(2), grad_input.stride(1), grad_input.stride(0)};
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:upsamplePSO];
      mtl_setArgs(computeEncoder,
                  grad_input,
                  grad_output,
                  input_strides,
                  output_strides,
                  input_sizes,
                  output_sizes,
                  scales,
                  align_corners);
      mtl_dispatch1DJob(computeEncoder, upsamplePSO, output_size[0] * output_size[1]);
    }
  });
}

} // namespace mps

TORCH_IMPL_FUNC(upsample_nearest1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, std::nullopt, scale, output, false, "nearest");
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_out_template(grad_output, output_size, input_size, std::nullopt, scale, grad_input, false, "nearest");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_mps)
(const Tensor& input, IntArrayRef output_size, std::optional<double> scale, const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, std::nullopt, scale, output, false, "nearest-exact");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, std::nullopt, scale, grad_input, false, "nearest-exact");
}

TORCH_IMPL_FUNC(upsample_nearest2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, scales_h, scales_w, output, false, "nearest");
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_out_template(grad_output, output_size, input_size, scales_h, scales_w, grad_input, false, "nearest");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, scales_h, scales_w, output, false, "nearest-exact");
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, scales_h, scales_w, grad_input, false, "nearest-exact");
}

TORCH_IMPL_FUNC(upsample_linear1d_out_mps)
(const Tensor& input, IntArrayRef output_size, bool align_corners, std::optional<double> scale, const Tensor& output) {
  mps::upsample_out_template(input, output_size, std::nullopt, std::nullopt, scale, output, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scale,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, std::nullopt, scale, grad_input, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, align_corners, scales_h, scales_w, output, "bilinear2d");
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_out_template(
      grad_output, output_size, input_size, scales_h, scales_w, grad_input, align_corners, "bilinear");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  mps::upsample_kernel_out_template(input, output_size, align_corners, scales_h, scales_w, output, "bicubic2d");
}

TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  mps::upsample_kernel_backward_out_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w, "bicubic2d");
}

static void upsample_nearest3d_out_mps_impl(Tensor const& input,
                                            IntArrayRef output_size,
                                            c10::optional<double> scales_d,
                                            c10::optional<double> scales_h,
                                            c10::optional<double> scales_w,
                                            const Tensor& output) {
  if (input.numel() == 0) {
    return;
  }

  struct Upsample3DParams {
    int64_t outNumEl;
    int64_t batchSz;
    int64_t numChannels;
    int64_t inpDepth;
    int64_t inpHeight;
    int64_t inpWidth;
    int64_t outDepth;
    int64_t outHeight;
    int64_t outWidth;
    float depthScale;
    float heightScale;
    float widthScale;
  };

  const auto inpDepth = input.size(2);
  const auto inpHeight = input.size(3);
  const auto inpWidth = input.size(4);

  const auto outDepth = output.size(2);
  const auto outHeight = output.size(3);
  const auto outWidth = output.size(4);

  const float scaleDepth = compute_scales_value<float>(scales_d, inpDepth, outDepth);
  const float scaleHeight = compute_scales_value<float>(scales_h, inpHeight, outHeight);
  const float scaleWidth = compute_scales_value<float>(scales_w, inpWidth, outWidth);

  const Upsample3DParams ups3dParams = {.outNumEl = output.numel(),
                                        .batchSz = output.size(0),
                                        .numChannels = output.size(1),
                                        .inpDepth = inpDepth,
                                        .inpHeight = inpHeight,
                                        .inpWidth = inpWidth,
                                        .outHeight = outHeight,
                                        .outWidth = outWidth,
                                        .outDepth = outDepth,
                                        .depthScale = scaleDepth,
                                        .heightScale = scaleHeight,
                                        .widthScale = scaleWidth};

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      NSError* error = nil;

      id<MTLComputePipelineState> kernelPSO = mps::upsample3DNearestNeighborPSO(device);
      TORCH_CHECK(kernelPSO, error.localizedDescription.UTF8String);

      // TODO
      // getMPSProfiler().beginProfileKernel(kernelPSO, kernel, {input, other});
      //  Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState:kernelPSO];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(input) offset:0 atIndex:0];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(output) offset:0 atIndex:1];
      [computeEncoder setBytes:&ups3dParams length:sizeof(ups3dParams) atIndex:2];

      // Grid Dispatch configuration
      // Configure 3D ThreadGroup
      NSUInteger w = kernelPSO.threadExecutionWidth;
      NSUInteger h = kernelPSO.maxTotalThreadsPerThreadgroup / w;
      NSUInteger d = 1;
      MTLSize bdim = MTLSizeMake(w, h, d);
      //NSLog(@"threadExecutionWidth=%lu, maxTotalThreadsPerThreadgroup=%lu, threadsPerThreadGroup(w=%lu, h=%lu, d=%lu)",
            //kernelPSO.threadExecutionWidth,
            //kernelPSO.maxTotalThreadsPerThreadgroup,
            //bdim.width, bdim.height, bdim.depth);

      // Configure threads Grid
      auto outNumEl = output.numel();
      auto gW = (outWidth + w - 1) / w;
      auto gH = (outHeight + h - 1) / h;
      auto gD = (outNumEl + (gH * gW) - 1) / (gH * gW);
      MTLSize tgpg = MTLSizeMake(gW, gH, gD);
      //NSLog(@"GridSz(w=%lu, h=%lu, d=%lu)", tgpg.width, tgpg.height, tgpg.depth);
      [computeEncoder dispatchThreadgroups:tgpg threadsPerThreadgroup:bdim];

      // getMPSProfiler().endProfileKernel(kernelPSO); //TODO
    }
  });
}

static void upsample_nearest3d_backward_out_mps_impl(Tensor const& input,
                                                     c10::optional<double> scales_d,
                                                     c10::optional<double> scales_h,
                                                     c10::optional<double> scales_w,
                                                     const Tensor& output) {
  if (input.numel() == 0) {
    return;
  }

  // out <-- Grad being calculated (output)
  // inp <-- 3DUpsample-ed Tensor (inpput)
  struct Upsample3DBackwardsParams {
    int64_t outNumEl;
    int64_t batchSz;
    int64_t numChans;

    int64_t outDepth; // out => Grad
    int64_t outHeight;
    int64_t outWidth;
    int64_t outBatchStride;
    int64_t outChannelStride;
    int64_t outDepthStride;
    int64_t outHeightStride;

    int64_t inpDepth; // inp => 3DUpsampled-Ouput
    int64_t inpHeight;
    int64_t inpWidth;
    int64_t inpBatchStride;
    int64_t inpChannelStride;
    int64_t inpDepthStride;
    int64_t inpHeightStride;

    float inpToOutDepthScale;
    float inpToOutHeightScale;
    float inpToOutWidthScale;
  };

  Upsample3DBackwardsParams backwardsParams = {
      .outNumEl = output.numel(), // Grad will have same numel as src

      .batchSz = output.size(0),
      .numChans = output.size(1),

      .outDepth = output.size(2), // out => Grad
      .outHeight = output.size(3),
      .outWidth = output.size(4),
      .outBatchStride = (output.size(1) * output.size(2) * output.size(3) * output.size(4)),
      .outChannelStride = (output.size(2) * output.size(3) * output.size(4)),
      .outDepthStride = (output.size(3) * output.size(4)),
      .outHeightStride = output.size(4),

      .inpDepth = input.size(2),
      .inpHeight = input.size(3),
      .inpWidth = input.size(4),
      .inpBatchStride = (input.size(1) * input.size(2) * input.size(3) * input.size(4)),
      .inpChannelStride = (input.size(2) * input.size(3) * input.size(4)),
      .inpDepthStride = (input.size(3) * input.size(4)),
      .inpHeightStride = input.size(4),

      .inpToOutDepthScale = compute_scales_value<float>(scales_d, input.size(2), output.size(2)),
      .inpToOutHeightScale = compute_scales_value<float>(scales_h, input.size(3), output.size(3)),
      .inpToOutWidthScale = compute_scales_value<float>(scales_w, input.size(4), output.size(4)),
  };

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      NSError* error = nil;

      id<MTLComputePipelineState> kernelPSO = mps::upsample3DNearestNeighborBackwardPSO(device);
      TORCH_CHECK(kernelPSO, error.localizedDescription.UTF8String);

      // TODO
      // getMPSProfiler().beginProfileKernel(kernelPSO, kernel, {input, other});
      //  Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState:kernelPSO];
      [computeEncoder setBuffer:mps::getMTLBufferStorage(output) offset:0 atIndex:0];
      [computeEncoder setBytes:&backwardsParams length:sizeof(backwardsParams) atIndex:1];

      // Grid Dispatch configuration
      // Configure 3D ThreadGroup
      NSUInteger w = kernelPSO.threadExecutionWidth;
      NSUInteger h = kernelPSO.maxTotalThreadsPerThreadgroup / w;
      NSUInteger d = 1;
      MTLSize bdim = MTLSizeMake(w, h, d);
      //NSLog(@"threadExecutionWidth=%lu, maxTotalThreadsPerThreadgroup=%lu, threadsPerThreadGroup(w=%lu, h=%lu, d=%lu)",
            //kernelPSO.threadExecutionWidth,
            //kernelPSO.maxTotalThreadsPerThreadgroup,
            //bdim.width,
            //bdim.height,
            //bdim.depth);

      // Configure threads Grid
      auto dstNumEl = output.numel();
      auto gW = (output.size(4) + w - 1) / w;
      auto gH = (output.size(3) + h - 1) / h;
      auto gD = (output.size(2) + (gH * gW) - 1) / (gH * gW);
      MTLSize tgpg = MTLSizeMake(gW, gH, gD);
      //NSLog(@"GridSz(w=%lu, h=%lu, d=%lu)", tgpg.width, tgpg.height, tgpg.depth);
      [computeEncoder dispatchThreadgroups:tgpg threadsPerThreadgroup:bdim];

      // getMPSProfiler().endProfileKernel(kernelPSO); //TODO
    }
  });
}

TORCH_IMPL_FUNC(upsample_nearest3d_out_mps)
(const Tensor& input,
 IntArrayRef output_size,
 c10::optional<double> scales_d,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& output) {
    upsample_nearest3d_out_mps_impl(input, output_size, scales_d, scales_h, scales_w, output);
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_mps)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 c10::optional<double> scales_d,
 c10::optional<double> scales_h,
 c10::optional<double> scales_w,
 const Tensor& grad_input) {
    upsample_nearest3d_backward_out_mps_impl(grad_output, scales_d, scales_h, scales_w, grad_input);
}

} // namespace at::native
