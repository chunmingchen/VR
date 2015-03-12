#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

//////////////////////////////////////////////////////////////////////////

#define EPS 1e-6f

#define OUT

#ifdef _DEBUG
#define ASSERT(b) {if (!(b)) {d_output[out_idx] = make_float4(1);}}

#else
#define ASSERT(b)
#endif


#endif

