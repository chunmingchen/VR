#ifndef CUDA_SHADER_H
#define CUDA_SHADER_H
#include "HostCommon.h"
#include "CudaCommon.h"
#include "cuda_render.h"
//#include "cutil_math.h"
//#include "my_cuda_math.h"

#define LINTERP_LOD_DATA //faster texture lookup, added on dec. 28, 2010
// drawback: 1. blocky segment  2. cannot linear-interpolate on associated colors (intensity different between levels)
#define NO_SHADING 0
#define PHONG_SHADING 1
//#define SHADING_TYPE NO_SHADING 
#define SHADING_TYPE PHONG_SHADING

#define SHADING_MIN_GRADIENT .1f	// threshold for calculating gradient

// cudaReadModeNormalizedFloat: read float (uchar/255.f)
// cudaReadModeElementType: read original data type
#ifdef LINTERP_LOD_DATA
texture<LOD_DATA_TYPE, 3, cudaReadModeNormalizedFloat> texBrickPool;
#else
texture<LOD_DATA_TYPE, 3, cudaReadModeElementType> texBrickPool;
#endif

// transfer function
texture<float4, 2, cudaReadModeElementType> texTrFn;
cudaArray *da_trFn = 0;
// <Phong shading
struct PhongParam {
	float ka;
	float kd;
	float ks;
	float n; // shininess
};
__constant__ PhongParam c_phong;
//>

//< Light
struct LightParam{
	float3 color;
	float3 pos; // in world space
};
__constant__ LightParam c_light;

//>

// tri-linear interpolation
inline __device__ float4 d_texLerp3(float3 &pos);
inline __device__ float3 d_get_normal(const float3 &pos);

///////////////////////////// kernel functions ////////////////////////////



#ifdef SEG_ALPHA_FORMAT
inline __device__
float4 d_trfnLookup(float seg, float alpha)
{
	// input: 0..1 float
	// trfn: 
	//  type: float4
	//  read mode: element
	//  normalized texcoord: no
	//  filter mode: linear
	return tex2D(texTrFn, (seg*255.f+.5f), alpha*255.f);
										//	return make_float4(alpha);
										//return tex2D(texTrFn, seg, alpha);
										//	return tex2D(texTrFn, (seg*255.f+.5f)/256.f, alpha);
}
#endif




inline __device__ float4 d_shader(const float3 &texPos)
{
#ifdef SEG_ALPHA_FORMAT
#ifdef LINTERP_LOD_DATA
	// nearest neighbor
	float seg = tex3D(texBrickPool, floor(texPos.x+.5f)+.5f, floor(texPos.y+.5f)+.5f, floor(texPos.z+.5f)+.5f).x;
	// interpolation
	float alpha = tex3D(texBrickPool, texPos.x, texPos.y, texPos.z).y;
	// trfn lookup
	//return tex2D(texTrFn, seg*255.f+.5f, alpha*255.f);
	return d_trfnLookup(seg, alpha);

#else
	return d_texLerp3(texPos);
#endif //LINTERP_LOD_DATA
#else
	return tex3D(texBrickPool, texPos.x, texPos.y, texPos.z);
#endif
}


inline __device__ float3 d_get_normal(const float3 &pos)
{
	float3 sample1, sample2;
	sample1.x = tex3D(texBrickPool, pos.x-.5f, pos.y, pos.z).y;  //y: alpha value
	sample2.x = tex3D(texBrickPool, pos.x+.5f, pos.y, pos.z).y;	// Cannot use offset 1.0 to previent referencing out of block blound
	sample1.y = tex3D(texBrickPool, pos.x, pos.y-.5f, pos.z).y;
	sample2.y = tex3D(texBrickPool, pos.x, pos.y+.5f, pos.z).y;
	sample1.z = tex3D(texBrickPool, pos.x, pos.y, pos.z-.5f).y;
	sample2.z = tex3D(texBrickPool, pos.x, pos.y, pos.z+.5f).y;
	return normalize(sample2-sample1);
}

// phone shading: supposed N, V, L are normalized
inline __device__ float4 d_phong_shading(const float4 material_color, const float3 N, const float3 V, const float3 L)
{
	if (material_color.w > 0) {
		// phong shading
		float diffuseLight = max(-dot(L,N), 0.f);

		float3 diffuse = make_float3(material_color) * (c_phong.ka +  c_phong.kd * diffuseLight);/*c_light.color */ 
		
		float3 H = normalize(L+V);
		//if (diffuseLight>0) {
			float specularLight = powf(max(-dot(H,N), 0.f), c_phong.n); // (why minus?)
			float specular = c_phong.ks *  specularLight;/*c_light.color */
			return make_float4(diffuse + make_float3(specular), material_color.w);
		//}else
		//	return diffuse ;
	}else {
		return material_color*c_phong.ka;
	}
}


// got color from const color data in nodes
inline __device__ float4 d_shader_constColor(const float4 &c)
{
#ifdef SEG_ALPHA_FORMAT
	return d_trfnLookup(c.x, c.y);
#else
	return c;
#endif
}



//////////////////////////////////////////////////////////////////////////
#ifndef LINTERP_LOD_DATA
inline __device__
float4 d_trfnLookup(uchar2 &t)
{
	return tex2D(texTrFn, ((float)t.x+.5f), (float)t.y);
}

inline __host__ __device__ float3 ceil(const float3 v)
{
	return make_float3(ceil(v.x), ceil(v.y), ceil(v.z));
}


inline __device__ 
float4 d_texLerp3(float3 &pos)
{
	float3 posf = floor(pos);
	float3 posc = ceil(pos);
	float3 d = pos-posf;
	float3 d1 = posc-pos;

	// better caching?
	uchar2 data[8];
	data[0] = tex3D(texBrickPool, posf.x, posf.y, posf.z);
	data[1] = tex3D(texBrickPool, posf.x, posf.y, posc.z);
	data[2] = tex3D(texBrickPool, posf.x, posc.y, posf.z);
	data[3] = tex3D(texBrickPool, posf.x, posc.y, posc.z);
	data[4] = tex3D(texBrickPool, posc.x, posf.y, posf.z);
	data[5] = tex3D(texBrickPool, posc.x, posf.y, posc.z);
	data[6] = tex3D(texBrickPool, posc.x, posc.y, posf.z);
	data[7] = tex3D(texBrickPool, posc.x, posc.y, posc.z);
	return (
				(	d_trfnLookup(data[0]) * d1.z
				  +	d_trfnLookup(data[1]) * d.z) * d1.y 
				+(	d_trfnLookup(data[2]) * d1.z
				  +	d_trfnLookup(data[3]) * d.z) *d.y 
			) * d1.x +
			(
				(	d_trfnLookup(data[4]) * d1.z
				  +	d_trfnLookup(data[5]) * d.z) * d1.y
				+(	d_trfnLookup(data[6]) * d1.z
				  +	d_trfnLookup(data[7]) * d.z) * d.y
			) * d.x;

}
#endif
//////////////////////////////////////////////////////////////////////////
#if 0 // associated color
inline __device__
void mix_color(OUT float4 &originalColor, float4 &addColor, float sampDist)
{
	if (addColor.w) {
		float w = (1.f - originalColor.w) * (1.f - powf(1.f - addColor.w, sampDist));
		originalColor.w += w;	
		w /= (addColor.w);
		originalColor.x += addColor.x * w;					
		originalColor.y += addColor.y * w;
		originalColor.z += addColor.z * w;
	}
}
#else	// non-associated color
#define mix_color(originalColor, addColor, sampDist)			\
{																\
	float w = (1.f - originalColor.w) * (1.0f - powf(1.0f - addColor.w , (float)sampDist));	\
	originalColor.x += addColor.x * w;					\
	originalColor.y += addColor.y * w;					\
	originalColor.z += addColor.z * w;					\
	originalColor.w += w;								\
}	
#endif

#if 0  // old: with intensity.  now replaced by using trfn 
// associated color
inline __device__
void mix_color(OUT float4 &originalColor, float4 &addColor, int sampDist, float intensity)
{
	if (addColor.w) {
		float w = intensity * (1.f - originalColor.w) * (1.f - powf(1.f - addColor.w, (float)sampDist));
		originalColor.w += w;	
		w /= (addColor.w);
		originalColor.x += addColor.x * w;					
		originalColor.y += addColor.y * w;
		originalColor.z += addColor.z * w;
	}
}
// non-associated color
#define mix_color(originalColor, addColor, sampDist)			\
{																\
	addColor.w = c_vrParam.intensity * (1.f - originalColor.w) * (1.0f - powf(1.0f - addColor.w , (float)sampDist));	\
	originalColor.x += addColor.x * addColor.w;					\
	originalColor.y += addColor.y * addColor.w;					\
	originalColor.z += addColor.z * addColor.w;					\
	originalColor.w += addColor.w;								\
}	
#endif
//////////////////////////////////////////////////////////////////////////


//////////////////////////// extern functions /////////////////////////////
extern "C"
void g_releaseTrFn()
{
	if (NULL!=da_trFn) {
		cudaThreadSynchronize();
		CUDA_SAFE_CALL(cudaFreeArray(da_trFn));
		da_trFn = NULL;
	}

}

extern "C"
void g_createTrFn(int w, int h)
{
	g_releaseTrFn();

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	CUDA_SAFE_CALL( cudaMallocArray(&da_trFn, &channelDesc, w, h) );

	// set texture parameters
	texTrFn.normalized = false;                      // access with normalized texture coordinates
	texTrFn.filterMode = cudaFilterModeLinear;      // linear interpolation
	texTrFn.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	texTrFn.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	CUDA_SAFE_CALL(cudaBindTextureToArray(texTrFn, da_trFn, channelDesc));
	cudaThreadSynchronize();
}

extern "C"
void g_uploadTrFn(void *data, int len)
{
	CUDA_SAFE_CALL( cudaMemcpyToArray(da_trFn, 0, 0, data, len*sizeof(float)*4, cudaMemcpyHostToDevice) );  
	//CUDA_SAFE_CALL(cudaBindTextureToArray(texTrFn, da_trFn));
}

//////////////////////////////////////////////////////////////////////////
extern "C"
void g_setPhongShading()
{
	PhongParam phong;
	phong.ka = .6f;
	phong.kd = .4f;
	phong.ks = .2f;
	phong.n = 20.f;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_phong, &phong, sizeof(PhongParam)) );
	
	LightParam light;
	light.color = make_float3(1.f, 1.f, 1.f);	// currently disabled
	light.pos = make_float3(0.09f, 0.19f , 1.f);  // upper right of eyes
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_light, &light, sizeof(LightParam)) );
		
}

//////////////////////////////////////////////////////////////////////////

#if 0
extern "C"
void vr_uploadVolume(uchar *h_volume, int w, int h, int d)
{

	if (d_volumeArray != NULL)
		cutilSafeCall(cudaFreeArray(d_volumeArray));

	cudaExtent ext;
	ext.width = w;
	ext.height = h;
	ext.depth = d;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cutilSafeCall( cudaMalloc3DArray(&d_volumeArray, &channelDesc, ext) );

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, w*sizeof(uchar)*4, w, h);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = ext;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cutilSafeCall( cudaMemcpy3D(&copyParams) );  

	// set texture parameters
	tex.normalized = true;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	cutilSafeCall(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

}


extern "C" 
void vr_freeCudaBuffers()
{
	cutilSafeCall(cudaFreeArray(d_volumeArray));
}

#endif

#endif //CUDA_SHADER_H

