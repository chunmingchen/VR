////////////////////////////////////////////////////////////////////////////////////////////////
//common settings for PC and GPU
////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef HOST_COMMON_H
#define HOST_COMMON_H

#define SEG_ALPHA_FORMAT // 2channels: segment - alpha
#define LOD_DATA_TYPE uchar2

//// regarding LOD rendering
// Modify here for your senario :
//#define CUDA_DEVICE_9400M
#define CUDA_DEVICE_8800   //8800 or up

// brickReqBuffer
#define BRICK_REQ_MAX_LEN 3	// affects request buffer length
#ifdef CUDA_DEVICE_8800
	#define INIT_BRICK_UPLOAD_LEVEL 3
	#define BASE_MAX_STEPS 256
	//#define BRICK_MEM_LIMIT 150000000 //limit brick memory size (bytes)
#elif defined(CUDA_DEVICE_9400M)
	#define INIT_BRICK_UPLOAD_LEVEL 2
	#define BASE_MAX_STEPS 100
#endif

// for debugging:
//#define PRINT_FPS
//#define PRINT_CURSER_COLOR

// for debugging cuda codes
//#define NO_BRICKS       // don't use brick pool. only use constant color 

//#define PINNED_MEM  // not working currently

//////////////////////////////////////////

#ifdef _DEBUG
#define PRINT(...) printf(__VA_ARGS__)
#else
#define PRINT(...) NULL  //<--this should work on C99
#endif

#define BLOCK_SIZE_W 16	//cuda block size per thread 
#define BLOCK_SIZE_H 16	//cuda block size per thread 

////////////////////////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#define CUDA_SAFE_CALL(call)\
{\
	cudaError err=call;\
	if(cudaSuccess != err)\
	{\
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        /*exit(-1); */                                                 \
	}\
}

////////////////////////////////////////////////////////////////////////////////////////////////
typedef unsigned int  uint;
typedef unsigned char uchar;


typedef struct {
	int imgWidth, imgHeight;
	int imgWidth_coalesced;
	float step;				// step size
	int maxSteps;			// limit steps per frame
	float clippingVisible;	// clipping plane
	float unitLen;			// UNIT_LEN 2.41421f  //screen_height=1 /tan(45/2/180*PI) = 2.41421  (FOV=45 degree) (in [0..1] space)
	int maxVolWidthInVoxel;	// max volume size to render
	float volBoundry[3];	// vol size in -1..1
	float invViewMatrix[16];
	float dof;				// user defined depth of field
	float unit_pixHeight;	// =  (float)c_vrParam.maxVolWidthInVoxel / (minw * c_vrParam.unitLen) * c_vrParam.dof;// todo: save as constant
	//float intensity;
	float *d_zBuffer;

	char restartRay;		// cannot use bool for kernel!?
	char bFastRender;
}VRParam;

/////////////////////////////
// LOD:


// need to make sure the size is the same in CPU and GPU
typedef struct  {
	float tvisible;
	float t;
	float tfar;
	int bricksUsedCount; // bricks traversed count
	////////////////////////////
	// for coalesced access issue, other parameters should be put in another structure
	//float4 outColor;
} LODPreState;






typedef struct {
	unsigned char a,b,g,r;
}TestColor;

typedef struct _VRState{
	int requestContinue;	// used in LOD

	// for debug
	//float debug;
	//float vec[4];
	//unsigned int global_id[3];
}VRState;




// brickReqBuffer
typedef unsigned int BrickReq;
typedef BrickReq BrickKeepReqBuffer[BRICK_REQ_MAX_LEN];

typedef struct /*__align__(16)*/ {
	BrickReq brickMissReq;	// 4 bytes
	BrickKeepReqBuffer brickKeepReqBuffer; // 4*BRICK_REQ_MAX_LEN
}LODRequest;

// Main LOD parameter
typedef struct {

	int brickWidth;		// 16 or 32
	int brickWidthLog;	// log(16 or 32)
	int levels;			// LOD levels
	int useLevels;		// max levels to be used by app.
	//
	int maxVolWidthInVoxel;
	float volBoundry[3];

	LODPreState *db_preState; // db: device buffer 
	LODRequest *db_request;		
	float4 *db_preColor;
	//char *db_segment;
}LODParam;

typedef uint2 TexelNode;

//////////////////////////////////////////////////////////////////////////
extern "C"
{
	// render
	void g_uploadVRParam(const VRParam *pVRParam);
	void g_getVRState(VRState *pVRState) ;
	void g_setVRState(VRState *vrState) ;
	void g_releaseRenderBuffers(VRParam *vrParam);
	void g_createRenderBuffers(VRParam *vrParam);



	// LOD
	bool g_createLODBuffers(LODParam *lodParam, LODRequest **ph_lodRequestArray, int imageSize_coalesced);
	void g_releaseLODBuffers(LODParam *lodParam, LODRequest **ph_lodRequestArray);

	bool g_createNodePool(int w, int h, int d);
	bool g_releaseNodePool();
	bool g_uploadOneNode(TexelNode *node, int x, int y, int z, int arrayPos);

	bool g_uploadLODParam(LODParam *lodParam);
	bool g_getLODRequest(LODRequest *db_request, LODRequest *lodRequest, int imageSize_coalesced);

	void g_createBrickPool(int w, int h, int d);
	void g_releaseBrickPool();
	void g_uploadBrickPool(LOD_DATA_TYPE *h_volume, int w, int h, int d, int x, int y, int z);

	// transfer function

	void g_releaseTrFn();
	void g_createTrFn(int w, int h);
	void g_uploadTrFn(void *data, int len);

	// shading
	void g_setPhongShading();
}

#endif //HOST_COMMON_H
