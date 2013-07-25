#include "config.hpp"
#ifdef MMM1D_GPU

#include <stdio.h>
#include <sys/time.h>
#include "interaction_data.hpp"
#include "grid.hpp"
#include "atomic.cuh"
#include "Mmm1dgpuForce.hpp"
#include "mmm1d.hpp"
typedef mmm1dgpu_real real;

#ifdef ELECTROSTATICS_GPU_DOUBLE_PRECISION
#define M_LN2f  0.6931471805599453094172321214581766
#define C_GAMMAf        0.57721566490153286060651209008
#define C_2PIf  (2*3.14159265358979323846264338328)
#else
#define M_LN2f  0.6931471805599453094172321214581766f
#define C_GAMMAf        0.57721566490153286060651209008f
#define C_2PIf  (2*3.14159265358979323846264338328f)
#endif

int deviceCount;
float *multigpu_factors;

static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s:%d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#include "specfunc.cuh"
#include "mmm-common.cuh"

__forceinline__ __device__ real sqpow(real x)
{
	return pow(x,2);
	//return x*x;
}
__forceinline__ __device__ real cbpow(real x)
{
	return pow(x,3);
	//return x*x*x;
}

#if 0
__global__ void besseltestKernel(real *a, real *b0, real *b1, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		b0[tid] = dev_K0(a[tid]);
		b1[tid] = dev_K1(a[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

int besseltest(real *a, real *b0, real *b1, int N)
{
	real *dev_a, *dev_b0, *dev_b1;
	HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(real)) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_b0, N*sizeof(real)) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_b1, N*sizeof(real)) );
	
	HANDLE_ERROR( cudaMemcpy(dev_a, a, N*sizeof(real), cudaMemcpyHostToDevice) );
	besseltestKernel<<<N/32,32>>>(dev_a, dev_b0, dev_b1, N);
	HANDLE_ERROR( cudaMemcpy(b0, dev_b0, N*sizeof(real), cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(b1, dev_b1, N*sizeof(real), cudaMemcpyDeviceToHost) );
	
	cudaFree(dev_a);
	cudaFree(dev_b0);
	cudaFree(dev_b1);
	
	return 0;
}

__global__ void modpsitestKernel(int order, real *a, real *b, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		if (order % 2 == 0)
			b[tid] = dev_mod_psi_even(order/2, a[tid]);
		else
			b[tid] = dev_mod_psi_odd(order/2, a[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

int modpsitest(int order, real *a, real *b, int N)
{
	real *dev_a, *dev_b;
	modpsi_init();
	if (order/2 >= n_modPsi)
		return 1;
	HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(real)) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_b, N*sizeof(real)) );
	HANDLE_ERROR( cudaMemcpy(dev_a, a, N*sizeof(real), cudaMemcpyHostToDevice) );
	modpsitestKernel<<<N/32,32>>>(order, dev_a, dev_b, N);
	HANDLE_ERROR( cudaMemcpy(b, dev_b, N*sizeof(real), cudaMemcpyDeviceToHost) );
	cudaFree(dev_a);
	cudaFree(dev_b);
	modpsi_destroy();
	return 0;
}
#endif

int mmm1dgpu_devcount()
{
	HANDLE_ERROR( cudaGetDeviceCount(&deviceCount) );
	
	// estimate the speed of all available GPUs so we can distribute work proportionally to them
	cudaDeviceProp prop;
	float multigpu_total = 0;
	multigpu_factors = (float*) malloc (deviceCount * sizeof(float));
	for (int i=0; i<deviceCount; i++)
	{
		HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );
		multigpu_factors[i] = prop.clockRate * prop.multiProcessorCount * prop.warpSize; // each multiprocessor can run approximately one warp per clock cycle
		multigpu_total += multigpu_factors[i];
	}
	for (int i=0; i<deviceCount; i++)
	{
		multigpu_factors[i] /= multigpu_total;
	}
	
	return deviceCount;
}

__device__ void sumReduction(real *input, real *sum)
{
	int tid = threadIdx.x;
	for (int i = blockDim.x/2; i > 0; i /= 2)
	{
		__syncthreads();
		if (tid < i)
			input[tid] += input[i+tid];
	}
	__syncthreads();
	if (tid == 0)
		sum[0] = input[0];
}

__global__ void sumKernel(real *data, int N)
{
	extern __shared__ real partialsums[];
	if (blockIdx.x != 0) return;
	if (threadIdx.x >= N)
		partialsums[threadIdx.x] = 0;
	else
		partialsums[threadIdx.x] = data[threadIdx.x];
	sumReduction(partialsums, data);
}

__constant__ real far_switch_radius_2 = 0.05*0.05;
__constant__ real boxz;
__constant__ real uz;
__constant__ real coulomb_prefactor = 1.0;
__constant__ int bessel_cutoff = 5;
__constant__ real maxPWerror = 1e-5;

real host_boxz;

__global__ void besselTuneKernel(int *result, real far_switch_radius, int maxCut)
{
	real arg = C_2PIf*uz*far_switch_radius;
	real pref = 4*uz*max(1.0f, C_2PIf*uz);
	real err;
	int P = 1;
	do
	{
		err = pref*dev_K1(arg*P)*exp(arg)/arg*(P-1 + 1/arg);
		P++;
	} while (err > maxPWerror && P <= maxCut);
	P--;

	result[0] = P;
}

int mmm1dgpu_tune(const real* r, const real* q, int N, real _maxPWerror, real _far_switch_radius, int _bessel_cutoff)
{
	real far_switch_radius = _far_switch_radius;
	int bessel_cutoff = _bessel_cutoff;
	real maxrad = host_boxz;

	if (_far_switch_radius < 0 && _bessel_cutoff < 0)
	// autodetermine switching and bessel cutoff radius
	{
		real *force = (real*) malloc(3*N*sizeof(real));
		real bestrad = 0, besttime = INFINITY;

		for (far_switch_radius = 0.05*maxrad; far_switch_radius < maxrad; far_switch_radius += 0.05*maxrad)
		{
			mmm1dgpu_set_params(0, 0, _maxPWerror, far_switch_radius, bessel_cutoff);
			mmm1dgpu_tune(r, q, N, _maxPWerror, far_switch_radius, -2); // tune bessel cutoff
			int runtime = mmm1dgpu_forces(r, q, force, N);
			if (runtime < besttime)
			{
				besttime = runtime;
				bestrad = far_switch_radius;
			}
		}
		far_switch_radius = bestrad;
		free(force);

		mmm1dgpu_set_params(0, 0, _maxPWerror, far_switch_radius, bessel_cutoff);
		mmm1dgpu_tune(r, q, N, _maxPWerror, far_switch_radius, -2); // tune bessel cutoff
	}

	else if (_bessel_cutoff < 0)
	// autodetermine bessel cutoff
	{
		int *dev_cutoff;
		int maxCut = 30;
		HANDLE_ERROR( cudaMalloc((void**)&dev_cutoff, sizeof(int)) );
		besselTuneKernel<<<1,1>>>(dev_cutoff, far_switch_radius, maxCut);
		HANDLE_ERROR( cudaMemcpy(&bessel_cutoff, dev_cutoff, sizeof(int), cudaMemcpyDeviceToHost) );
		cudaFree(dev_cutoff);
		if (_bessel_cutoff != -2 && bessel_cutoff >= maxCut) // we already have our switching radius and only need to determine the cutoff, i.e. this is the final tuning round
		{
			printf("No reasonable Bessel cutoff could be determined.\n");
			exit(EXIT_FAILURE);
		}

		mmm1dgpu_set_params(0, 0, _maxPWerror, far_switch_radius, bessel_cutoff);
	}

	return 0;
}

int mmm1dgpu_set_params(real _boxz, real _coulomb_prefactor, real _maxPWerror, real _far_switch_radius, int _bessel_cutoff)
{
	if (_boxz > 0 && _far_switch_radius > _boxz)
	{
		printf("Far switch radius (%f) must not be larger than the box length (%f).\n", _far_switch_radius, _boxz);
		exit(EXIT_FAILURE);
	}
	real _far_switch_radius_2 = _far_switch_radius*_far_switch_radius;
	real _uz = 1.0/_boxz;
	for (int d = 0; d < deviceCount; d++)
	{
		cudaSetDevice(d);
		if (_far_switch_radius >= 0)
		{
			HANDLE_ERROR( cudaMemcpyToSymbol(far_switch_radius_2, &_far_switch_radius_2, sizeof(real)) );
			mmm1d_params.far_switch_radius_2 = _far_switch_radius*_far_switch_radius;
		}
		if (_boxz > 0)
		{
			host_boxz = _boxz;
			HANDLE_ERROR( cudaMemcpyToSymbol(boxz, &_boxz, sizeof(real)) );
			HANDLE_ERROR( cudaMemcpyToSymbol(uz, &_uz, sizeof(real)) );
		}
		if (_coulomb_prefactor != 0)
		{
			HANDLE_ERROR( cudaMemcpyToSymbol(coulomb_prefactor, &_coulomb_prefactor, sizeof(real)) );
		}
		if (_bessel_cutoff > 0)
		{
			HANDLE_ERROR( cudaMemcpyToSymbol(bessel_cutoff, &_bessel_cutoff, sizeof(int)) );
			mmm1d_params.bessel_cutoff = _bessel_cutoff;
		}
		if (_maxPWerror > 0)
		{
			HANDLE_ERROR( cudaMemcpyToSymbol(maxPWerror, &_maxPWerror, sizeof(real)) );
			mmm1d_params.maxPWerror = _maxPWerror;
		}
	}

	if (_far_switch_radius >= 0 && _bessel_cutoff > 0)
		printf("@@@ Using far switch radius %f and bessel cutoff %d\n", _far_switch_radius, _bessel_cutoff);

	return 0;
}

__global__ void forcesKernel(const __restrict__ real *r, const __restrict__ real *q, __restrict__ real *force, int N, int pairs, int tStart = 0, int tStop = -1)
{
	if (tStop < 0)
		tStop = N*N;

	for (int tid = threadIdx.x + blockIdx.x * blockDim.x + tStart; tid < tStop; tid += blockDim.x * gridDim.x)
	{
		int p1 = tid%N, p2 = tid/N;
		real x = r[3*p2] - r[3*p1], y = r[3*p2+1] - r[3*p1+1], z = r[3*p2+2] - r[3*p1+2];
		real rxy2 = sqpow(x) + sqpow(y);
		real rxy = sqrt(rxy2);
		real sum_r = 0, sum_z = 0;
		
		if (boxz <= 0.0) return; // otherwise we'd get into an infinite loop if we're not initialized correctly

		while (fabs(z) > boxz/2) // make sure we take the shortest distance
			z -= (z > 0? 1 : -1)*boxz;

		if (p1 == p2 || rxy == 0) // TODO: rxy==0 is wrong!!!
		{
			rxy = 1; // so the multiplication at the end doesn't fail with NaNs
		}
		else if (rxy2 <= far_switch_radius_2) // near formula
		{
			real uzz = uz*z;
			real uzr = uz*rxy;
			sum_z = dev_mod_psi_odd(0, uzz);
			real uzrpow = uzr;
			for (int n = 1; n < device_n_modPsi; n++)
			{
				real sum_r_old = sum_r;
				real mpe = dev_mod_psi_even(n, uzz);
     			real mpo = dev_mod_psi_odd(n, uzz);

     			sum_r += 2*n*mpe * uzrpow;
     			uzrpow *= uzr;
     			sum_z += mpo * uzrpow;
     			uzrpow *= uzr;

     			if (fabs(sum_r_old - sum_r) < maxPWerror)
					break;
			}

			sum_r *= sqpow(uz);
			sum_z *= sqpow(uz);

			sum_r += rxy*cbpow(rsqrt(rxy2+pow(z,2)));
			sum_r += rxy*cbpow(rsqrt(rxy2+pow(z+boxz,2)));
			sum_r += rxy*cbpow(rsqrt(rxy2+pow(z-boxz,2)));

			sum_z += z*cbpow(rsqrt(rxy2+pow(z,2)));
			sum_z += (z+boxz)*cbpow(rsqrt(rxy2+pow(z+boxz,2)));
			sum_z += (z-boxz)*cbpow(rsqrt(rxy2+pow(z-boxz,2)));
		}
		else // far formula
		{
			for (int p = 1; p < bessel_cutoff; p++)
			{
				real arg = C_2PIf*uz*p;
				sum_r += p*dev_K1(arg*rxy)*cos(arg*z);
				sum_z += p*dev_K0(arg*rxy)*sin(arg*z);
			}
			sum_r *= sqpow(uz)*4*C_2PIf;
			sum_z *= sqpow(uz)*4*C_2PIf;
			sum_r += 2*uz/rxy;
		}

		real pref = coulomb_prefactor*q[p1]*q[p2];
		if (pairs)
		{
			force[3*(p1+p2*N-tStart)] = pref*sum_r/rxy*x;
			force[3*(p1+p2*N-tStart)+1] = pref*sum_r/rxy*y;
			force[3*(p1+p2*N-tStart)+2] = pref*sum_z;
		}
		else
		{
#ifdef ELECTROSTATICS_GPU_DOUBLE_PRECISION
			atomicadd8(&force[3*p2], pref*sum_r/rxy*x);
			atomicadd8(&force[3*p2+1], pref*sum_r/rxy*y);
			atomicadd8(&force[3*p2+2], pref*sum_z);
#else
			atomicadd(&force[3*p2], pref*sum_r/rxy*x);
			atomicadd(&force[3*p2+1], pref*sum_r/rxy*y);
			atomicadd(&force[3*p2+2], pref*sum_z);
#endif
		}
	}
}

__global__ void energiesKernel(const __restrict__ real *r, const __restrict__ real *q, __restrict__ real *energy, int N, int pairs, int tStart = 0, int tStop = -1)
{
	if (tStop < 0)
		tStop = N*N;

	extern __shared__ real partialsums[];
	if (!pairs)
	{
		partialsums[threadIdx.x] = 0;
		__syncthreads();
	}
	for (int tid = threadIdx.x + blockIdx.x * blockDim.x + tStart; tid < tStop; tid += blockDim.x * gridDim.x)
	{
		int p1 = tid%N, p2 = tid/N;
		real z = r[3*p2+2] - r[3*p1+2];
		real rxy2 = sqpow(r[3*p2] - r[3*p1]) + sqpow(r[3*p2+1] - r[3*p1+1]);
		real rxy = sqrt(rxy2);
		real sum_e = 0;

		if (boxz <= 0.0) return; // otherwise we'd get into an infinite loop if we're not initialized correctly

		while (fabs(z) > boxz/2) // make sure we take the shortest distance
			z -= (z > 0? 1 : -1)*boxz;

		if (p1 == p2)
		{
		}
		else if (rxy2 <= far_switch_radius_2) // near formula
		{
			real uzz = uz*z;
			real uzr2 = sqpow(uz*rxy);
			real uzrpow = uzr2;
			sum_e = dev_mod_psi_even(0, uzz);
			for (int n = 1; n < device_n_modPsi; n++)
			{
				real sum_e_old = sum_e;
				real mpe = dev_mod_psi_even(n, uzz);
     			sum_e += mpe * uzrpow;
     			uzrpow *= uzr2;
				
				if (fabs(sum_e_old - sum_e) < maxPWerror)
					break;
			}

			sum_e *= -1*uz;
			sum_e -= 2*uz*C_GAMMAf;
			sum_e += 1*(rsqrt(rxy2+sqpow(z)));
			sum_e += 1*(rsqrt(rxy2+sqpow(z+boxz)));
			sum_e += 1*(rsqrt(rxy2+sqpow(z-boxz)));
		}
		else // far formula
		{
			sum_e = -(log(rxy*uz/2) + C_GAMMAf)/2;
			for (int p = 1; p < bessel_cutoff; p++)
			{
				real arg = C_2PIf*uz*p;
				sum_e += dev_K0(arg*rxy)*cos(arg*z);
			}
			sum_e *= uz*4;
		}

		if (pairs)
		{
			energy[p1+p2*N-tStart] = coulomb_prefactor*q[p1]*q[p2]*sum_e;
		}
		else
		{
			partialsums[threadIdx.x] += coulomb_prefactor*q[p1]*q[p2]*sum_e;
		}
	}
	if (!pairs)
	{
		sumReduction(partialsums, &energy[blockIdx.x]);
	}
}

__global__ void vectorReductionKernel(real *src, real *dst, int N, int tStart = 0, int tStop = -1)
{
	if (tStop < 0)
		tStop = N*N;

	for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += blockDim.x * gridDim.x)
	{
		int offset = ((tid + (tStart % N)) % N);
		
		for (int i = 0; tid+i*N < (tStop - tStart); i++)
		{
			#pragma unroll 3
			for (int d = 0; d<3; d++)
			{
				dst[3*offset+d] -= src[3*(tid+i*N)+d];
			}
		}
	}
}

void mmm1dgpu_init()
{
	mmm1dgpu_devcount();
	modpsi_init();
}

/*	pairs=0: return forces (using atomicAdd)
	pairs=2: return forces (using global memory reduction)
	pairs=1: return force pairs */
long long mmm1dgpu_forces(const real *r, const real *q, real *force, int N, int pairs)
{
	struct timeval begin, end;
	gettimeofday(&begin, NULL);

	if (host_boxz <= 0)
	{
		printf("Error: box length is zero!\n");
		exit(EXIT_FAILURE);
	}

	// for all but the largest systems, it is faster to store force pairs and then sum them up
	// so unless we're limited by memory, do the latter
	if (!pairs)
	{
		int tStart = 0;
		pairs = 2;
		for (int d = 0; d < deviceCount; d++)
		{
			int tStop = tStart + multigpu_factors[d]*N*N;
			if (tStop > N*N || d == (deviceCount-1)) tStop = N*N;

			size_t freeMem, totalMem;
			cudaSetDevice(d);
			cudaMemGetInfo(&freeMem, &totalMem);
			if (freeMem/2 < (tStop-tStart)*sizeof(real)) // don't use more than half the device's memory
			{
				printf("Switching to atomicAdd due to memory constraints.\n");
				pairs = 0;
				break;
			}

			tStart = tStop;
		}
	}
	
	float multigpu_total = 0;
	cudaStream_t *stream = (cudaStream_t *) malloc (deviceCount * sizeof(cudaStream_t));
	cudaEvent_t *eventStart = (cudaEvent_t *) malloc (deviceCount * sizeof(cudaEvent_t));
	cudaEvent_t *eventStop = (cudaEvent_t *) malloc (deviceCount * sizeof(cudaEvent_t));

	real **dev_r = (real**) malloc (deviceCount * sizeof(real*));
	real **dev_q = (real**) malloc (deviceCount * sizeof(real*));
	real **dev_force;
	if (pairs == 2)
		dev_force = (real**) malloc (2*deviceCount * sizeof(real*));
	else
		dev_force = (real**) malloc (deviceCount * sizeof(real*));

	float *elapsedTime = (float*) malloc (deviceCount * sizeof(float));
	real **forcePartial;
	if (pairs == 1 && deviceCount > 1)
	{
		HANDLE_ERROR( cudaHostRegister(force, 3*N*N*sizeof(real), cudaHostRegisterPortable) );
	}
	else if (pairs != 1)
	{
		forcePartial = (real**) malloc (deviceCount * sizeof(real*));
	}

	int tStart = 0;
	for (int d = 0; d < deviceCount; d++)
	{
		int tStop = tStart + multigpu_factors[d]*N*N;
		if (tStop > N*N || d == (deviceCount-1)) tStop = N*N;

		int numThreads = 64;
		int numBlocks;
		if ((tStop-tStart) < numThreads)
			numBlocks = 1;
		else
			numBlocks = (tStop-tStart)/numThreads+1;
		if (numBlocks > 65535)
			numBlocks = 65535;
		
		cudaSetDevice(d);
		cudaStreamCreate(&stream[d]);
		HANDLE_ERROR( cudaMalloc((void**)&dev_r[d], 3*N*sizeof(real)) );
		HANDLE_ERROR( cudaMalloc((void**)&dev_q[d], N*sizeof(real)) );
		if (pairs)
		{
			HANDLE_ERROR( cudaMalloc((void**)&dev_force[d], 3*(tStop-tStart)*sizeof(real)) );
		}
		else
		{
			HANDLE_ERROR( cudaMalloc((void**)&dev_force[d], 3*N*sizeof(real)) );
			HANDLE_ERROR( cudaMemsetAsync(dev_force[d], 0, 3*N*sizeof(real), stream[d]) ); // zero out for atomic add
		}

		HANDLE_ERROR( cudaMemcpyAsync(dev_r[d], r, 3*N*sizeof(real), cudaMemcpyHostToDevice, stream[d]) );
		HANDLE_ERROR( cudaMemcpyAsync(dev_q[d], q, N*sizeof(real), cudaMemcpyHostToDevice, stream[d]) );

		HANDLE_ERROR( cudaEventCreate(&eventStart[d]) );
		HANDLE_ERROR( cudaEventCreate(&eventStop[d]) );
		HANDLE_ERROR( cudaEventRecord(eventStart[d], stream[d]) );
		
		forcesKernel<<<numBlocks, numThreads, 0, stream[d]>>>(dev_r[d], dev_q[d], dev_force[d], N, pairs, tStart, tStop);
	
		HANDLE_ERROR( cudaEventRecord(eventStop[d], stream[d]) );

		if (pairs == 2)
		{
			// call reduction kernel
			HANDLE_ERROR( cudaMalloc((void**)&dev_force[d+deviceCount], 3*N*sizeof(real)) );
			HANDLE_ERROR( cudaMemsetAsync(dev_force[d+deviceCount], 0, 3*N*sizeof(real), stream[d]) ); // zero out for reduction
			vectorReductionKernel<<<N/numThreads+1, numThreads, 0, stream[d]>>>(dev_force[d], dev_force[d+deviceCount], N, tStart, tStop);
			HANDLE_ERROR( cudaMallocHost(&forcePartial[d], 3*N*sizeof(real)) );
			HANDLE_ERROR( cudaMemcpyAsync(forcePartial[d], dev_force[d+deviceCount], 3*N*sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}

		else if (pairs == 1)
		{
			HANDLE_ERROR( cudaMemcpyAsync(&force[3*tStart], dev_force[d], 3*(tStop-tStart)*sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}
		else
		{
			HANDLE_ERROR( cudaMallocHost(&forcePartial[d], 3*N*sizeof(real)) );
			HANDLE_ERROR( cudaMemcpyAsync(forcePartial[d], dev_force[d], 3*N*sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}
		
		multigpu_factors[d] = tStop-tStart;
		tStart = tStop;
	}

	float elapsedTimeMax = 0;
	
	for (int d = 0; d < deviceCount; d++)
	{
		cudaSetDevice(d);
		
		HANDLE_ERROR( cudaEventSynchronize(eventStop[d]) );
		HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime[d], eventStart[d], eventStop[d]) );
		printf(">>> Calculated on GPU %d in %3.3f ms\n", d, elapsedTime[d]);
		HANDLE_ERROR( cudaEventDestroy(eventStart[d]) );
		HANDLE_ERROR( cudaEventDestroy(eventStop[d]) );
		
		cudaStreamSynchronize(stream[d]);
		cudaFree(dev_r[d]);
		cudaFree(dev_q[d]);
		cudaFree(dev_force[d]);
		if (pairs == 2)
			cudaFree(dev_force[d+deviceCount]);
		cudaStreamDestroy(stream[d]);
		
		if (pairs != 1)
		{
			for (int i = 0; i < 3*N; i++)
				force[i] += forcePartial[d][i];
			cudaFreeHost(forcePartial[d]);
		}
		
		// update multigpu load balancing factors
		multigpu_factors[d] /= elapsedTime[d];
		multigpu_total += multigpu_factors[d];
		if (elapsedTime[d] > elapsedTimeMax)
			elapsedTimeMax = elapsedTime[d];
	}
	for (int d = 0; d < deviceCount; d++)
	{
		multigpu_factors[d] /= multigpu_total;
	}
	
	free(stream);
	free(eventStart);
	free(eventStop);
	free(dev_r);
	free(dev_q);
	free(dev_force);
	free(elapsedTime);
	if (pairs == 1 && deviceCount > 1)
	{
		cudaHostUnregister(force); // this is very slow
	}
	else if (pairs != 1)
	{
		free(forcePartial);
	}
	cudaSetDevice(0);

	gettimeofday(&end, NULL);
	long long microsec = (end.tv_usec - begin.tv_usec) + 1e6*(end.tv_sec - begin.tv_sec);

	return  ((microsec << 32) + (int)(elapsedTimeMax*1000));
}

long long mmm1dgpu_energies(const real *r, const real *q, real *energy, int N, int pairs)
{
	struct timeval begin, end;
	gettimeofday(&begin, NULL);

	if (host_boxz <= 0)
	{
		printf("Error: box length is zero!\n");
		exit(EXIT_FAILURE);
	}

	float multigpu_total = 0;
	cudaStream_t *stream = (cudaStream_t *) malloc (deviceCount * sizeof(cudaStream_t));
	cudaEvent_t *eventStart = (cudaEvent_t *) malloc (deviceCount * sizeof(cudaEvent_t));
	cudaEvent_t *eventStop = (cudaEvent_t *) malloc (deviceCount * sizeof(cudaEvent_t));

	real **dev_r = (real**) malloc (deviceCount * sizeof(real*));
	real **dev_q = (real**) malloc (deviceCount * sizeof(real*));
	real **dev_energy = (real**) malloc (deviceCount * sizeof(real*));
	float *elapsedTime = (float*) malloc (deviceCount * sizeof(float));
	real *energyPartial;
	if (pairs && deviceCount > 1)
	{
		HANDLE_ERROR( cudaHostRegister(energy, N*N*sizeof(real), cudaHostRegisterPortable) );
	}
	else if (!pairs)
	{
		HANDLE_ERROR( cudaMallocHost(&energyPartial, deviceCount*sizeof(real)) );
	}

	int tStart = 0;
	for (int d = 0; d < deviceCount; d++)
	{
		int tStop = tStart + multigpu_factors[d]*N*N;
		if (tStop > N*N || d == (deviceCount-1)) tStop = N*N;

		int numThreads = 64;
		int numBlocks;
		if ((tStop-tStart) < numThreads)
			numBlocks = 1;
		else
			numBlocks = (tStop-tStart)/numThreads+1;
		if (numBlocks > 65535)
			numBlocks = 65535;
		
		cudaSetDevice(d);
		cudaStreamCreate(&stream[d]);
		HANDLE_ERROR( cudaMalloc((void**)&dev_r[d], 3*N*sizeof(real)) );
		HANDLE_ERROR( cudaMalloc((void**)&dev_q[d], N*sizeof(real)) );
		if (pairs)
		{
			HANDLE_ERROR( cudaMalloc((void**)&dev_energy[d], N*N*sizeof(real)) );
		}
		else
		{
			HANDLE_ERROR( cudaMalloc((void**)&dev_energy[d], numBlocks*sizeof(real)) );
		}

		HANDLE_ERROR( cudaMemcpyAsync(dev_r[d], r, 3*N*sizeof(real), cudaMemcpyHostToDevice, stream[d]) );
		HANDLE_ERROR( cudaMemcpyAsync(dev_q[d], q, N*sizeof(real), cudaMemcpyHostToDevice, stream[d]) );

		HANDLE_ERROR( cudaEventCreate(&eventStart[d]) );
		HANDLE_ERROR( cudaEventCreate(&eventStop[d]) );
		HANDLE_ERROR( cudaEventRecord(eventStart[d], stream[d]) );

		energiesKernel<<<numBlocks, numThreads, numThreads*sizeof(real), stream[d]>>>(dev_r[d], dev_q[d], dev_energy[d], N, pairs, tStart, tStop);

		HANDLE_ERROR( cudaEventRecord(eventStop[d], stream[d]) );

		if (pairs)
		{
			HANDLE_ERROR( cudaMemcpyAsync(&energy[tStart], dev_energy[d], (tStop-tStart)*sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}
		else
		{
			int size2 = pow(2,ceil(log2((float) numBlocks))); // Reduction only works on powers of two
			sumKernel<<<1,size2,size2*sizeof(real)>>>(dev_energy[d], numBlocks);
			HANDLE_ERROR( cudaMemcpyAsync(&energyPartial[d], dev_energy[d], sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}

		multigpu_factors[d] = tStop-tStart;
		tStart = tStop;
	}

	float elapsedTimeMax = 0;

	for (int d = 0; d < deviceCount; d++)
	{
		cudaSetDevice(d);
		
		HANDLE_ERROR( cudaEventSynchronize(eventStop[d]) );
		HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime[d], eventStart[d], eventStop[d]) );
		printf(">>> Calculated on GPU %d in %3.3f ms\n", d, elapsedTime[d]);
		HANDLE_ERROR( cudaEventDestroy(eventStart[d]) );
		HANDLE_ERROR( cudaEventDestroy(eventStop[d]) );
		
		cudaStreamSynchronize(stream[d]);
		cudaFree(dev_r[d]);
		cudaFree(dev_q[d]);
		cudaFree(dev_energy[d]);
		cudaStreamDestroy(stream[d]);

		if (!pairs)
		{
			energy[0] += energyPartial[d];
		}

		// update multigpu load balancing factors
		multigpu_factors[d] /= elapsedTime[d];
		multigpu_total += multigpu_factors[d];
		if (elapsedTime[d] > elapsedTimeMax)
			elapsedTimeMax = elapsedTime[d];
	}
	for (int d = 0; d < deviceCount; d++)
	{
		multigpu_factors[d] /= multigpu_total;
	}

	free(stream);
	free(eventStart);
	free(eventStop);
	free(dev_r);
	free(dev_q);
	free(dev_energy);
	free(elapsedTime);
	if (pairs && deviceCount > 1)
	{
		cudaHostUnregister(energy); // this is very slow
	}
	else if (!pairs)
	{
		cudaFreeHost(energyPartial);
	}
	cudaSetDevice(0);

	gettimeofday(&end, NULL);
	long long microsec = (end.tv_usec - begin.tv_usec) + 1e6*(end.tv_sec - begin.tv_sec);
	return ((microsec << 32) + (int)(elapsedTimeMax*1000));
}

/* C++ Espresso Interface code below */

Mmm1dgpuForce::Mmm1dgpuForce(real _coulomb_prefactor, real _maxPWerror, real _far_switch_radius, int _bessel_cutoff)
:initialized(0), N(-1), coulomb_prefactor(_coulomb_prefactor), maxPWerror(_maxPWerror), far_switch_radius(_far_switch_radius), bessel_cutoff(_bessel_cutoff)
{
	if (PERIODIC(0) || PERIODIC(1) || !PERIODIC(2))
	{
		printf("MMM1D requires periodicity (0,0,1)\n");
		exit(EXIT_FAILURE);
	}

#if defined(ELECTROSTATICS_GPU_DOUBLE_PRECISION)
	HANDLE_ERROR( cudaGetDeviceCount(&deviceCount) );
	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		if (devProp.major < 2)
		{
			printf("Compute capability 2.0 or higher is required for double precision support.\n");
			exit(EXIT_FAILURE);
		}
	}
#endif
	mmm1dgpu_init();
	coulomb.method = COULOMB_MMM1D_GPU;
}

Mmm1dgpuForce::~Mmm1dgpuForce()
{
	if (initialized) destroy();
}

void Mmm1dgpuForce::init(SystemInterface &s)
{
	// only reinitialize if the number of particles has changed
	if (N == s.npart())
		return;

	if (initialized) destroy();
	initialized = 1;

	N = s.npart();
	F.reserve(s.npart());

	if (s.box()[2] <= 0)
	{
		printf("Error: box length is zero!\n");
		exit(EXIT_FAILURE);
	}

	if (N <= 0 && far_switch_radius < 0)
	{
		printf("Warning: Please add particles to system before intializing.\n");
		printf("Tuning will be disabled! Setting far switch radius to half box length.\n");
		far_switch_radius = s.box()[2]/2;
	}

	force = (real*) malloc(3*N*sizeof(real));
	r = (real*) malloc(3*N*sizeof(real));
	q = (real*) malloc(N*sizeof(real));
	int offset = 0;
	for (SystemInterface::const_vec_iterator &it = s.rBegin(); it != s.rEnd(); ++it)
	{
		for (int d = 0; d < 3; d++)
			r[3*offset+d] = (*it)[d];
		offset++;
	}
	offset = 0;
	for (SystemInterface::const_real_iterator &it = s.qBegin(); it != s.qEnd(); ++it)
	{
		q[offset] = *it;
		offset++;
	}

	mmm1dgpu_set_params(s.box()[2], coulomb_prefactor, maxPWerror, far_switch_radius,bessel_cutoff);
	mmm1dgpu_tune(r, q, N, maxPWerror, far_switch_radius, bessel_cutoff);

	// for all but the largest systems, it is faster to store force pairs and then sum them up
	// so unless we're limited by memory, do the latter
	int tStart = 0;
	pairs = 2;
	for (int d = 0; d < deviceCount; d++)
	{
		int tStop = tStart + multigpu_factors[d]*N*N;
		if (tStop > N*N || d == (deviceCount-1)) tStop = N*N;

		cudaSetDevice(d);

		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);
		if (freeMem/2 < (tStop-tStart)*sizeof(real)) // don't use more than half the device's memory
		{
			printf("Switching to atomicAdd due to memory constraints.\n");
			pairs = 0;
			break;
		}

		tStart = tStop;
	}
	
	stream = (cudaStream_t *) malloc (deviceCount * sizeof(cudaStream_t));
	eventStart = (cudaEvent_t *) malloc (deviceCount * sizeof(cudaEvent_t));
	eventStop = (cudaEvent_t *) malloc (deviceCount * sizeof(cudaEvent_t));

	dev_r = (real**) malloc (deviceCount * sizeof(real*));
	dev_q = (real**) malloc (deviceCount * sizeof(real*));
	if (pairs == 2)
		dev_force = (real**) malloc (2*deviceCount * sizeof(real*));
	else
		dev_force = (real**) malloc (deviceCount * sizeof(real*));

	forcePartial = (real**) malloc (deviceCount * sizeof(real*));

	tStart = 0;
	for (int d = 0; d < deviceCount; d++)
	{
		int tStop = tStart + multigpu_factors[d]*N*N;
		if (tStop > N*N || d == (deviceCount-1)) tStop = N*N;

		cudaSetDevice(d);

		numThreads = 64;
		if ((tStop-tStart) < numThreads)
			numBlocks = 1;
		else
			numBlocks = (tStop-tStart)/numThreads+1;
		if (numBlocks > 65535)
			numBlocks = 65535;
					
		HANDLE_ERROR( cudaStreamCreate(&stream[d]) );
		HANDLE_ERROR( cudaMalloc((void**)&dev_r[d], 3*N*sizeof(real)) );
		HANDLE_ERROR( cudaMalloc((void**)&dev_q[d], N*sizeof(real)) );
		if (pairs)
		{
			HANDLE_ERROR( cudaMalloc((void**)&dev_force[d], 3*(tStop-tStart)*sizeof(real)) );
			if (pairs == 2)
			{
				HANDLE_ERROR( cudaMalloc((void**)&dev_force[d+deviceCount], 3*N*sizeof(real)) );
			}
		}
		else
		{
			HANDLE_ERROR( cudaMalloc((void**)&dev_force[d], 3*N*sizeof(real)) );
		}
		HANDLE_ERROR( cudaMallocHost(&forcePartial[d], 3*N*sizeof(real)) );

		HANDLE_ERROR( cudaEventCreate(&eventStart[d]) );
		HANDLE_ERROR( cudaEventCreate(&eventStop[d]) );

		tStart = tStop;
	}
}

void Mmm1dgpuForce::run(SystemInterface &s)
{
	if (coulomb.method != COULOMB_MMM1D_GPU)
	{
		printf("Error: It is currently not supported to disable forces using the EspressoSystemInterface.\n");
		exit(EXIT_FAILURE);
	}
	if (N != s.npart())
	{
		printf("Error: number of particles changed between init (%d) and run (%d).\n", N, s.npart());
		exit(EXIT_FAILURE);
	}
	if (host_boxz != s.box()[2])
	{
		printf("Error: box length changed between init (%d) and run (%d).\n", host_boxz, s.box()[2]);
		exit(EXIT_FAILURE);
	}

	// update coulomb prefactor
	coulomb_prefactor = coulomb.prefactor;
	mmm1dgpu_set_params(0, coulomb_prefactor);

	F.clear();
	memset(force, 0, 3*N*sizeof(real));
	int offset = 0;
	for (SystemInterface::const_vec_iterator &it = s.rBegin(); it != s.rEnd(); ++it)
	{
		for (int d = 0; d < 3; d++)
			r[3*offset+d] = (*it)[d];
		offset++;
	}
	offset = 0;
	for (SystemInterface::const_real_iterator &it = s.qBegin(); it != s.qEnd(); ++it)
	{
		q[offset] = *it;
		offset++;
	}

	int tStart = 0;
	for (int d = 0; d < deviceCount; d++)
	{
		int tStop = tStart + multigpu_factors[d]*N*N;
		if (tStop > N*N || d == (deviceCount-1)) tStop = N*N;

		cudaSetDevice(d);

		if (!pairs)
		{
			HANDLE_ERROR( cudaMemsetAsync(dev_force[d], 0, 3*N*sizeof(real), stream[d]) ); // zero out for atomic add
		}
		/*else
		{
			HANDLE_ERROR( cudaMemsetAsync(dev_force[d], 0, 3*(tStop-tStart)*sizeof(real), stream[d]) ); // not necessary
		}*/

		HANDLE_ERROR( cudaMemcpyAsync(dev_r[d], r, 3*N*sizeof(real), cudaMemcpyHostToDevice, stream[d]) );
		HANDLE_ERROR( cudaMemcpyAsync(dev_q[d], q, N*sizeof(real), cudaMemcpyHostToDevice, stream[d]) );

		HANDLE_ERROR( cudaEventRecord(eventStart[d], stream[d]) );
		forcesKernel<<<numBlocks, numThreads, 0, stream[d]>>>(dev_r[d], dev_q[d], dev_force[d], N, pairs, tStart, tStop);
		HANDLE_ERROR( cudaEventRecord(eventStop[d], stream[d]) );

		if (pairs == 2)
		{
			// call reduction kernel
			HANDLE_ERROR( cudaMemsetAsync(dev_force[d+deviceCount], 0, 3*N*sizeof(real), stream[d]) ); // zero out for reduction
			vectorReductionKernel<<<N/numThreads+1, numThreads, 0, stream[d]>>>(dev_force[d], dev_force[d+deviceCount], N, tStart, tStop);
			HANDLE_ERROR( cudaMemcpyAsync(forcePartial[d], dev_force[d+deviceCount], 3*N*sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}
		else
		{
			HANDLE_ERROR( cudaMemcpyAsync(forcePartial[d], dev_force[d], 3*N*sizeof(real), cudaMemcpyDeviceToHost, stream[d]) );
		}
		
		multigpu_factors[d] = tStop-tStart;
		tStart = tStop;
	}
}

bool Mmm1dgpuForce::isReady()
{
	for (int d = 0; d < deviceCount; d++)
	{
		cudaSetDevice(d);
		if (cudaStreamQuery(stream[d]) != cudaSuccess)
			return 0;
	}

	float elapsedTimeMax = 0;
	float multigpu_total = 0;
	
	// if we are ready, pull down the data
	for (int d = 0; d < deviceCount; d++)
	{
		cudaSetDevice(d);
		
		float elapsedTime;
		HANDLE_ERROR( cudaEventSynchronize(eventStop[d]) );
		HANDLE_ERROR( cudaEventElapsedTime(&elapsedTime, eventStart[d], eventStop[d]) );
		
		cudaStreamSynchronize(stream[d]);

		for (int i = 0; i < 3*N; i++)
			force[i] += forcePartial[d][i];
		
		// update multigpu load balancing factors
		multigpu_factors[d] /= elapsedTime;
		multigpu_total += multigpu_factors[d];
		if (elapsedTime > elapsedTimeMax)
			elapsedTimeMax = elapsedTime;
	}
	for (int d = 0; d < deviceCount; d++)
	{
		multigpu_factors[d] /= multigpu_total;
	}

	for (int i = 0; i < N; i++)
	{
		F.push_back(SystemInterface::Vector3(force[3*i], force[3*i+1], force[3*i+2]));
	}
	
	cudaSetDevice(0);

	return 1;
}

void Mmm1dgpuForce::destroy()
{
	if (!initialized) return;
	initialized = 0;

	for (int d = 0; d < deviceCount; d++)
	{
		cudaSetDevice(d);
		
		HANDLE_ERROR( cudaEventDestroy(eventStart[d]) );
		HANDLE_ERROR( cudaEventDestroy(eventStop[d]) );

		cudaFreeHost(forcePartial[d]);
		
		cudaFree(dev_r[d]);
		cudaFree(dev_q[d]);
		cudaFree(dev_force[d]);
		if (pairs == 2)
			cudaFree(dev_force[d+deviceCount]);
		cudaStreamDestroy(stream[d]);
	}

	free(stream);
	free(eventStart);
	free(eventStop);
	free(dev_r);
	free(dev_q);
	free(dev_force);
	free(forcePartial);

	free(r);
	free(q);
	free(force);
}

#endif /* MMM1D_GPU */