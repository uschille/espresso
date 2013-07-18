#ifndef MMM1DGPUFORCE_H
#define MMM1DGPUFORCE_H

#include "config.hpp"
#ifdef MMM1D_GPU

#include <cuda.h>
#include <cuda_runtime.h>

// the following works around an incompatibility between Eigen and the nvcc preprocessor
#define EIGEN_DONT_VECTORIZE
#include "VectorForce.hpp"

#ifdef ELECTROSTATICS_GPU_DOUBLE_PRECISION
typedef double mmm1dgpu_real;
#else
typedef float mmm1dgpu_real;
#endif

#ifdef __cplusplus
extern "C" {
#endif
int mmm1dgpu_devcount();
void mmm1dgpu_init();
int mmm1dgpu_set_params(mmm1dgpu_real boxz = 0, mmm1dgpu_real coulomb_prefactor = 0, mmm1dgpu_real maxPWerror = -1, mmm1dgpu_real far_switch_radius = -1, int bessel_cutoff = -1);
int mmm1dgpu_tune(const mmm1dgpu_real* r, const mmm1dgpu_real* q, int N, mmm1dgpu_real maxPWerror, mmm1dgpu_real far_switch_radius = -1, int bessel_cutoff = -1);
long long mmm1dgpu_forces(const mmm1dgpu_real *r, const mmm1dgpu_real *q, mmm1dgpu_real *force, int N, int pairs = 0);
long long mmm1dgpu_energies(const mmm1dgpu_real *r, const mmm1dgpu_real *q, mmm1dgpu_real *energy, int N, int pairs = 0);
#if 0
int besseltest(mmm1dgpu_real *a, mmm1dgpu_real *b0, mmm1dgpu_real *b1, int N);
int modpsitest(int order, mmm1dgpu_real *a, mmm1dgpu_real *b, int N);
void mmm1dgpu_cpp_forces(mmm1dgpu_real *r, mmm1dgpu_real *q, mmm1dgpu_real *force, int N);
#endif
#ifdef __cplusplus
}
#endif

class Mmm1dgpuForce : public VectorForce {
public:
  Mmm1dgpuForce(mmm1dgpu_real coulomb_prefactor, mmm1dgpu_real maxPWerror, mmm1dgpu_real far_switch_radius = -1, int bessel_cutoff = -1);
  ~Mmm1dgpuForce();
  void init(SystemInterface &s);
  void run(SystemInterface &s);
  bool isReady();
  void destroy();
private:
	int pairs;
	cudaStream_t *stream;
	cudaEvent_t *eventStart, *eventStop;
	mmm1dgpu_real **dev_r, **dev_q, **dev_force;
	mmm1dgpu_real **forcePartial;

	int numThreads;
	int numBlocks;

	mmm1dgpu_real *r, *q, *force;
	int N;

	mmm1dgpu_real coulomb_prefactor, maxPWerror, far_switch_radius;
	int bessel_cutoff;

	int initialized;
};

#endif /* MMM1D_GPU */
#endif /* MMM1DGPUFORCE_H */
