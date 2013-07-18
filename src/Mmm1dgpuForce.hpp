#ifndef MMM1DGPUFORCE_H
#define MMM1DGPUFORCE_H

#include "config.hpp"
#ifdef MMM1D_GPU

#include <cuda.h>
#include <cuda_runtime.h>

// the following works around an incompatibility between Eigen and the nvcc preprocessor
#define EIGEN_DONT_VECTORIZE
#include "VectorForce.hpp"

typedef double real; // TODO this shouldn't be in the header file

#ifdef __cplusplus
extern "C" {
#endif
int mmm1dgpu_devcount();
void mmm1dgpu_init();
int mmm1dgpu_set_params(real boxz = 0, real coulomb_prefactor = 0, real maxPWerror = -1, real far_switch_radius = -1, int bessel_cutoff = -1);
int mmm1dgpu_tune(const real* r, const real* q, int N, real maxPWerror, real far_switch_radius = -1, int bessel_cutoff = -1);
long long mmm1dgpu_forces(const real *r, const real *q, real *force, int N, int pairs = 0);
long long mmm1dgpu_energies(const real *r, const real *q, real *energy, int N, int pairs = 0);
#if 0
int besseltest(real *a, real *b0, real *b1, int N);
int modpsitest(int order, real *a, real *b, int N);
void mmm1dgpu_cpp_forces(real *r, real *q, real *force, int N);
#endif
#ifdef __cplusplus
}
#endif

class Mmm1dgpuForce : public VectorForce {
public:
  Mmm1dgpuForce(real coulomb_prefactor, real maxPWerror, real far_switch_radius = -1, int bessel_cutoff = -1);
  ~Mmm1dgpuForce();
  void init(SystemInterface &s);
  void run(SystemInterface &s);
  bool isReady();
  void destroy();
private:
	int pairs;
	cudaStream_t *stream;
	cudaEvent_t *eventStart, *eventStop;
	real **dev_r, **dev_q, **dev_force;
	real **forcePartial;

	int numThreads;
	int numBlocks;

	real *r, *q, *force;
	int N;

	real coulomb_prefactor, maxPWerror, far_switch_radius;
	int bessel_cutoff;

	int initialized;
};

#endif /* MMM1D_GPU */
#endif /* MMM1DGPUFORCE_H */
