__device__ inline void atomicadd(float* address, float value)
{
#if !defined __CUDA_ARCH__ || __CUDA_ARCH__ >= 200
  atomicAdd(address, value);
#elif __CUDA_ARCH__ >= 110
	int oldval, newval, readback;
	oldval = __float_as_int(*address);
	newval = __float_as_int(__int_as_float(oldval) + value);
	while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval)
	{
		oldval = readback;
		newval = __float_as_int(__int_as_float(oldval) + value);
	}
#else
#error atomicAdd needs compute capability 1.1 or higher
#endif
}

__device__ inline void atomicadd8(double* address, double value)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
	unsigned long long oldval, newval, readback;
	oldval = __double_as_longlong(*address);
	newval = __double_as_longlong(__longlong_as_double(oldval) + value);
	while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
	{
		oldval = readback;
		newval = __double_as_longlong(__longlong_as_double(oldval) + value);
	}
#else
	atomicadd((float*) address, (float) value);
#endif
}