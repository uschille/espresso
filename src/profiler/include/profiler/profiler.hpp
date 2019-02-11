#ifndef PROFILER_PROFILER_HPP
#define PROFILER_PROFILER_HPP

#ifdef HAVE_CALIPER
#include <caliper/cali.h>

#define ESPRESSO_PROFILER_CXX_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define ESPRESSO_PROFILER_CXX_MARK_LOOP_BEGIN CALI_CXX_MARK_LOOP_BEGIN
#define ESPRESSO_PROFILER_CXX_MARK_LOOP_END CALI_CXX_MARK_LOOP_END
#define ESPRESSO_PROFILER_CXX_MARK_LOOP_ITERATION CALI_CXX_MARK_LOOP_ITERATION
#define ESPRESSO_PROFILER_MARK_FUNCTION_BEGIN CALI_MARK_FUNCTION_BEGIN
#define ESPRESSO_PROFILER_MARK_FUNCTION_END CALI_MARK_FUNCTION_END
#define ESPRESSO_PROFILER_MARK_LOOP_BEGIN CALI_MARK_LOOP_BEGIN
#define ESPRESSO_PROFILER_MARK_LOOP_END CALI_MARK_LOOP_END
#define ESPRESSO_PROFILER_MARK_ITERATION_BEGIN CALI_MARK_ITERATION_BEGIN
#define ESPRESSO_PROFILER_MARK_ITERATION_END CALI_MARK_ITERATION_END
#define ESPRESSO_PROFILER_WRAP_STATEMENT CALI_WRAP_STATEMENT
#define ESPRESSO_PROFILER_MARK_BEGIN CALI_MARK_BEGIN
#define ESPRESSO_PROFILER_MARK_END CALI_MARK_END
#else
#define ESPRESSO_PROFILER_CXX_MARK_FUNCTION
#define ESPRESSO_PROFILER_CXX_MARK_LOOP_BEGIN
#define ESPRESSO_PROFILER_CXX_MARK_LOOP_END
#define ESPRESSO_PROFILER_CXX_MARK_LOOP_ITERATION
#define ESPRESSO_PROFILER_MARK_FUNCTION_BEGIN
#define ESPRESSO_PROFILER_MARK_FUNCTION_END
#define ESPRESSO_PROFILER_MARK_LOOP_BEGIN
#define ESPRESSO_PROFILER_MARK_LOOP_END
#define ESPRESSO_PROFILER_MARK_ITERATION_BEGIN
#define ESPRESSO_PROFILER_MARK_ITERATION_END
#define ESPRESSO_PROFILER_WRAP_STATEMENT
#define ESPRESSO_PROFILER_MARK_BEGIN
#define ESPRESSO_PROFILER_MARK_END
#endif

#endif
