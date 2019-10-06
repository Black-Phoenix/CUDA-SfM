#pragma once

#include "common.h"
#include <vector>
#include <curand.h>
#include <cublas_v2.h>
#include "svd.h"

using namespace std;
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
namespace Kernals {
	Common::PerformanceTimer& timer();
}
