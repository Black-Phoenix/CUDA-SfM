set(SOURCE_FILES
    "common.h"
    "common.cu"
    "kernels.h"
    "kernels.cu"
    "svd.h"
    )

cuda_add_library(SfM
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
