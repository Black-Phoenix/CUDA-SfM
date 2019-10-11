set(SOURCE_FILES
    "common.h"
    "common.cu"
    "kernels.h"
    "sfm.cu"
    "sfm.h"
    "svd.h"
    )

cuda_add_library(SfM
    ${SOURCE_FILES}
    OPTIONS -arch=sm_61
    )
