import subprocess

from warp_drive.managers.function_manager import CUDAFunctionManager
from warp_drive.utils.common import get_project_root

if __name__ == "__main__":
    cuda_function_manager = CUDAFunctionManager()
    # main_file is the source code
    main_file = f"{get_project_root()}/warp_drive/cuda_includes/test_build.cu"
    # cubin_file is the targeted compiled exe
    cubin_file = f"{get_project_root()}/warp_drive/cuda_bin/test_build.fatbin"
    print("Running Unittest ... ")
    print(f"Compiling {main_file} -> {cubin_file}")
    cuda_function_manager._compile(main_file, cubin_file)

    cmd = f"pytest {get_project_root()}/tests"
    test_process = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
    try:
        outs, errs = test_process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        test_process.kill()
        outs, errs = test_process.communicate()
        print("Unit Test Timeout")
