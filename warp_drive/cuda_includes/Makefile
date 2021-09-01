
compile-test:
	mkdir -p ../cuda_bin
	nvcc --cubin -arch=sm_70 test_build.cu -o ../cuda_bin/test_build.cubin
	pytest ../../tests

