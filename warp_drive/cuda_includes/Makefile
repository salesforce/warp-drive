
compile-test:
	mkdir -p ../cuda_bin
	nvcc --fatbin -arch=compute_30 -code=sm_30 -code=sm_50 -code=sm_60 -code=sm_70 test_build.cu -o ../cuda_bin/test_build.fatbin
	pytest ../../tests

