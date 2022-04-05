// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

compile-test:
	mkdir -p ../cuda_bin
	nvcc --fatbin -arch=compute_37 -code=compute_37 -code=sm_37 -code=sm_50 -code=sm_60 -code=sm_70 test_build.cu -o ../cuda_bin/test_build.fatbin
	nvcc --fatbin -arch=compute_37 -code=compute_37 -code=sm_37 -code=sm_50 -code=sm_60 -code=sm_70 test_build_multiblocks.cu -o ../cuda_bin/test_build_multiblocks.fatbin
	pytest ../../tests

