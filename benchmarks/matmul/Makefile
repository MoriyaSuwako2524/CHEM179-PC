# Build the three variants of the plain-C matmul library (Linux / macOS).
# macOS: Apple clang has no -march=native on arm64 -> use `make CC=clang NATIVE=-mcpu=native`
CC ?= gcc
NATIVE ?= -march=native
EXT ?= so

all: libmm_O2.$(EXT) libmm_O3native.$(EXT) libmm_O3native512.$(EXT)

libmm_O2.$(EXT): matmul.c
	$(CC) -O2 -fPIC -shared $< -o $@

libmm_O3native.$(EXT): matmul.c
	$(CC) -O3 $(NATIVE) -fPIC -shared $< -o $@

# GCC prefers 256-bit vectors on Skylake-SP even with AVX-512; this one asks for 512-bit
libmm_O3native512.$(EXT): matmul.c
	$(CC) -O3 $(NATIVE) -mprefer-vector-width=512 -fPIC -shared $< -o $@

clean:
	rm -f libmm_*.$(EXT)
