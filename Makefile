include common.mk

#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# Includes
INC += -I$(MGPU_DIR) -I$(CUB_DIR) -I$(GRB_DIR)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

all: gbfs gsssp glgc ggc

gbfs: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gbfs example/gbfs.cu ${INC} $(GRB_DEPS) $(LIBS)

gsssp: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gsssp example/gsssp.cu ${INC} $(GRB_DEPS) $(LIBS)

glgc: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/glgc example/glgc.cu ${INC} $(GRB_DEPS) $(LIBS)

ggc: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/ggc example/ggc.cu ${INC} $(GRB_DEPS) $(LIBS)

clean:
	rm -f bin/gbfs bin/gsssp bin/glgc bin/ggc

lint:
	scripts/lint.py graphblas cpp $(GRB_DIR)example $(GRB_DIR)graphblas $(GRB_DIR)test --exclude_path $(GRB_DIR)graphblas/backend/sequential
