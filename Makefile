include common.mk

#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# Includes
INC += -I$(MGPU_DIR) -I$(CUB_DIR) -I$(BOOST_DIR) -I$(GRB_DIR)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

all: gbfs gdiameter gsssp glgc gmis ggc ggc_cusparse gpr

gbfs: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gbfs example/gbfs.cu $(INC) $(GRB_DEPS) $(LIBS)

gdiameter: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gdiameter example/gdiameter.cu $(INC) $(GRB_DEPS) $(LIBS)

gsssp: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gsssp example/gsssp.cu $(INC) $(GRB_DEPS) $(LIBS)

glgc: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/glgc example/glgc.cu $(INC) $(GRB_DEPS) $(LIBS)

gmis: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gmis example/gmis.cu $(INC) $(GRB_DEPS) $(LIBS)

ggc: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/ggc example/ggc.cu $(INC) $(GRB_DEPS) $(LIBS)

ggc_cusparse: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/ggc_cusparse example/ggc_cusparse.cu $(INC) $(GRB_DEPS) $(LIBS)

gpr: example/*
	mkdir -p bin
	nvcc -g $(ARCH) $(OPTIONS) -o bin/gpr example/gpr.cu $(INC) $(GRB_DEPS) $(LIBS)

clean:
	rm -f bin/gbfs bin/gdiameter bin/gsssp bin/glgc bin/gmis bin/ggc bin/ggc_cusparse bin/gpr

lint:
	scripts/lint.py graphblas cpp $(GRB_DIR)example $(GRB_DIR)graphblas $(GRB_DIR)test --exclude_path $(GRB_DIR)graphblas/backend/sequential
