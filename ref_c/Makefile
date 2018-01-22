VPATH=./
EXEC=app_yolo_ref_c
OBJDIR=./obj/

ifeq ($(shell uname -m),x86_64)
	HOST_ARCH = x86_64
endif
ifeq ($(shell uname -m),aarch64)
	HOST_ARCH = arm
endif

CC=gcc
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS= -lm 
COMMON= 
ifeq ($(HOST_ARCH),x86_64)
CFLAGS=-Wall -Wfatal-errors -ftree-vectorize
endif
ifeq ($(HOST_ARCH),arm)
CFLAGS=-Wall -Wfatal-errors -ftree-vectorize
endif
ifeq ($(NEON_INTRINSICS),1)
OPTS=-O3 -DUSE_ARM_NEON_OPT=1 #-g
else
OPTS=-O3 #-g
endif

CFLAGS+=$(OPTS)

OBJ=main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(OBJS)
	 $(CC) `pkg-config --cflags opencv` $(COMMON) $(CFLAGS) $^ `pkg-config --libs opencv` -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

