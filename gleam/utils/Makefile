ifeq ($(UNAME),Darwin)
  PYCONF="python-config"
else
	PYCONF="python2-config"
endif

INCLUDE_FLAGS := $(shell ${PYCONF} --includes)
NUMPY_INCLUDE := -I$(shell python -c "import numpy; print(numpy.get_include())")
LIBRARY_FLAGS := $(shell ${PYCONF} --ldflags)
CFLAGS := -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -arch x86_64

all: optimized.c optimized.so

optimized.c:
	cython optimized.pyx

optimized.so:
	clang $(LIBRARY_FLAGS) $(CFLAGS) $(INCLUDE_FLAGS) $(NUMPY_INCLUDE) -o optimized.so optimized.c

clean:
	rm optimized.c optimized.so

