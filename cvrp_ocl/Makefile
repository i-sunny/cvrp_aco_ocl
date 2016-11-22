# Makefile for cvrp_ocl
VERSION=1.0

OPTIM_FLAGS=-O2
WARN_FLAGS=-Wall -ansi -pedantic
CPPFLAGS=$(WARN_FLAGS) $(OPTIM_FLAGS) -std=gnu++11 -lOpenCL
CC=g++

# INC=-I/usr/local/cuda-7.5/include
VPATH = gpu
SUBDIR = gpu

# To change the default timer implementation, uncomment the line below
# or call 'make TIMER=unix'
TIMER=unix
LDLIBS=-lm

OBJS= $(SUBDIR)/g_type.o $(SUBDIR)/g_aco.o $(SUBDIR)/g_sa.o io.o localSearch.o main.o $(TIMER)_timer.o move.o neighbourSearch.o problem.o utilities.o vrpHelper.o OpenCLInfo.o
EXE=main


all: clean cvrp_ocl

clean:
	@$(RM) *.o */*.o $(EXE) cvrp_ocl

cvrp_ocl: $(OBJS)
	$(CC) $(CPPFLAGS) $(OBJS) -o $(EXE)

io.o: io.cpp io.h

localSearch.o: localSearch.cpp localSearch.h

main.o: main.cpp

$(TIMER)_timer.o: $(TIMER)_timer.cpp timer.h

move.o: move.cpp move.h

neighbourSearch.o: neighbourSearch.cpp neighbourSearch.h

problem.o: problem.cpp problem.h

utilities.o: utilities.cpp utilities.h

vrpHelper.o: vrpHelper.cpp vrpHelper.h

OpenCLInfo.o: OpenCLInfo.cpp OpenCLInfo.h

$(SUBDIR)/g_sa.o: 	g_sa.cpp g_sa.h
$(SUBDIR)/g_aco: 	g_aco.cpp g_aco.h
$(SUBDIR)/g_type: 	g_type.cpp g_type.h


dist : DIST_SRC_FILES=*.cpp *.h Makefile
dist : all
	@(mkdir -p ../cvrp_ocl-$(VERSION)			\
	&& rsync -rlpC --exclude=.svn $(DIST_SRC_FILES) ../cvrp_ocl-$(VERSION)/ \
        && cd .. 	\
	&& tar cf - cvrp_ocl-$(VERSION) | gzip -f9 > cvrp_ocl-$(VERSION).tar.gz \
	&& rm -rf ./cvrp_ocl-$(VERSION)					\
	&& echo "cvrp_ocl-$(VERSION).tar.gz created." && cd $(CWD) )
