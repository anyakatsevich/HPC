XX = g++ # define a variable CXX
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

TARGETS = $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c))

# default first rule
all : $(TARGETS)
#
# # match all targets % such that there is a source file %.cpp
# # Automatic variables: $< (first dependency), $^ (dependency list), $@ (target)
%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< -o $@

%:%.c *.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS)


.PHONY: all, clean
