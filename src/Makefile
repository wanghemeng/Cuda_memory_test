.PHONY: all
all: no_stream with_stream zero_copy unified_buff

no_stream: no_stream.cu
	nvcc no_stream.cu -o ../bin/no_stream

with_stream: with_stream.cu
	nvcc with_stream.cu -o ../bin/with_stream

zero_copy: zero_copy.cu
	nvcc zero_copy.cu -o ../bin/zero_copy

unified_buff: unified_buff.cu
	nvcc unified_buff.cu -o ../bin/unified_buff

run:
	../bin/no_stream
	../bin/with_stream
	../bin/zero_copy
	../bin/unified_buff

clean:
	rm -rf ../bin/*