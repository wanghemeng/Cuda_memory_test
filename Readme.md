# Cuda memory test

A series of tests for memory operation.

This project contains four different programs:

1. A ordinary cuda program. Contains memory load, kernel execution and memory store.

2. A stream optimized program. Use streams(default number is 2) to hide the memory operations.

3. Use zero-copy memory to test.

4. Use unified memory to test. (The result may because the data is moved to gpu and never changed. Add some cpu operations will obtain different results.)

## Implementation

To compile all the programs:

`mkdir bin`

`cd src`

`make`

To test the result:

`make run`

## Results

The result is listed below(on 3080 Laptop):

| Program | time |
| --- | ----------- |
| no stream | 61.27 + 69.79 + 60.58 = 191.64ms |
| with stream | 110.16ms |
| zero copy | 73.31ms |
| unified | 68.73ms |
