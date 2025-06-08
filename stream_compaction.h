#ifndef STREAM_COMPACTION_H
#define STREAM_COMPACTION_H

extern "C" void cuda_stream_compaction(float *h_in, float **h_out, unsigned int size, unsigned int *new_size);

#endif // STREAM_COMPACTION_H
