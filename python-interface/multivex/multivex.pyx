cimport multivex.multivex as multivex
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport uintptr_t
import array
cimport numpy as np
import numpy as np


np.import_array()


def scan(object input_data):
    if isinstance(input_data, list):
        return scan_list(input_data)

    elif isinstance(input_data, array.array):
        return scan_array(input_data)

    elif isinstance(input_data, np.ndarray):
        return scan_numpy(input_data)

    else:
        raise TypeError("Unsupported type")

scan.__doc__ = """
    Perform an exclusive scan (prefix sum) on the input data.

    Parameters:
    input_data (list, array.array, or numpy.ndarray) 

    Returns:
    The result of the scan is new object of same type as the input.
    """


cdef np.ndarray scan_numpy(np.ndarray input_np_array):
    cdef unsigned int n = np.PyArray_SIZE(input_np_array)

    if input_np_array.dtype != np.float32:
        raise TypeError("Input elements of numpy.ndarray must be of type 'numpy.float32'")

    if n == 0:
        return np.empty(0, dtype=np.float32)
    
    cdef np.ndarray[np.float32_t, ndim=1] c_output_np_array = None

    cdef float* c_h_in = <float*>np.PyArray_DATA(input_np_array)
    cdef float* c_h_out = NULL

    c_output_np_array = np.empty(n, dtype=np.float32)
    c_h_out = <float*>np.PyArray_DATA(c_output_np_array)
    
    multivex.cuda_scan3(c_h_in, c_h_out, n)

    return c_output_np_array


cdef list scan_list(list input_list):
    cdef unsigned int n = len(input_list)
    cdef float* c_h_in = NULL
    cdef float* c_h_out = NULL
    cdef int i
    cdef list output_list

    if n == 0:
        return []

    c_h_in = <float*>PyMem_Malloc(n * sizeof(float))
    if not c_h_in:
        raise MemoryError("Failed to allocate memory for input C array")
    for i in range(n):
        try:
            c_h_in[i] = float(input_list[i])
        except (TypeError, ValueError) as e:
            PyMem_Free(c_h_in)
            raise type(e)(f"Error converting list element {input_list[i]} to float: {e}")

    c_h_out = <float*>PyMem_Malloc(n * sizeof(float))
    if not c_h_out:
        raise MemoryError("Failed to allocate memory for output C array")

    multivex.cuda_scan3(c_h_in, c_h_out, n)

    output_list = [c_h_out[i] for i in range(n)]

    if c_h_in is not NULL:
        PyMem_Free(c_h_in)
    if c_h_out is not NULL:
        PyMem_Free(c_h_out)

    return output_list


cdef scan_array(object input_array):
    cdef unsigned int n = len(input_array)
    cdef float* c_h_in = NULL
    cdef float* c_h_out = NULL
    cdef object c_output_array
    cdef uintptr_t address

    if n == 0:
        return array.array('f')

    if input_array.typecode != 'f':
        raise TypeError("Input array.array must be of type 'f' (float)")
    
    address = input_array.buffer_info()[0]
    c_h_in = <float*>address

    c_output_array = array.array('f', [0] * n)

    address = c_output_array.buffer_info()[0]
    c_h_out = <float*>address

    multivex.cuda_scan3(c_h_in, c_h_out, n)

    return c_output_array



def reduce(object input_data):
    # if isinstance(input_data, list):
    #     return scan_list(input_data)
    #
    # elif isinstance(input_data, array.array):
    #     return scan_array(input_data)

    if isinstance(input_data, np.ndarray):
        return reduce_numpy(input_data)

    else:
        raise TypeError("Unsupported type")


cdef np.ndarray reduce_numpy(np.ndarray input_np_array):
    cdef unsigned int n = np.PyArray_SIZE(input_np_array)

    if input_np_array.dtype != np.float32:
        raise TypeError("Input elements of numpy.ndarray must be of type 'numpy.float32'")

    if n == 0:
        return np.empty(0, dtype=np.float32)
    
    cdef np.ndarray[np.float32_t, ndim=1] c_output_np_array = None

    cdef float* c_h_in = <float*>np.PyArray_DATA(input_np_array)
    cdef float* c_h_out = NULL

    c_output_np_array = np.empty(1, dtype=np.float32)
    c_h_out = <float*>np.PyArray_DATA(c_output_np_array)
    
    multivex.cuda_reduction3(c_h_in, c_h_out, n)

    return c_output_np_array



def stream_compaction(object input_data):
    # if isinstance(input_data, list):
    #     return scan_list(input_data)
    #
    # elif isinstance(input_data, array.array):
    #     return scan_array(input_data)

    if isinstance(input_data, np.ndarray):
        return stream_compaction_numpy(input_data)

    else:
        raise TypeError("Unsupported type")


cdef np.ndarray stream_compaction_numpy(np.ndarray input_np_array):
    cdef unsigned int n = np.PyArray_SIZE(input_np_array)

    if input_np_array.dtype != np.float32:
        raise TypeError("Input elements of numpy.ndarray must be of type 'numpy.float32'")

    if n == 0:
        return np.empty(0, dtype=np.float32)
    
    cdef float* c_h_in = <float*>np.PyArray_DATA(input_np_array)
    cdef float* c_h_out = NULL
    cdef unsigned int out_size = 0

    
    multivex.cuda_stream_compaction(c_h_in, &c_h_out, n, &out_size)

    cdef np.npy_intp dims[1]
    dims[0] = out_size

    cdef np.ndarray c_output_np_array = np.PyArray_SimpleNewFromData(
        1,
        dims,
        np.NPY_FLOAT,
        <void*>c_h_out
    )

    return c_output_np_array



def sort1(object input_data):
    # if isinstance(input_data, list):
    #     return scan_list(input_data)
    #
    # elif isinstance(input_data, array.array):
    #     return scan_array(input_data)

    if isinstance(input_data, np.ndarray):
        sort1_numpy(input_data)

    else:
        raise TypeError("Unsupported type")


cdef np.ndarray sort1_numpy(np.ndarray input_np_array):
    cdef unsigned int n = np.PyArray_SIZE(input_np_array)

    if input_np_array.dtype != np.int32:
        raise TypeError("Input elements of numpy.ndarray must be of type 'numpy.int32'")

    if n == 0:
        return np.empty(0, dtype=np.float32)
    
    cdef int* c_h_in = <int*>np.PyArray_DATA(input_np_array)
    
    multivex.bitonic_sort(c_h_in, n)

