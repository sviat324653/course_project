import multivex
import array
import timeit
import numpy as np
import time


print(f"Successfully imported multivex library: {multivex}")
print(f"Functions available: {dir(multivex)}")

print("\n--- Testing scan ---")

help(multivex.scan)

input_list = [10.0, 20.0, 5.0, 15.0]
n_list = len(input_list)
print(f"Input list: {input_list}")

try:
    output_list = multivex.scan(input_list)
    print(f"Output from list: {output_list}")
except Exception as e:
    print(f"Error calling scan with list: {e}")

input_arr = array.array('f', [1.5, 2.5, 3.5])
n_arr = len(input_arr)
print(f"\nInput array.array: {list(input_arr)}")

try:
    output_arr = multivex.scan(input_arr)
    print(f"Output from array.array: {output_arr}")
except Exception as e:
    print(f"Error calling scan with array.array: {e}")


np_input = np.random.uniform(0.0, 100.0, size=(1 << 29)).astype(np.float32)
np_input1 = np.array([1.5, 2.5, 3.5, 4.0]).astype(np.float32)
print("\nInput NumPy array: ")
benchmarked = lambda: multivex.scan(np_input1)
tim = timeit.timeit(benchmarked, number=1)
print("elapsed time for multivex.scan: ", tim * 1000, "milliseconds")
output_np_mv1 = multivex.scan(np_input)
start = time.time()
output_np_mv2 = np.add.accumulate(np_input)
end = time.time()
print(f"elapsed time for numpy.add.accumulate: {end - start:.6f} seconds")

print(f"Output from NumPy array: {output_np_mv1[0:10]} ..... {output_np_mv1[-10:-1]}")
print(f"Output from NumPy array: {output_np_mv2[0:9]} ..... {output_np_mv1[-10:-1]}")


print("\n--- Testing Edge Cases ---")
try:
    print("Empty list input for scan:", multivex.scan([]))
    print("Empty array.array input for scan:", multivex.scan(array.array('f')))
    print("Empty numpy.ndarray input for scan:", multivex.scan(np.empty(0, dtype=np.float32)))

except Exception as e:
    print(f"Error with empty test: {e}")

try:
    multivex.scan([1.0, "not a float", 3.0])
except (TypeError, ValueError) as e:
    print(f"Caught expected error for bad list element: {e}")

try:
    bad_array = array.array('i', [1,2,3])
    multivex.scan(bad_array)
except TypeError as e:
    print(f"Caught expected error for bad array.array type: {e}")

try:
    np_int_input = np.array([1, 2, 3], dtype=np.int32)
    multivex.scan(np_int_input)
except TypeError as e:
    print(f"Caught expected error for bad NumPy dtype: {e}")
