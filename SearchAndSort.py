# Collection of Top 16 DSA Code Snippets for Searching and Sorting (Intermediate Level)
# Includes core algorithms plus advanced variants for sorting and binary search-based problems
# Use for daily practice in coding challenges (e.g., LeetCode, HackerRank)

# 1. Binary Search
# Overview: Efficiently finds an element in a sorted array by dividing the search interval in half. Time: O(log n), Space: O(1).
# Use Cases: Searching in sorted logs, finding insertion points in databases, or challenges like "Search in Rotated Sorted Array".
# Why It's Useful: Core for any sorted data problem; variants handle edge cases like duplicates or rotations.
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # Not found

# 2. QuickSort
# Overview: Divide-and-conquer sorting using a pivot to partition the array. Average Time: O(n log n), Worst: O(n²), Space: O(log n).
# Use Cases: Sorting large datasets in memory-constrained environments, or challenges like "Kth Largest Element in an Array".
# Why It's Useful: Fast in practice; understanding pivots helps with partition-based problems.
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 3. MergeSort
# Overview: Stable divide-and-conquer sort that splits and merges arrays. Time: O(n log n), Space: O(n).
# Use Cases: External sorting for big data (e.g., files too large for memory), or "Sort List" in linked lists.
# Why It's Useful: Guaranteed performance; merge step is key for inversion counts or multi-way merges.
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 4. HeapSort (Using Min-Heap)
# Overview: Builds a heap and extracts min elements for sorting. Time: O(n log n), Space: O(1).
# Use Cases: Priority queues in scheduling, or "Merge K Sorted Lists".
# Why It's Useful: In-place and efficient; heaps are foundational for top-K problems.
import heapq

def heap_sort(arr):
    heapq.heapify(arr)  # Build min-heap
    sorted_arr = []
    while arr:
        sorted_arr.append(heapq.heappop(arr))
    return sorted_arr

# 5. Insertion Sort
# Overview: Builds a sorted array by inserting elements one by one. Time: O(n²), Best: O(n), Space: O(1).
# Use Cases: Small or nearly sorted arrays (e.g., online sorting streams), or as part of TimSort hybrids.
# Why It's Useful: Simple for adaptive sorting; good for understanding comparisons.
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 6. Linear Search
# Overview: Sequentially checks each element. Time: O(n), Space: O(1).
# Use Cases: Unsorted small lists, or when data is streamed (e.g., finding max in array).
# Why It's Useful: Baseline for search; optimize to binary when sorted.
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1  # Not found

# 7. Selection Sort
# Overview: Repeatedly selects the minimum element and swaps it. Time: O(n²), Space: O(1).
# Use Cases: When swaps are costly (e.g., flash memory), or simple educational sorts.
# Why It's Useful: Teaches selection; useful in k-smallest element variants.
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# 8. Bubble Sort
# Overview: Repeatedly swaps adjacent elements if out of order. Time: O(n²), Best: O(n), Space: O(1).
# Use Cases: Detecting sorted arrays early, or small educational datasets.
# Why It's Useful: Optimized version teaches early termination; contrasts with efficient sorts.
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# 9. Radix Sort
# Overview: Sorts integers by digit places using counting sort. Time: O(d(n + k)) where d=digits, k=base, Space: O(n + k).
# Use Cases: Sorting fixed-length integers (e.g., phone numbers), or non-comparative sorts.
# Why It's Useful: Linear time for integers; beats comparison sorts in specific cases.
def radix_sort(arr):
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    for i in range(n):
        arr[i] = output[i]

# 10. Bucket Sort
# Overview: Distributes elements into buckets and sorts them individually. Time: Average O(n + k), Space: O(n + k).
# Use Cases: Uniformly distributed floats (e.g., graphics rendering), or hybrid with other sorts.
# Why It's Useful: Efficient for certain distributions; teaches non-comparison sorting strategies.
def bucket_sort(arr):
    if not arr:
        return arr
    min_val, max_val = min(arr), max(arr)
    bucket_count = len(arr)
    buckets = [[] for _ in range(bucket_count)]
    for num in arr:
        index = int(bucket_count * (num - min_val) / (max_val - min_val + 1e-9))
        buckets[index].append(num)
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))  # Or use insertion_sort for small buckets
    return sorted_arr

# 11. Counting Sort
# Overview: Non-comparative sort for integers in a known range, counting occurrences. Time: O(n + k) where k is range, Space: O(k).
# Use Cases: Sorting integers with small range (e.g., student scores), or as a subroutine in radix sort.
# Why It's Useful: Linear time for constrained inputs; foundational for radix sort and histogram-based problems.
def counting_sort(arr):
    if not arr:
        return arr
    max_val = max(arr)
    min_val = min(arr)
    range_val = max_val - min_val + 1
    count = [0] * range_val
    output = [0] * len(arr)
    
    # Count occurrences
    for num in arr:
        count[num - min_val] += 1
    
    # Cumulative count
    for i in range(1, range_val):
        count[i] += count[i - 1]
    
    # Place elements in sorted order
    for num in reversed(arr):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1
    
    return output

# 12. Binary Search for First/Last Occurrence
# Overview: Finds the first or last occurrence of a target in a sorted array with duplicates. Time: O(log n), Space: O(1).
# Use Cases: Problems like "Find First and Last Position of Element in Sorted Array" or range queries in databases.
# Why It's Useful: Extends binary search to handle duplicates; critical for boundary-related problems.
def binary_search_first_last(arr, target, find_first=True):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            if find_first:
                right = mid - 1  # Continue searching left for first occurrence
            else:
                left = mid + 1   # Continue searching right for last occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# 13. TimSort (Simplified Hybrid of MergeSort and Insertion Sort)
# Overview: Hybrid of merge sort and insertion sort, used in Python's sorted(). Time: O(n log n), Best: O(n), Space: O(n).
# Use Cases: General-purpose sorting in production (e.g., Python’s sorted()), or "Sort Characters By Frequency".
# Why It's Useful: Real-world relevance; teaches hybrid algorithms and adaptive sorting.
def timsort(arr):
    # Python's built-in sorted() uses TimSort; this is a simplified version
    # Use insertion sort for small runs (threshold ~32 in practice)
    RUN_SIZE = 32
    def insertion_sort_run(arr, start, end):
        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1
            while j >= start and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    # Split into runs and sort with insertion sort
    n = len(arr)
    for i in range(0, n, RUN_SIZE):
        insertion_sort_run(arr, i, min(i + RUN_SIZE - 1, n - 1))
    
    # Merge runs using merge function from merge_sort
    size = RUN_SIZE
    while size < n:
        for left in range(0, n, size * 2):
            mid = left + size - 1
            right = min(left + size * 2 - 1, n - 1)
            if mid < right:
                merged = merge(arr[left:mid + 1], arr[mid + 1:right + 1])
                arr[left:right + 1] = merged
        size *= 2
    return arr

# 14. Binary Search on Rotated Sorted Array
# Overview: Finds a target in a sorted array rotated at an unknown pivot. Time: O(log n), Space: O(1).
# Use Cases: "Search in Rotated Sorted Array" (LeetCode #33), handling sorted arrays with rotation.
# Why It's Useful: Common interview problem; tests adaptation of binary search.
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1  # Not found

# 15. In-Place QuickSort
# Overview: In-place version of QuickSort using partitioning. Average Time: O(n log n), Worst: O(n²), Space: O(log n).
# Use Cases: Memory-efficient sorting in constrained environments, or "Sort List" variants.
# Why It's Useful: Optimizes space compared to list-comprehension QuickSort; widely used in practice.
def quicksort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = partition(arr, low, high)
        quicksort_inplace(arr, low, pi - 1)
        quicksort_inplace(arr, pi + 1, high)
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# 16. Cycle Sort
# Overview: Sorts an array with minimal memory writes by finding cycles. Time: O(n²), Space: O(1).
# Use Cases: Problems requiring minimal writes like "First Missing Positive" (LeetCode #41), or duplicate detection.
# Why It's Useful: Niche for constraint-specific problems; teaches cycle-based sorting.
def cycle_sort(arr):
    n = len(arr)
    for cycle_start in range(n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1
        if pos == cycle_start:
            continue
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
    return arr

# Example usage for testing (uncomment to run):

if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    print("Binary Search:", binary_search([11, 12, 22, 25, 34, 64, 90], 25))  # Output: 3
    print("QuickSort:", quicksort(arr.copy()))
    print("MergeSort:", merge_sort(arr.copy()))
    print("HeapSort:", heap_sort(arr.copy()))
    print("Insertion Sort:", insertion_sort(arr.copy()))
    print("Linear Search:", linear_search(arr.copy(), 22))  # Output: 4
    print("Selection Sort:", selection_sort(arr.copy()))
    print("Bubble Sort:", bubble_sort(arr.copy()))
    print("Radix Sort:", radix_sort([170, 45, 75, 90, 802, 24, 2, 66]))
    print("Bucket Sort:", bucket_sort([0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21]))
    print("Counting Sort:", counting_sort([4, 2, 2, 8, 3, 3, 1]))
    print("Binary Search First:", binary_search_first_last([5, 7, 7, 8, 8, 10], 8, True))  # Output: 3
    print("Binary Search Last:", binary_search_first_last([5, 7, 7, 8, 8, 10], 8, False))  # Output: 4
    print("TimSort:", timsort(arr.copy()))
    print("Search Rotated:", search_rotated([4,5,6,7,0,1,2], 0))  # Output: 4
    print("In-Place QuickSort:", quicksort_inplace(arr.copy()))
    print("Cycle Sort:", cycle_sort([20, 40, 50, 10, 30]))
