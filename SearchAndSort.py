# Collection of Top 10 DSA Code Snippets for Searching and Sorting
# Use for daily practice in coding challenges (e.g., LeetCode, HackerRank)

# 1. Binary Search
# Overview: Efficiently finds an element in a sorted array by dividing the search interval in half. Time: O(log n), Space: O(1).
# Use Cases: Searching in sorted logs, finding insertion points in databases, or challenges like "Search in Rotated Sorted Array".
# Why It's Useful: Core for any sorted data problem; variants handle edge cases like duplicates or rotations.
def binary_search(arr,target):
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

# Example usage for testing (uncomment to run):

if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    print("Binary Search:", binary_search([11, 12, 22, 25, 34, 64, 90], 25))  # Output: 3
    print("QuickSort:", quicksort(arr.copy()))  # Output: [11, 12, 22, 25, 34, 64, 90]
    print("MergeSort:", merge_sort(arr.copy()))
    print("HeapSort:", heap_sort(arr.copy()))
    print("Insertion Sort:", insertion_sort(arr.copy()))
    print("Linear Search:", linear_search(arr.copy(), 22))  # Output: 4
    print("Selection Sort:", selection_sort(arr.copy()))
    print("Bubble Sort:", bubble_sort(arr.copy()))
    print("Radix Sort:", radix_sort([170, 45, 75, 90, 802, 24, 2, 66]))
    print("Bucket Sort:", bucket_sort([0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21]))
