def two_sum(nums, target):
    # Create a dictionary to store the complement of each element
    num_dict = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        # Check if the complement exists in the dictionary
        if complement in num_dict:
            # If found, return the indices of the two numbers
            return [num_dict[complement], i]
        
        # Otherwise, store the current number in the dictionary
        num_dict[num] = i
    
    # If no solution is found, return an empty list or handle it as needed
    return []

# Example usage:
nums = [2, 7, 11, 15]
target = 9
result = two_sum(nums, target)
print(result)  # Output should be [0, 1] (since nums[0] + nums[1] == 9)

def majority_element(nums):
    # Initialize variables to store the candidate and its count
    candidate = None
    count = 0
    
    # Iterate through the array
    for num in nums:
        # If count is 0, set the current element as the candidate
        if count == 0:
            candidate = num
        
        # If the current element is the same as the candidate, increment count; otherwise, decrement count
        if num == candidate:
            count += 1
        else:
            count -= 1
    
    # At the end, the candidate should be the majority element
    return candidate

# Example usage:
nums = [3, 3, 4, 2, 4, 4, 2, 4, 4]
result = majority_element(nums)
print(result)  # Output should be 4 (as 4 appears more than n/2 times)

def product_except_self(nums):
    n = len(nums)
    
    # Initialize two arrays to store products to the left and right of each element
    left_products = [1] * n
    right_products = [1] * n
    
    # Calculate the products to the left of each element
    left_product = 1
    for i in range(n):
        left_products[i] = left_product
        left_product *= nums[i]
    
    # Calculate the products to the right of each element
    right_product = 1
    for i in range(n - 1, -1, -1):
        right_products[i] = right_product
        right_product *= nums[i]
    
    # Multiply the left and right products for the final result
    output = [left_products[i] * right_products[i] for i in range(n)]
    
    return output

# Example usage:
nums = [1, 2, 3, 4]
result = product_except_self(nums)
print(result)  # Output should be [24, 12, 8, 6]

def max_subarray(nums):
    max_sum = nums[0]  # Initialize the maximum sum with the first element
    current_sum = nums[0]  # Initialize the current sum with the first element
    
    for num in nums[1:]:
        # Calculate the maximum of the current number and the current number plus the current sum
        current_sum = max(num, current_sum + num)
        # Update the maximum sum if the current sum is greater
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Example usage:
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result = max_subarray(nums)
print(result)  # Output should be 6 (the subarray [4, -1, 2, 1] has the largest sum)

def right_rotate(nums, k):
    k = k % len(nums)  # Handle cases where k is greater than the array length
    rotated = nums[-k:] + nums[:-k]
    return rotated

# Example usage:
nums = [1, 2, 3, 4, 5]
k = 2
result = right_rotate(nums, k)
print(result)  # Output should be [4, 5, 1, 2, 3]

def left_rotate(nums, k):
    k = k % len(nums)  # Handle cases where k is greater than the array length
    rotated = nums[k:] + nums[:k]
    return rotated

# Example usage:
nums = [1, 2, 3, 4, 5]
k = 2
result = left_rotate(nums, k)
print(result)  # Output should be [3, 4, 5, 1, 2]

