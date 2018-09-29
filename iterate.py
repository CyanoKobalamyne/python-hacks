def generate_nd_array(dimensions, fill):
    """Generates an N-dimensional array (nested list).

    arguments:
        dimensions (list/tuple): the size of the array for each dimension,
                                 needs to be of length >= 1
        fill (object): the value to fill the (innermost) array with

    returns:
        list: the generated array

    Implementation note: this function is iterative and doesn't use recursion.
    """
    array = [] # Nested list we are building.
    stack = [] # Keeps track of upper-dimensional arrays inside the loop.
    level = 0 # Current index into the dimensions list, how deep down we are.
    while True:
        if level + 1 == len(dimensions):
            # Innermost list, fill up with the given value.
            array.extend(fill for _ in range(dimensions[level]))
        if len(array) < dimensions[level]:
            # This level of the array is not filled up yet, add new sublist.
            subarray = []
            array.append(subarray)
            stack.append(array)
            array = subarray
            level += 1
        elif stack:
            # This level is done but there are other levels above us.
            array = stack.pop()
            level -= 1
        else:
            # Uppermost level is done, so array is done.
            break
    return array
