import tensorflow as tf

# Load the custom operator library
magma_lib = tf.load_op_library('./magma_cholesky.so')

def pad_matrices(matrices,n):
    """
    Pads matrices to the nearest multiple of 16 using tf.pad.
    
    Returns:
    - Padded TensorFlow tensor with width adjusted to the nearest multiple of 16.
    """
    if (n % 16 == 0):
        return matrices, False
    
    padded_n = (n + 15) & ~15  # Round up to the nearest multiple of 16
    if isinstance(matrices, list):
        matrices = [pad_matrix(matrix,padded_n - n) for matrix in matrices]
        return matrices, True
    else:
        padded_matrix = pad_matrix(matrices,padded_n - n)
        return padded_matrix, True


def pad_matrix(matrix, pad_size):

    paddings = tf.constant([[0,0], [0, pad_size]], dtype=tf.int32)

    padded_matrix = tf.pad(matrix, paddings, mode='CONSTANT', constant_values=0)

    return padded_matrix


def M_cholesky(input_tensor,pad=False):
    """
    Applies the custom Cholesky decomposition operator on the input tensor.

    Args:
        input_tensor (tf.Tensor): The input matrix or array of matrices to decompose.

    Returns:
        tf.Tensor: The Cholesky decomposition of the input tensor.
    """
    n = len(input_tensor[1])
    changed = False
    if pad:
        input_tensor, changed = pad_matrices(input_tensor,n)
   
    result = magma_lib.magma_cholesky(input_tensor,n)

#   if pad and changed:
#        if len(result.shape) == 3:
#            result = tf.slice(result, [0, 0, 0], [result.shape[0], n, n])
#        else:
#            result = tf.slice(result, [0, 0], [n, n])

    return result

