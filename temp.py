import torch
import cupy as cp

BASE = 2**16

def float_to_binary(array):
    sign_array = cp.signbit(array)
    array = cp.abs(array).view(cp.uint32)
    exponent_array = (((array & 0b01111111100000000000000000000000) >> 23) - 127).astype(cp.uint8) # TODO: interpret as uint or int?
    mantissa_array = (array & 0b00000000011111111111111111111111) | 0b00000000100000000000000000000000 # OR for implicit bit
    return sign_array, exponent_array, mantissa_array

# tensor = torch.rand((256, 128, 3, 3), dtype=torch.float32)
tensor = torch.tensor([[0.1, 0.249, 0.3], [0.4, 0.5, 0.6]])
weights = cp.asarray(tensor)
signs, exponents, mantissas = float_to_binary(weights)
print(f'**Signs**\n{signs.shape}\n{signs.dtype}\n{signs}\n\n**Exponents**\n{exponents.shape}\n{exponents.dtype}\n{exponents}\n\n**Mantissas**\n{mantissas.shape}\n{mantissas.dtype}\n{mantissas}\n\n')

def tensor_to_bignum(array):
    remainders = []
    while cp.any(array != 0):
        remainders.append(array % BASE)
        array = array // BASE
    return cp.stack(remainders, axis=-1)

bignum_mantissas = tensor_to_bignum(mantissas)
print(f'**Bignum Mantissas**\n{bignum_mantissas.shape}\n{bignum_mantissas.dtype}\n{bignum_mantissas}\n\n')

def int_to_bignum(val):
    result = []
    while val != 0:
        val, remainder = divmod(val, BASE)
        result.append(remainder)
    return cp.array(result, dtype=cp.uint32)

n = int_to_bignum(2**3072-1)
print(f'**Bignum Public Key**\n{n.shape}\n{n.dtype}\n{n}\n\n')

def bignum_multiply(a, b):
    if a.shape[-1] > b.shape[-1]:
        product = cp.expand_dims(a, axis=-2) * cp.expand_dims(b, axis=-1)
    else:
        product = cp.expand_dims(a, axis=-1) * cp.expand_dims(b, axis=-2)
    while cp.any(product >= BASE):
        # print(f'**Product**\n{product.shape}\n{product.dtype}\n{product}\n\n')
        carries = cp.concatenate([cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32), product // BASE], axis=-1)
        # print(f'**Carries**\n{carries.shape}\n{carries.dtype}\n{carries}\n\n')
        results = cp.concatenate([product % BASE, cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32)], axis=-1)
        # print(f'**Result**\n{results.shape}\n{results.dtype}\n{results}\n\n')
        product = carries + results
        # print(f'------------------------------------\n')
    # print(f'**Product**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    product = cp.stack([cp.trace(cp.flip(product, axis=-1), axis1=-1, axis2=-2, offset=offset) for offset in range(-product.shape[-1]+1, product.shape[-2])], axis=-1)
    # print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    while cp.any(product >= BASE):
        # print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
        carries = cp.concatenate([cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32), product // BASE], axis=-1)
        # print(f'**Carries**\n{carries.shape}\n{carries.dtype}\n{carries}\n\n')
        results = cp.concatenate([product % BASE, cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32)], axis=-1)
        # print(f'**Result**\n{results.shape}\n{results.dtype}\n{results}\n\n')
        product = carries + results
        # print(f'------------------------------------\n')
    # print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    while cp.all(product[...,-1:] == 0) and product.shape[-1] > 1:
        product, _ = cp.split(product, [-1], axis=-1)
        # print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    return product

# product = bignum_multiply(bignum_mantissas, n)
# print(f'**Product**\n{product.shape}\n{product.dtype}\n{product}\n\n')

def bignum_add(a, b):
    if a.shape[-1] < b.shape[-1]:
        a = cp.concatenate([a, cp.zeros(b.shape[-1] - a.shape[-1], dtype=cp.uint32)], axis=-1)
    elif b.shape[-1] < a.shape[-1]:
        b = cp.concatenate([b, cp.zeros(a.shape[-1] - b.shape[-1], dtype=cp.uint32)], axis=-1)

    summed = a + b
    while cp.any(summed >= BASE):
        carries = cp.concatenate([cp.zeros(summed.shape[:-1] + (1,), dtype=cp.uint32), summed // BASE], axis=-1)
        results = cp.concatenate([summed % BASE, cp.zeros(summed.shape[:-1] + (1,), dtype=cp.uint32)], axis=-1)
        summed = carries + results

    if cp.all(summed[...,-1:] == 0):
        summed, _ = cp.split(summed, [-1], axis=-1)
    return summed



def bignum_floor_divide(a, b):
    """"
    Use polynomial long division to compute a = bq + r, i.e a/b
    Inputs:
        a: (*, max_degree)
        b: (degree)
    """
    q = cp.zeros((a.shape[:-1]))
    r = a
    d = b.shape[-1]
    c = b[-1]

def bignum_modulo(a, b):
    """"implements a % b for bignum tensors"""
    # return a//b * b - a
    pass