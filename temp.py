import torch
import cupy as cp

BASE = 2**16

def float_to_binary(array):
    sign_array = cp.signbit(array)
    array = cp.abs(array).view(cp.uint32)
    exponent_array = (((array & 0b01111111100000000000000000000000) >> 23) - 127).astype(cp.uint8) # TODO: interpret as uint or int?
    mantissa_array = (array & 0b00000000011111111111111111111111) | 0b00000000100000000000000000000000 # OR for implicit bit
    return sign_array, exponent_array, mantissa_array

weights = cp.asarray(torch.tensor([[0.1, 0.249, 0.3], [0.4, 0.5, 0.6]]))
signs, exponents, mantissas = float_to_binary(weights)
print(f'**Signs**\n{signs.shape}\n{signs.dtype}\n{signs}\n\n**Exponents**\n{exponents.shape}\n{exponents.dtype}\n{exponents}\n\n**Mantissas**\n{mantissas.shape}\n{mantissas.dtype}\n{mantissas}\n\n')

def tensor_to_bignum(array):
    remainders = []
    while cp.any(array != 0):
        remainders.append(array % BASE)
        array = array // BASE
    return cp.stack(remainders, axis=2)

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
        print(f'**Product**\n{product.shape}\n{product.dtype}\n{product}\n\n')
        carries = cp.concatenate([cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32), product // BASE], axis=-1)
        print(f'**Carries**\n{carries.shape}\n{carries.dtype}\n{carries}\n\n')
        results = cp.concatenate([product % BASE, cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32)], axis=-1)
        print(f'**Result**\n{results.shape}\n{results.dtype}\n{results}\n\n')
        product = carries + results
        print(f'------------------------------------\n')
    print(f'**Product**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    product = cp.stack([cp.trace(cp.flip(product, axis=-1), axis1=-1, axis2=-2, offset=offset) for offset in range(-product.shape[-1]+1, product.shape[-2])], axis=-1)
    print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    while cp.any(product >= BASE):
        print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
        carries = cp.concatenate([cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32), product // BASE], axis=-1)
        print(f'**Carries**\n{carries.shape}\n{carries.dtype}\n{carries}\n\n')
        results = cp.concatenate([product % BASE, cp.zeros(product.shape[:-1] + (1,), dtype=cp.uint32)], axis=-1)
        print(f'**Result**\n{results.shape}\n{results.dtype}\n{results}\n\n')
        product = carries + results
        print(f'------------------------------------\n')
    print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    while cp.all(product[...,-1:] == 0):
        product, _ = cp.split(product, [-1], axis=-1)
        print(f'**Sum**\n{product.shape}\n{product.dtype}\n{product}\n\n')
    return product

product = bignum_multiply(bignum_mantissas, n)
print(f'**Product**\n{product.shape}\n{product.dtype}\n{product}\n\n')

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

def bignum_geq(a, b):
    """Returns a mask which is True for the elements of a which are >= b"""



    if a.shape[-1] > b.shape[-1]:
        return True
    elif a.shape[-1] < b.shape[-1]:
        return False
    else:

        t = cp.any(a[..., -1:] >= b[..., -1:])

def bignum_floor_divide(a, b):
    """"Use polynomial long division to compute a = bq + r, i.e a/b"""
    a_degree = a.shape[-1]
    b_degree = b.shape[-1]
    if a_degree < b_degree:
        raise ValueError(f'Degree of dividend cannot be less than degree of divisor')

def bignum_modulo(a, b):
    """"implements a % b for bignum tensors"""
    # return a//b * b - a
    pass

# a = torch.arange(0,8).view(2,2,2).to(torch.int32)
# b = torch.arange(0,8).view(1,8).to(torch.int32)

# mask = b == 0
# b = b + mask
# print(mask)

# print(a)
# print(a.view(a.numel(),1) % b * ~mask)

# res = bignum_multiply(a,b)
# print(res.size())
# print(res)




# t = torch.zeros((1), dtype=torch.int32)
# t[0] = 1036831949
# # t[0] = -1110651699
# print(t)
# print(t >> 31)


# 0b01000010001100110011001100110011 # -0.1
# 0b01000010001100110011001100110011
# 0b00111101110011001100110011001101 # 0.1
# 0b0 # 0.1 >> 32


# 0b01000010001100110011001100110011 # 1110651699
# 0b00111101110011001100110011001101 # -1110651699



# print(sign_tensor, sign_tensor.size(), bin(sign_tensor), len(bin(sign_tensor.item())) - 2)
# print(exponent_tensor, exponent_tensor.size(), bin(exponent_tensor), len(bin(exponent_tensor)) - 2)
# print(mantissa_tensor, mantissa_tensor.size(), bin(mantissa_tensor), len(bin(mantissa_tensor)) - 2)

# print(bin(torch.tensor([0b1])))

# torch.set_printoptions(precision=100)

# t32 = torch.tensor([0.1]).to(torch.float32)
# t64 = torch.tensor([0.1]).to(torch.float64)

# print(t32.frexp())
# print(t64.frexp())


# import math

# def float_to_ieee32(number):
#     whole = int(number)
#     fractional = number - whole
#     whole_bitstring = bin(whole)[2:]

#     fractional_bitstring = ""
#     for _ in range(23):
#         fractional = fractional * 2
#         if fractional >= 1:
#             fractional_bitstring += "1"
#             fractional -= 1
#         else:
#             fractional_bitstring += "0"

#     print(f'{whole_bitstring}.{fractional_bitstring}')

#     int_len = len(whole_bitstring)
#     frac_len = len(fractional_bitstring)
#     mantissa = whole_bitstring+fractional_bitstring
#     first_one_index = mantissa.find("1")
#     exponent = int_len - first_one_index - 1
#     print(mantissa + "0" * -exponent)
#     mantissa = (mantissa + "0" * -exponent)[first_one_index+1:first_one_index+24]
#     print(mantissa, len(mantissa))
#     print(exponent)

# def float_to_ieee32(number):
#     mantissa, exponent = math.frexp(number)
#     mantissa = bin(mantissa)

# def float_to_ieee64(number):
#     mantissa, exponent = math.frexp(number)
#     mantissa = bin(mantissa)

# # Function converts the value passed as
# # parameter to it's decimal representation
# def decimal_converter(num):
# 	while num > 1:
# 		num /= 10
# 	return num

# # Driver Code
# print(float_to_ieee32(0.1))
# print(float_to_ieee64(0.1))


# 110011001100110011000000 23
# -4