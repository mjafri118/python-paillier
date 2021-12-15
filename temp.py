def float_to_ieee32(number):
    whole = int(number)
    fractional = number - whole
    whole_bitstring = bin(whole)[2:]

    fractional_bitstring = ""
    for _ in range(23):
        fractional = fractional * 2
        if fractional >= 1:
            fractional_bitstring += "1"
            fractional -= 1
        else:
            fractional_bitstring += "0"

    print(f'{whole_bitstring}.{fractional_bitstring}')

    int_len = len(whole_bitstring)
    frac_len = len(fractional_bitstring)
    mantissa = whole_bitstring+fractional_bitstring
    first_one_index = mantissa.find("1")
    exponent = int_len - first_one_index - 1
    print(mantissa + "0" * -exponent)
    mantissa = (mantissa + "0" * -exponent)[first_one_index+1:first_one_index+24]
    print(mantissa, len(mantissa))
    print(exponent)

# def float_to_ieee32(number):
# 	for x in range(3):

# 		# Multiply the decimal value by 2
# 		# and separate the whole number part
# 		# and decimal part
# 		whole, dec = str((decimal_converter(dec)) * 2).split(".")

# 		# Convert the decimal part
# 		# to integer again
# 		dec = int(dec)

# 		# Keep adding the integer parts
# 		# receive to the result variable
# 		res += whole

# 	return res

# Function converts the value passed as
# parameter to it's decimal representation
def decimal_converter(num):
	while num > 1:
		num /= 10
	return num

# Driver Code
print(float_to_ieee32(0.1))

10111.00000011001100110011001