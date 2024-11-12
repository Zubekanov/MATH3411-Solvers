from typing import List

def poly_add(p1: List[int], p2: List[int], p: int) -> List[int]:
    """Add two polynomials over Z_p."""
    length = max(len(p1), len(p2))
    result = [(0) for _ in range(length)]
    for i in range(length):
        coef1 = p1[i] if i < len(p1) else 0
        coef2 = p2[i] if i < len(p2) else 0
        result[i] = (coef1 + coef2) % p
    # Remove leading zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def poly_mul(p1: List[int], p2: List[int], p: int) -> List[int]:
    """Multiply two polynomials over Z_p."""
    result = [0]*(len(p1)+len(p2)-1)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i+j] = (result[i+j] + p1[i]*p2[j]) % p
    return result

def poly_mod(poly: List[int], mod_poly: List[int], p: int) -> List[int]:
    """Reduce a polynomial modulo the minimal polynomial over Z_p."""
    poly = poly.copy()  # Make a copy to avoid modifying the input list
    deg_mod = len(mod_poly) - 1  # Degree of the minimal polynomial
    while len(poly) >= len(mod_poly):
        coef = poly[-1]  # Leading coefficient of the current polynomial
        if coef != 0:
            deg_poly = len(poly) - 1  # Degree of the current polynomial
            shift = deg_poly - deg_mod
            # Subtract coef * x^shift * mod_poly from poly
            for i in range(len(mod_poly)):
                index = shift + i
                neg = (-coef * mod_poly[i]) % p  # Compute additive inverse modulo p
                poly[index] = (poly[index] + neg) % p
        poly.pop()  # Remove the highest degree term, which is now zero
    # Remove leading zeros
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly

def poly_equal(p1: List[int], p2: List[int]) -> bool:
    """Check if two polynomials are equal."""
    return p1 == p2

def poly_str(poly: List[int]) -> str:
    """Convert a polynomial to string representation."""
    terms = []
    for i, coef in enumerate(poly):
        if coef != 0:
            term = f"{coef}" if i == 0 else f"{coef}*a^{i}"
            terms.append(term)
    return ' + '.join(terms) if terms else '0'

def generate_powers(field_size: int, min_poly: List[int]) -> str:
    """Generate powers of 'a' until a^n = 1, omitting exponents and coefficients of 1."""
    result = ""
    p = field_size
    # Initial polynomial 'a' (which is x)
    a = [0, 1]  # Represents 'a' or 'x'
    powers = [a]
    current_power = a.copy()
    n = 1
    result += f"a^{n} = {poly_str_no_ones(current_power)}\n"
    while True:
        # Multiply by 'a' and reduce modulo minimal polynomial
        current_power = poly_mul(current_power, a, p)
        current_power = poly_mod(current_power, min_poly, p)
        n += 1
        result += f"a^{n} = {poly_str_no_ones(current_power)}\n"
        # Check if current_power is [1], which represents '1'
        if poly_equal(current_power, [1]):
            result += f"\nFound minimal n such that a^{n} = 1: n = {n}\n"
            break
        # Prevent infinite loops in case minimal polynomial is invalid
        if n > p**len(min_poly):
            result += "\nExceeded maximum expected period. Check the minimal polynomial.\n"
            break
        powers.append(current_power)
    
    return result

def poly_str_no_ones(poly: List[int]) -> str:
    """Convert a polynomial to string representation without printing exponents or coefficients of 1."""
    terms = []
    for i, coef in enumerate(poly):
        if coef != 0:
            if i == 0:
                term = f"{coef}" if coef != 1 else "1"
            elif coef == 1:
                term = f"a^{i}" if i > 1 else "a"
            else:
                term = f"{coef}*a^{i}" if i > 1 else f"{coef}*a"
            terms.append(term)
    return ' + '.join(terms) if terms else '0'