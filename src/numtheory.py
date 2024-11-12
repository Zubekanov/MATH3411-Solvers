from typing import List
import math

class NumberTheory:
    @staticmethod
    def factors(n: int):
        factors_list = []
        while n > 1:
            for factor in range(2, n + 1):
                if n % factor == 0:
                    factors_list.append(factor)
                    n //= factor
                    break
        return factors_list

    @staticmethod
    def interface_factors(input: List[str]):
        result = ""
        prev_factor = -1
        exponent = 1
        for factor in NumberTheory.factors(int(input[0])):
            if factor == prev_factor:
                exponent += 1
            else:
                if prev_factor != -1:
                    result += " × "
                    result += f"{prev_factor}^{exponent}"
                exponent = 1
            prev_factor = factor
        result += f" × {prev_factor}^{exponent}"
        return f"{input[0]} = " + result[3:]

    @staticmethod
    def coprime(a: int, n: int):
        return (a ** NumberTheory.totient(n)) % n
    
    @staticmethod
    def interface_coprime(input: List[str]):
        a = int(input[0])
        n = int(input[1])
        return f"{a}^φ({n}) mod {n} = {NumberTheory.coprime(a, n)}"

    @staticmethod
    def totient(n: int):
        factors = NumberTheory.factors(n)
        prev_factor = -1
        totient = 1
        for factor in factors:
            if factor == prev_factor:
                totient *= factor
            else:
                prev_factor = factor
                totient *= (factor - 1)
        return totient

    @staticmethod
    def interface_totient(input: List[str]):
        n = int(input[0])
        totient = NumberTheory.totient(n)
        return f"φ({n}) = {totient}\n"
    
    @staticmethod
    def miller_rabin(input: List[str]):
        values = [int(i) for i in input[0:-1]]
        n = int(input[-1])
        # write n-1 as 2^s * d with d odd.
        s = 0
        d = n - 1
        while d % 2 == 0:
            s += 1
            d //= 2
        
        result = ""
        for a in values:
            # strong if a^d = 1 mod n
            if (a ** d) % n == 1:
                result += f"N = {n} is a Strong Pseudoprime to Base {a}. (Test 1)\n"
            elif any((a ** ((2 ** r) * d) % n == (n - 1)) for r in range(1, s)):
                result += f"N = {n} is a Strong Pseudoprime to Base {a}. (Test 2)\n"
            else:
                result += f"N = {n} is not a Strong Pseudoprime to Base {a}.\n"
        
        return result

    @staticmethod
    def inverse(input: List[str]):
        num = int(input[0])
        mod = int(input[1])
        for n in range(0, mod):
            if (n * num) % mod == 1:
                return f"Inverse of {num} in Z_{mod} is {n}.\n"
        return f"No inverse of {num} in Z_{mod}.\n"
    
    @staticmethod
    def powwow(input: List[str]):
        base = int(input[0])
        exponent = int(input[1])
        mod = int(input[2])

        value = base

        for i in range(1, exponent):
            value *= base
            value %= mod
        
        return f"{base}^{exponent} mod {mod} = {value}.\n"
    
    @staticmethod
    def primitives(input: List[str]):
        mod = int(input[0])
        primitives = []

        # Step 1: Identify the multiplicative group elements (coprime with mod)
        multiplicative_group = [x for x in range(1, mod) if math.gcd(x, mod) == 1]
        group_size = len(multiplicative_group)

        if group_size == 0:
            return f"No multiplicative group exists for Z_{mod}.\n"

        # Step 2: Iterate through each candidate in the multiplicative group
        for i in multiplicative_group:
            generated = set()
            value = i

            # Generate powers of i modulo mod
            while value not in generated:
                generated.add(value)
                value = (value * i) % mod

                # Early termination if all group elements are generated
                if len(generated) == group_size:
                    primitives.append(i)
                    break

        if primitives:
            return f"Primitives of Z_{mod} are {primitives}.\n"
        else:
            return f"No primitive roots exist for Z_{mod}.\n"
