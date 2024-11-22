from enum import Enum
from typing import List, Tuple
import math
from fractions import Fraction
import heapq
from itertools import product

class Delimiter(Enum):
    COMMA_SPLIT     = "|"

class Comma:
    @staticmethod
    def decode(input: List[str]):
        if Delimiter.COMMA_SPLIT.value not in input:
            return "No code candidates were given.\n"
        
        codewords = []
        candidates = []
        longest_candidate = -1
        split = False

        for token in input:
            if token == Delimiter.COMMA_SPLIT.value:
                split = True
            else:
                if not split:
                    codewords.append(token)
                else:  
                    if len(token) > longest_candidate:
                        longest_candidate = len(token)
                    candidates.append(token)

        result = ""

        for candidate in candidates:
            decode_list = codewords
            decode_list.append(candidate)
            result += candidate.ljust(longest_candidate)
            result += " --> Decodable.\n" if Comma.is_uniquely_decodable(decode_list) else " --> Not Decodable.\n"

        return result

    @staticmethod
    def kraftmcmillan(input: List[str]):
        radix_input = input[0]
        if radix_input.isdigit():
            radix = int(radix_input)
            radix_unknown = False
        else:
            radix_unknown = True
            radix = None

        K_input = input[1]
        if K_input.isdigit() or '/' in K_input:
            try:
                K_value = Fraction(K_input)
                K_unknown = False
            except:
                K_unknown = True
                K_value = None
        elif K_input == 'K':
            K_unknown = True
            K_value = None
        else:
            K_unknown = True
            K_value = None

        codeword_lengths = []
        unknown_lengths = []
        for l in input[2:]:
            if l.isdigit():
                codeword_lengths.append(int(l))
            else:
                unknown_lengths.append(l)
                codeword_lengths.append(None)

        if unknown_lengths:
            length_unknown = True
        else:
            length_unknown = False

        unknown_variables = []
        if radix_unknown:
            unknown_variables.append('radix')
        if K_unknown:
            unknown_variables.append('K')
        if length_unknown:
            unknown_variables.append('codeword_length')

        if 'radix' in unknown_variables and 'K' in unknown_variables:
            if length_unknown:
                return "Error: Cannot solve for multiple unknowns (radix, K, and codeword length). Please provide at most two unknowns."
            else:
                radix_candidate = 2
                found = False
                while radix_candidate <= 1000:  # Arbitrary upper limit
                    try:
                        K_calculated = sum([Fraction(1, radix_candidate ** L) for L in codeword_lengths])
                    except TypeError:
                        return "Error: Codeword lengths must be integers."
                    if K_calculated <= 1:
                        found = True
                        break
                    radix_candidate += 1
                if found:
                    K_value = K_calculated
                    return f"The minimum radix r is {radix_candidate}, and the Kraft-McMillan coefficient K is {K_value}."
                else:
                    return "No suitable radix found that satisfies the Kraft-McMillan inequality."
        elif len(unknown_variables) == 1:
            unknown_variable = unknown_variables[0]
            if unknown_variable == 'codeword_length':
                if None in codeword_lengths:
                    unknown_index = codeword_lengths.index(None)
                else:
                    return "Error: Unknown codeword length not found."
                known_sum = sum([Fraction(1, radix ** L) for L in codeword_lengths if L is not None])
                delta = K_value - known_sum
                if delta <= 0:
                    return "No solution: The sum of the known codeword terms exceeds or equals K."
                x = -math.log(float(delta), radix)
                unknown_length = math.ceil(x)
                total_K = known_sum + Fraction(1, radix ** unknown_length)
                if total_K != K_value:
                    return "No integer solution found for the unknown codeword length."
                return f"The unknown codeword length ℓ is {unknown_length}."
            elif unknown_variable == 'K':
                if None in codeword_lengths:
                    return "Error: Cannot compute K with unknown codeword lengths."
                K_value = sum([Fraction(1, radix ** L) for L in codeword_lengths])
                return f"The Kraft-McMillan coefficient K is {K_value}."
            elif unknown_variable == 'radix':
                if None in codeword_lengths:
                    return "Error: Cannot compute radix with unknown codeword lengths."
                radix_candidate = 2
                found = False
                while radix_candidate <= 1000:  # Arbitrary upper limit
                    K_calculated = sum([Fraction(1, radix_candidate ** L) for L in codeword_lengths])
                    if K_calculated <= 1:
                        found = True
                        break
                    radix_candidate += 1
                if found:
                    return f"The radix r is {radix_candidate}."
                else:
                    return "No suitable radix found that satisfies the Kraft-McMillan inequality."
        else:
            return "Error: Cannot solve for multiple unknowns or insufficient information provided."
        
    @staticmethod
    def huffman_average_length(input: List[str]):
        radix = int(input[0])
        r = radix

        denominator = int(input[1])

        numerators = [int(num) for num in input[2:]]
        probabilities = [Fraction(num, denominator) for num in numerators]

        n = len(probabilities)

        k = math.ceil((n - 1) / (r - 1))
        required_symbols = k * (r - 1) + 1
        num_dummy_symbols = required_symbols - n

        probabilities.extend([Fraction(0, 1)] * num_dummy_symbols)

        nodes = [(prob, True, i) for i, prob in enumerate(probabilities)]

        while len(nodes) > 1:
            nodes.sort(key=lambda x: (x[0], x[1]))
            selected_nodes = nodes[:r]
            combined_prob = sum(node[0] for node in selected_nodes)
            new_node = (combined_prob, False, selected_nodes)
            nodes = nodes[r:]
            nodes.append(new_node)

        codeword_lengths = [0] * len(probabilities)

        def assign_lengths(node, length):
            prob, is_leaf, data = node
            if is_leaf:
                index = data
                codeword_lengths[index] = length
            else:
                for child in data:
                    assign_lengths(child, length + 1)

        root_node = nodes[0]
        assign_lengths(root_node, 0)

        original_codeword_lengths = codeword_lengths[:n]
        avg_length = sum(probabilities[i] * original_codeword_lengths[i] for i in range(n))

        avg_length = avg_length.limit_denominator()

        return f"The average codeword length is {avg_length}."

    @staticmethod
    def is_uniquely_decodable(codewords: List[str]) -> bool:
        L = set()

        L1 = set()
        for x in codewords:
            for y in codewords:
                if x != y and x.startswith(y):
                    suffix = x[len(y):]
                    if suffix:
                        L1.add(suffix)
        L.update(L1)

        prev_L = set()
        while L != prev_L:
            if '' in L:
                return False
            prev_L = L.copy()
            new_L = set()
            for suffix in L:
                for codeword in codewords:
                    if suffix.startswith(codeword):
                        new_suffix = suffix[len(codeword):]
                        if new_suffix:
                            new_L.add(new_suffix)
                        else:
                            return False
                    if codeword.startswith(suffix) and suffix != codeword:
                        new_suffix = codeword[len(suffix):]
                        if new_suffix:
                            new_L.add(new_suffix)
                        else:
                            return False
            L = L.union(new_L)
        return True

    def extension_length(input_params: List[str]) -> Fraction:
        try:
            radix = int(input_params[0])
            extension = int(input_params[1])
            denominator = int(input_params[2])
            numerators = list(map(int, input_params[3:]))
        except (ValueError, IndexError) as e:
            raise ValueError("Invalid input parameters. Ensure radix, extension, denominator, and numerators are correctly provided.") from e

        q = len(numerators)  # Number of original symbols

        if radix < 2:
            raise ValueError("Radix must be at least 2.")
        if extension < 1:
            raise ValueError("Extension level must be at least 1.")
        if denominator <= 0:
            raise ValueError("Denominator must be a positive integer.")
        if any(num < 0 for num in numerators):
            raise ValueError("Numerators must be non-negative integers.")
        if sum(numerators) > denominator:
            raise ValueError("Sum of numerators cannot exceed the denominator.")

        original_symbols = [f's{i+1}' for i in range(q)]
        original_probs = [Fraction(num, denominator) for num in numerators]

        S_n_symbols = [tuple(seq) for seq in product(original_symbols, repeat=extension)]

        S_n_probs = []
        for symbol in S_n_symbols:
            prob = Fraction(1,1)
            for char in symbol:
                index = original_symbols.index(char)
                prob *= original_probs[index]
            S_n_probs.append(prob)

        sorted_S_n = sorted(zip(S_n_symbols, S_n_probs), key=lambda x: (-x[1], x[0]))

        heap = []
        for symbol, prob in sorted_S_n:
            heap.append((prob, [symbol]))  # Each heap element: (probability, [list of symbols])

        heapq.heapify(heap)

        codeword_lengths = {symbol: 0 for symbol, prob in sorted_S_n}

        while len(heap) >1:
            # Number of nodes to combine
            num_to_combine = radix

            # If the number of nodes minus 1 is not divisible by (radix -1), add dummy nodes
            while (len(heap) -1) % (radix -1) !=0:
                heapq.heappush(heap, (Fraction(0), []))  # Dummy node with probability 0 and no symbols

            combined_prob = Fraction(0)
            combined_symbols = []
            for _ in range(radix):
                prob, symbols = heapq.heappop(heap)
                combined_prob += prob
                combined_symbols.extend(symbols)
                for sym in symbols:
                    codeword_lengths[sym] +=1

            heapq.heappush(heap, (combined_prob, combined_symbols))

        average_length = Fraction(0,1)
        for (symbol, prob) in sorted_S_n:
            average_length += prob * codeword_lengths[symbol]
        
        return f"Length of extension is {average_length}.\n"

    @staticmethod
    def shannon_fano(input: List[str]) -> str:
        if len(input) < 3:
            return "Error: Insufficient input. Expected at least radix, one numerator, and denominator."

        try:
            radix = int(input[0])

            numerators = [int(i) for i in input[1:-1]]
            denominator = int(input[-1])

            if denominator == 0:
                return "Error: Denominator cannot be zero."

            probabilities = []
            for num in numerators:
                if num < 0:
                    return "Error: Numerators cannot be negative."
                prob = Fraction(num, denominator)
                if prob > 1:
                    return "Error: Probability cannot exceed 1."
                probabilities.append(prob)

            total_prob = sum(probabilities)
            if not math.isclose(float(total_prob), 1.0, abs_tol=1e-6):
                return f"Error: Probabilities sum to {float(total_prob)}, expected 1."

            total_length = Fraction(0, 1)
            for prob in probabilities:
                if prob == 0:
                    return "Error: Probability of zero encountered."
                log_prob = math.log(prob, radix)
                ceil_length = math.ceil(-log_prob)
                total_length += prob * ceil_length

            return f"Average length is {total_length} bits"

        except ValueError:
            return "Error: Invalid input format. Ensure all inputs are integers."
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def generate_shannon_fano_code(input: List[str]):
        radix = int(input[0])
        numerators = [int(i) for i in input[1:-1]]
        denominator = int(input[-1])
        probabilities = [n / denominator for n in numerators]

        word_lengths = [math.ceil(-math.log(p, radix)) for p in probabilities]

        sorted_indices = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
        sorted_lengths = [word_lengths[i] for i in sorted_indices]

        codeword_set = set()
        codewords = [None] * len(numerators)
        next_codeword = [0] * max(sorted_lengths)

        for idx, length in enumerate(sorted_lengths):
            while True:
                codeword = ''.join(str(d) for d in next_codeword[:length])
                is_prefix_free = all(not codeword.startswith(cw) and not cw.startswith(codeword) for cw in codeword_set)
                if is_prefix_free:
                    break
                for i in reversed(range(length)):
                    if next_codeword[i] < radix - 1:
                        next_codeword[i] += 1
                        for j in range(i + 1, length):
                            next_codeword[j] = 0
                        break
                else:
                    raise ValueError("Cannot assign codeword; all combinations exhausted.")

            codeword_set.add(codeword)
            original_index = sorted_indices[idx]
            codewords[original_index] = codeword

            if next_codeword[length - 1] < radix - 1:
                next_codeword[length - 1] += 1
            else:
                for i in reversed(range(length)):
                    if next_codeword[i] < radix - 1:
                        next_codeword[i] += 1
                        for j in range(i + 1, length):
                            next_codeword[j] = 0
                        break
                else:
                    next_codeword = [0] * (max(sorted_lengths) + 1)

        result = ""
        position = 1
        for codeword in codewords:
            result += f"Codeword S_{position} = {codeword}\n"
            position += 1
        return result

    @staticmethod
    def huffman_conv(input: List[str]):
        radix = int(input[0])
        numerators = [int(x) for x in input[1:-1]]
        denominator = int(input[-1])

        floats = [numerator / denominator for numerator in numerators]
        sum = 0
        for float in floats:
            sum -= float * math.log(float, radix)
        
        return f"Average codeword length as n → ∞ is {sum:.3f}"

class BCHCode:
    @staticmethod
    def poly_add(a: List[int], b: List[int], modulus: int) -> List[int]:
        # Same as before
        max_len = max(len(a), len(b))
        result = []
        for i in range(max_len):
            coeff_a = a[i] if i < len(a) else 0
            coeff_b = b[i] if i < len(b) else 0
            result.append((coeff_a + coeff_b) % modulus)
        # Remove leading zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    @staticmethod
    def poly_sub(a: List[int], b: List[int], modulus: int) -> List[int]:
        # Subtraction is the same as addition in GF(2)
        return BCHCode.poly_add(a, b, modulus)

    @staticmethod
    def poly_mul(a: List[int], b: List[int], modulus: int) -> List[int]:
        # Same as before
        result = [0] * (len(a) + len(b) - 1)
        for i, coeff_a in enumerate(a):
            for j, coeff_b in enumerate(b):
                result[i + j] = (result[i + j] + coeff_a * coeff_b) % modulus
        # Remove leading zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    @staticmethod
    def poly_divmod(a: List[int], b: List[int], modulus: int) -> Tuple[List[int], List[int]]:
        # Adjusted for binary field
        a = a[:]
        quotient = [0] * (len(a) - len(b) + 1)
        while len(a) >= len(b):
            degree_diff = len(a) - len(b)
            quotient[degree_diff] = 1
            subtract_poly = [0] * degree_diff + b
            a = BCHCode.poly_sub(a, subtract_poly, modulus)
            # Remove leading zeros
            while len(a) > 0 and a[-1] == 0:
                a.pop()
        remainder = a
        return quotient, remainder

    @staticmethod
    def poly_eval(poly: List[int], x: int, field_size: int, modulus_poly: int) -> int:
        """
        Evaluates a polynomial at a given point x in GF(2^m) using the field representation.
        """
        result = 0
        for coeff in reversed(poly):
            result = (result * x + coeff) % field_size
            # Reduce modulo the modulus polynomial if necessary
            if result >= field_size:
                result ^= modulus_poly
        return result

    @staticmethod
    def bch_encode(input: List[str]) -> List[int]:
        # Same as before
        modulus = int(input[0])
        information_poly_coeffs = [int(c) for c in input[1].split(',')]
        generator_poly_coeffs = [int(c) for c in input[2].split(',')]

        # Reverse the polynomials to have lowest degree first
        information_poly_coeffs = information_poly_coeffs[::-1]
        generator_poly_coeffs = generator_poly_coeffs[::-1]

        # Multiply I(x) by x^(n - k), where n is the code length and k is the message length
        n = len(generator_poly_coeffs) + len(information_poly_coeffs) - 1
        k = len(information_poly_coeffs)
        shifted_information = [0] * (n - k) + information_poly_coeffs

        # Compute the remainder R(x) = shifted_information mod generator_poly
        _, remainder = BCHCode.poly_divmod(shifted_information, generator_poly_coeffs, modulus)

        # Compute the codeword polynomial C(x) = shifted_information + remainder
        codeword_poly_coeffs = BCHCode.poly_add(shifted_information, remainder, modulus)

        # Reverse back to highest degree first
        codeword_poly_coeffs = codeword_poly_coeffs[::-1]

        return codeword_poly_coeffs

    @staticmethod
    def bch_decode(input: List[str]) -> Tuple[List[int], List[int]]:
        """
        Decodes the received codeword using the BCH decoding algorithm.
        input: A list of strings where
            input[0]: modulus (field characteristic, e.g., '2' for binary field)
            input[1]: received codeword coefficients (comma-separated string)
            input[2]: generator polynomial coefficients (comma-separated string)
            input[3]: field extension degree m (for GF(2^m))
            input[4]: modulus polynomial coefficients for GF(2^m) (comma-separated string)
        Returns a tuple of:
            - corrected codeword polynomial coefficients as a list of integers
            - decoded message polynomial coefficients as a list of integers
        """
        # Parse input
        modulus = int(input[0])
        received_poly_coeffs = [int(c) for c in input[1].split(',')]
        generator_poly_coeffs = [int(c) for c in input[2].split(',')]
        m = int(input[3])
        modulus_poly_coeffs = [int(c) for c in input[4].split(',')]

        # Reverse the polynomials to have lowest degree first
        received_poly_coeffs = received_poly_coeffs[::-1]
        generator_poly_coeffs = generator_poly_coeffs[::-1]
        modulus_poly = BCHCode.poly_to_int(modulus_poly_coeffs)

        # Field size
        field_size = 2 ** m

        # Step 1: Compute syndromes
        t = (len(generator_poly_coeffs) - 1) // m  # Error-correcting capability
        syndromes = []
        for i in range(1, 2 * t + 1):
            s = BCHCode.evaluate_syndrome(received_poly_coeffs, i, m, modulus_poly)
            syndromes.append(s)

        # Check if all syndromes are zero (no errors)
        if all(s == 0 for s in syndromes):
            # No errors, return the message part of the codeword
            corrected_codeword = received_poly_coeffs[::-1]
            message = corrected_codeword[-(len(received_poly_coeffs) - len(generator_poly_coeffs) + 1):]
            return corrected_codeword, message

        # Step 2: Use Berlekamp-Massey algorithm to find error locator polynomial
        sigma = BCHCode.berlekamp_massey_algorithm(syndromes, field_size, modulus_poly)

        # Step 3: Find the roots of the error locator polynomial (Chien search)
        error_positions = BCHCode.chien_search(sigma, field_size, modulus_poly)

        # Step 4: Correct the errors in the received codeword
        corrected_poly = received_poly_coeffs[:]
        for position in error_positions:
            corrected_poly[position] ^= 1  # Flip the bit

        # Step 5: Extract the original message
        corrected_codeword = corrected_poly[::-1]
        message_length = len(corrected_codeword) - len(generator_poly_coeffs) + 1
        message = corrected_codeword[-message_length:]

        return corrected_codeword, message

    @staticmethod
    def poly_to_int(poly_coeffs: List[int]) -> int:
        """
        Converts a polynomial represented by coefficients to an integer.
        """
        result = 0
        for coeff in poly_coeffs:
            result = (result << 1) | coeff
        return result

    @staticmethod
    def int_to_poly(x: int, degree: int) -> List[int]:
        """
        Converts an integer to a polynomial represented by coefficients.
        """
        coeffs = []
        for i in range(degree + 1):
            coeffs.append((x >> i) & 1)
        return coeffs

    @staticmethod
    def evaluate_syndrome(received_poly: List[int], power: int, m: int, modulus_poly: int) -> int:
        """
        Evaluates the syndrome S_power for the received polynomial.
        """
        # Generate alpha^power
        alpha_power = BCHCode.gf_pow(2, power, m, modulus_poly)
        # Evaluate the polynomial at alpha^power
        syndrome = 0
        for i, coeff in enumerate(received_poly):
            if coeff != 0:
                exp = (i * power) % (2 ** m - 1)
                term = BCHCode.gf_pow(2, exp, m, modulus_poly)
                syndrome ^= term
        return syndrome

    @staticmethod
    def gf_pow(base: int, exp: int, m: int, modulus_poly: int) -> int:
        """
        Computes base^exp in GF(2^m) using the modulus polynomial.
        """
        result = 1
        for _ in range(exp):
            result = BCHCode.gf_mult(result, base, modulus_poly)
        return result

    @staticmethod
    def gf_mult(a: int, b: int, modulus_poly: int) -> int:
        """
        Multiplies two elements in GF(2^m) represented as integers.
        """
        result = 0
        while b > 0:
            if b & 1:
                result ^= a
            a <<= 1
            if a & (1 << modulus_poly.bit_length() - 1):
                a ^= modulus_poly
            b >>= 1
        return result

    @staticmethod
    def berlekamp_massey_algorithm(syndromes: List[int], field_size: int, modulus_poly: int) -> List[int]:
        """
        Implements the Berlekamp-Massey algorithm to find the error locator polynomial.
        """
        n = len(syndromes)
        c = [1] + [0] * n  # Error locator polynomial
        b = [1] + [0] * n
        l = 0
        m = -1
        for r in range(n):
            # Compute discrepancy
            d = syndromes[r]
            for i in range(1, l + 1):
                d ^= BCHCode.gf_mult(c[i], syndromes[r - i], modulus_poly)
            if d != 0:
                temp = c[:]
                for i in range(r - m, n):
                    c[i] ^= BCHCode.gf_mult(d, b[i - (r - m)], modulus_poly)
                if 2 * l <= r:
                    l = r + 1 - l
                    m = r
                    b = temp
        # Truncate the polynomial to degree l
        return c[:l + 1]

    @staticmethod
    def chien_search(sigma: List[int], field_size: int, modulus_poly: int) -> List[int]:
        """
        Performs Chien search to find the roots of the error locator polynomial.
        Returns the error positions.
        """
        error_positions = []
        n = field_size - 1
        for i in range(n):
            x = BCHCode.gf_pow(2, i, modulus_poly.bit_length() - 1, modulus_poly)
            result = 0
            for coeff in reversed(sigma):
                result = BCHCode.gf_mult(result, x, modulus_poly) ^ coeff
            if result == 0:
                error_positions.append(n - 1 - i)
        return error_positions

class BCHCode:
    @staticmethod
    def poly_add(a: List[int], b: List[int], modulus: int) -> List[int]:
        # Same as before
        max_len = max(len(a), len(b))
        result = []
        for i in range(max_len):
            coeff_a = a[i] if i < len(a) else 0
            coeff_b = b[i] if i < len(b) else 0
            result.append((coeff_a + coeff_b) % modulus)
        # Remove leading zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    @staticmethod
    def poly_sub(a: List[int], b: List[int], modulus: int) -> List[int]:
        # Subtraction is the same as addition in GF(2)
        return BCHCode.poly_add(a, b, modulus)

    @staticmethod
    def poly_mul(a: List[int], b: List[int], modulus: int) -> List[int]:
        # Same as before
        result = [0] * (len(a) + len(b) - 1)
        for i, coeff_a in enumerate(a):
            for j, coeff_b in enumerate(b):
                result[i + j] = (result[i + j] + coeff_a * coeff_b) % modulus
        # Remove leading zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    @staticmethod
    def poly_divmod(a: List[int], b: List[int], modulus: int) -> Tuple[List[int], List[int]]:
        # Adjusted for binary field
        a = a[:]
        quotient = [0] * (len(a) - len(b) + 1)
        while len(a) >= len(b):
            degree_diff = len(a) - len(b)
            quotient[degree_diff] = 1
            subtract_poly = [0] * degree_diff + b
            a = BCHCode.poly_sub(a, subtract_poly, modulus)
            # Remove leading zeros
            while len(a) > 0 and a[-1] == 0:
                a.pop()
        remainder = a
        return quotient, remainder

    @staticmethod
    def poly_eval(poly: List[int], x: int, field_size: int, modulus_poly: int) -> int:
        """
        Evaluates a polynomial at a given point x in GF(2^m) using the field representation.
        """
        result = 0
        for coeff in reversed(poly):
            result = (result * x + coeff) % field_size
            # Reduce modulo the modulus polynomial if necessary
            if result >= field_size:
                result ^= modulus_poly
        return result

    @staticmethod
    def bch_encode(input: List[str]) -> List[int]:
        # Same as before
        modulus = int(input[0])
        information_poly_coeffs = [int(c) for c in input[1].split(',')]
        generator_poly_coeffs = [int(c) for c in input[2].split(',')]

        # Reverse the polynomials to have lowest degree first
        information_poly_coeffs = information_poly_coeffs[::-1]
        generator_poly_coeffs = generator_poly_coeffs[::-1]

        # Multiply I(x) by x^(n - k), where n is the code length and k is the message length
        n = len(generator_poly_coeffs) + len(information_poly_coeffs) - 1
        k = len(information_poly_coeffs)
        shifted_information = [0] * (n - k) + information_poly_coeffs

        # Compute the remainder R(x) = shifted_information mod generator_poly
        _, remainder = BCHCode.poly_divmod(shifted_information, generator_poly_coeffs, modulus)

        # Compute the codeword polynomial C(x) = shifted_information + remainder
        codeword_poly_coeffs = BCHCode.poly_add(shifted_information, remainder, modulus)

        # Reverse back to highest degree first
        codeword_poly_coeffs = codeword_poly_coeffs[::-1]

        return codeword_poly_coeffs

    @staticmethod
    def bch_decode(input: List[str]) -> Tuple[List[int], List[int]]:
        """
        Decodes the received codeword using the BCH decoding algorithm.
        input: A list of strings where
            input[0]: modulus (field characteristic, e.g., '2' for binary field)
            input[1]: received codeword coefficients (comma-separated string)
            input[2]: generator polynomial coefficients (comma-separated string)
            input[3]: field extension degree m (for GF(2^m))
            input[4]: modulus polynomial coefficients for GF(2^m) (comma-separated string)
        Returns a tuple of:
            - corrected codeword polynomial coefficients as a list of integers
            - decoded message polynomial coefficients as a list of integers
        """
        # Parse input
        modulus = int(input[0])
        received_poly_coeffs = [int(c) for c in input[1].split(',')]
        generator_poly_coeffs = [int(c) for c in input[2].split(',')]
        m = int(input[3])
        modulus_poly_coeffs = [int(c) for c in input[4].split(',')]

        # Reverse the polynomials to have lowest degree first
        received_poly_coeffs = received_poly_coeffs[::-1]
        generator_poly_coeffs = generator_poly_coeffs[::-1]
        modulus_poly = BCHCode.poly_to_int(modulus_poly_coeffs)

        # Field size
        field_size = 2 ** m

        # Step 1: Compute syndromes
        t = (len(generator_poly_coeffs) - 1) // m  # Error-correcting capability
        syndromes = []
        for i in range(1, 2 * t + 1):
            s = BCHCode.evaluate_syndrome(received_poly_coeffs, i, m, modulus_poly)
            syndromes.append(s)

        # Check if all syndromes are zero (no errors)
        if all(s == 0 for s in syndromes):
            # No errors, return the message part of the codeword
            corrected_codeword = received_poly_coeffs[::-1]
            message = corrected_codeword[-(len(received_poly_coeffs) - len(generator_poly_coeffs) + 1):]
            return corrected_codeword, message

        # Step 2: Use Berlekamp-Massey algorithm to find error locator polynomial
        sigma = BCHCode.berlekamp_massey_algorithm(syndromes, field_size, modulus_poly)

        # Step 3: Find the roots of the error locator polynomial (Chien search)
        error_positions = BCHCode.chien_search(sigma, field_size, modulus_poly)

        # Step 4: Correct the errors in the received codeword
        corrected_poly = received_poly_coeffs[:]
        for position in error_positions:
            corrected_poly[position] ^= 1  # Flip the bit

        # Step 5: Extract the original message
        corrected_codeword = corrected_poly[::-1]
        message_length = len(corrected_codeword) - len(generator_poly_coeffs) + 1
        message = corrected_codeword[-message_length:]

        return corrected_codeword, message

    @staticmethod
    def poly_to_int(poly_coeffs: List[int]) -> int:
        """
        Converts a polynomial represented by coefficients to an integer.
        """
        result = 0
        for coeff in poly_coeffs:
            result = (result << 1) | coeff
        return result

    @staticmethod
    def int_to_poly(x: int, degree: int) -> List[int]:
        """
        Converts an integer to a polynomial represented by coefficients.
        """
        coeffs = []
        for i in range(degree + 1):
            coeffs.append((x >> i) & 1)
        return coeffs

    @staticmethod
    def evaluate_syndrome(received_poly: List[int], power: int, m: int, modulus_poly: int) -> int:
        """
        Evaluates the syndrome S_power for the received polynomial.
        """
        # Generate alpha^power
        alpha_power = BCHCode.gf_pow(2, power, m, modulus_poly)
        # Evaluate the polynomial at alpha^power
        syndrome = 0
        for i, coeff in enumerate(received_poly):
            if coeff != 0:
                exp = (i * power) % (2 ** m - 1)
                term = BCHCode.gf_pow(2, exp, m, modulus_poly)
                syndrome ^= term
        return syndrome

    @staticmethod
    def gf_pow(base: int, exp: int, m: int, modulus_poly: int) -> int:
        """
        Computes base^exp in GF(2^m) using the modulus polynomial.
        """
        result = 1
        for _ in range(exp):
            result = BCHCode.gf_mult(result, base, modulus_poly)
        return result

    @staticmethod
    def gf_mult(a: int, b: int, modulus_poly: int) -> int:
        """
        Multiplies two elements in GF(2^m) represented as integers.
        """
        result = 0
        while b > 0:
            if b & 1:
                result ^= a
            a <<= 1
            if a & (1 << modulus_poly.bit_length() - 1):
                a ^= modulus_poly
            b >>= 1
        return result

    @staticmethod
    def berlekamp_massey_algorithm(syndromes: List[int], field_size: int, modulus_poly: int) -> List[int]:
        """
        Implements the Berlekamp-Massey algorithm to find the error locator polynomial.
        """
        n = len(syndromes)
        c = [1] + [0] * n  # Error locator polynomial
        b = [1] + [0] * n
        l = 0
        m = -1
        for r in range(n):
            # Compute discrepancy
            d = syndromes[r]
            for i in range(1, l + 1):
                d ^= BCHCode.gf_mult(c[i], syndromes[r - i], modulus_poly)
            if d != 0:
                temp = c[:]
                for i in range(r - m, n):
                    c[i] ^= BCHCode.gf_mult(d, b[i - (r - m)], modulus_poly)
                if 2 * l <= r:
                    l = r + 1 - l
                    m = r
                    b = temp
        # Truncate the polynomial to degree l
        return c[:l + 1]

    @staticmethod
    def chien_search(sigma: List[int], field_size: int, modulus_poly: int) -> List[int]:
        """
        Performs Chien search to find the roots of the error locator polynomial.
        Returns the error positions.
        """
        error_positions = []
        n = field_size - 1
        for i in range(n):
            x = BCHCode.gf_pow(2, i, modulus_poly.bit_length() - 1, modulus_poly)
            result = 0
            for coeff in reversed(sigma):
                result = BCHCode.gf_mult(result, x, modulus_poly) ^ coeff
            if result == 0:
                error_positions.append(n - 1 - i)
        return error_positions