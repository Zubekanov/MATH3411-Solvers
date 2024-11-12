from enum import Enum
from typing import List
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
        """
        Computes the average codeword length for Shannon-Fano coding and returns it as a fraction.
        
        Parameters:
        - input (List[str]): A list of strings where:
            - input[0] is the radix (e.g., '2' for binary),
            - input[1:-1] are the numerators (frequency counts) as strings,
            - input[-1] is the denominator (total frequency) as a string.
        
        Returns:
        - str: A string representing the average codeword length in the form 'Average length is X/Y bits'.
        """
        # Validate input length
        if len(input) < 3:
            return "Error: Insufficient input. Expected at least radix, one numerator, and denominator."

        try:
            # Extract radix
            radix = int(input[0])

            # Extract numerators and denominator
            numerators = [int(i) for i in input[1:-1]]
            denominator = int(input[-1])

            # Validate denominator
            if denominator == 0:
                return "Error: Denominator cannot be zero."

            # Validate numerators and compute probabilities as Fractions
            probabilities = []
            for num in numerators:
                if num < 0:
                    return "Error: Numerators cannot be negative."
                prob = Fraction(num, denominator)
                if prob > 1:
                    return "Error: Probability cannot exceed 1."
                probabilities.append(prob)

            # Validate that probabilities sum to 1 (within a tolerance)
            total_prob = sum(probabilities)
            if not math.isclose(float(total_prob), 1.0, abs_tol=1e-6):
                return f"Error: Probabilities sum to {float(total_prob)}, expected 1."

            # Compute codeword lengths and average length
            total_length = Fraction(0, 1)
            for prob in probabilities:
                if prob == 0:
                    return "Error: Probability of zero encountered."
                # Compute codeword length: ceil(-log_r(p_i))
                log_prob = math.log(prob, radix)
                ceil_length = math.ceil(-log_prob)
                total_length += prob * ceil_length

            # Represent the average length as a fraction
            return f"Average length is {total_length} bits"

        except ValueError:
            return "Error: Invalid input format. Ensure all inputs are integers."
        except Exception as e:
            return f"Error: {str(e)}"

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
