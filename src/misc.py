from typing import List
import math
import string
import itertools

class Misc:
    @staticmethod
    def get_help_message():
        return (
            "Available Commands:\n"
            "\n"
            "BASIS <matrix> <length> <base> <message>\n"
            "    Encodes the message using the given matrix, length, and base. The parity bits are represented by '_'.\n"
            "\n"
            "CODEWORD <string>\n"
            "    Generates the ASCII character that provides the parity check for the given string in binary representation.\n"
            "\n"
            "COMMA <decode|encode> <message>\n"
            "    Decodes or encodes a message using comma codes.\n"
            "\n"
            "CORRECT <binary_string>\n"
            "    Identifies and corrects a single error in an ASCII binary string.\n"
            "\n"
            "CHECK <base> <matrix> <codeword>\n"
            "    Checks a codeword against a linear code defined by the given matrix and base. Attempts to correct a single error if detected.\n"
            "\n"
            "DECODABLE <codewords> | <candidates>\n"
            "    Checks if the candidate codewords are uniquely decodable with the given set of codewords.\n"
            "\n"
            "EXTENSION <radix> <extension_length> <denominator> <numerators>\n"
            "    Calculates the average length of the code extension based on the given parameters.\n"
            "\n"
            "FACTORS <integer>\n"
            "    Computes and displays the factors of the given integer.\n"
            "\n"
            "HELP\n"
            "    Displays this help message.\n"
            "\n"
            "HUFFMAN <radix> <denominator> <numerators>\n"
            "    Computes the average codeword length of a Huffman code given the radix and symbol probabilities.\n"
            "\n"
            "HUFFMANCONV <radix> <numerators> <denominator>\n"
            "    Calculates the average codeword length as the code approaches infinite length.\n"
            "\n"
            "ISBN <isbn_numbers>\n"
            "    Validates ISBN numbers and computes missing digits represented by '_'.\n"
            "\n"
            "INVERSE <number> <modulus>\n"
            "    Computes the multiplicative inverse of the number modulo the given modulus.\n"
            "\n"
            "KMT <radix> <K> <codeword_lengths>\n"
            "    Applies the Kraft-McMillan theorem to determine if a prefix code is possible with the given codeword lengths.\n"
            "\n"
            "MR <value> <modulus>\n"
            "    Performs the Miller-Rabin primality test to check if 'value' is a strong pseudoprime to the base 'modulus'.\n"
            "\n"
            "MUL <base> <matrix1> <length1> <matrix2> <length2>\n"
            "    Multiplies two matrices in the given base.\n"
            "\n"
            "NULL <matrix> <length> <base> <message>\n"
            "    Finds a null vector for the given matrix and encodes the message with parity bits represented by '_'.\n"
            "\n"
            "ENTROPY <elem1> <elem2> <elem3> <elem4> <numerator1> <numerator2> <denominator>\n"
            "    Calculates the Markov entropy for a transition matrix and equilibrium distribution.\n"
            "\n"
            "POLYNOMIAL <field_size> <coefficients>\n"
            "    Generates polynomial terms over the specified field size until it cycles back to 1.\n"
            "\n"
            "POWER <base> <exponent> <modulus>\n"
            "    Computes (base^exponent) mod modulus efficiently.\n"
            "\n"
            "PRIMITIVES <modulus>\n"
            "    Finds all primitive roots modulo the given modulus.\n"
            "\n"
            "QUIT\n"
            "    Exits the interface.\n"
            "\n"
            "SF <radix> <numerators> <denominator>\n"
            "    Calculates Shannon-Fano codeword lengths based on symbol probabilities.\n"
            "\n"
            "SFC <radix> <numerators> <denominator>\n"
            "    Generates Shannon-Fano codewords given the radix and symbol probabilities.\n"
            "\n"
            "SPB <message_length> <error_correction> <radix> <code_length>\n"
            "    Solves for the unknown parameter in the sphere-packing bound equation.\n"
            "\n"
            "TOTIENT <integer>\n"
            "    Computes Euler's totient function φ(n) for the given integer.\n"
            "\n"
            "BCH_ENCODE <modulus> <information_polynomial> <generator_polynomial>\n"
            "    Encodes a message using BCH codes. Provide the modulus (usually 2 for binary codes),\n"
            "    the information polynomial coefficients, and the generator polynomial coefficients.\n"
            "\n"
            "BCH_DECODE <modulus> <received_codeword> <generator_polynomial> <field_degree> <modulus_polynomial>\n"
            "    Decodes a received codeword using BCH codes. Provide the modulus, received codeword coefficients,\n"
            "    generator polynomial coefficients, field extension degree (m), and modulus polynomial coefficients for GF(2^m).\n"
            )


    # Solves for one unknown in the sphere packing theorem
    # errors = minimum distance - 1 // 2
    @staticmethod
    def sphere_packing_bounds(input: List[str]):
        if len(input) < 3:
            return "Sphere Packing Bounds calculation requires arg1 = length and arg2 = errors."

        message = int(input[0]) if input[0].isnumeric() else None
        input[0] = message
        errors = int(input[1]) if input[1].isnumeric() else None
        input[1] = errors
        radix = int(input[2]) if input[2].isnumeric() else None
        input[2] = radix
        length = int(input[3]) if input[3].isnumeric() else None
        input[3] = length

        if input[:4].count(None) != 1: return "One argument should be nonnumeric."
        
        if message != None: message_len = radix ** message
        if errors != None and length != None: errors_comb = sum(math.comb(length, i) for i in range(errors + 1))
        if length != None: length_exp = radix ** length
        
        if message == None:
            code_len = length_exp / errors_comb
            return "Maximum code length is ⌊" + f"{code_len:.4f}" + "⌋ = " + str(int(code_len)) + ".\n"
        
        if errors == None:
            ratio = length_exp // message_len
            
            curr_error = 0
            errors_comb = 0
            while errors_comb < ratio:
                errors_comb += math.comb(length, curr_error)
                curr_error += 1
            
            return "Error correction capability is t = " + str(curr_error - 1) + ".\n"
        
        if length == None:
            curr_length = 0
            
            curr_errors_comb = sum(math.comb(curr_length, i) for i in range(errors + 1))
            curr_length_exp = radix ** curr_length
            
            while(message_len * curr_errors_comb > curr_length_exp):
                curr_length += 1
                curr_length_exp *= radix
                curr_errors_comb = sum(math.comb(curr_length, i) for i in range(errors + 1))
                
            return "Minimum code length is n = " + str(curr_length) + ".\n"

    @staticmethod
    def analyze_and_decrypt_ciphertext(text: str, top_n_shifts: int = 3, max_keywords: int = 10) -> None:
        language_frequencies = {
            'A': 8.167, 'B': 1.492, 'C': 2.782, 'D': 4.253, 'E': 12.702,
            'F': 2.228, 'G': 2.015, 'H': 6.094, 'I': 6.966, 'J': 0.153,
            'K': 0.772, 'L': 4.025, 'M': 2.406, 'N': 6.749, 'O': 7.507,
            'P': 1.929, 'Q': 0.095, 'R': 5.987, 'S': 6.327, 'T': 9.056,
            'U': 2.758, 'V': 0.978, 'W': 2.361, 'X': 0.150, 'Y': 1.974,
            'Z': 0.074
        }

        # Remove non-alphabetic characters and convert to uppercase
        clean_text = ''.join(filter(str.isalpha, text)).upper()

        n = len(clean_text)
        if n <= 1:
            print("Text is too short to analyze.")
            return

        # Calculate the Index of Coincidence (IC)
        frequency = {}
        for letter in clean_text:
            frequency[letter] = frequency.get(letter, 0) + 1

        numerator = sum(f * (f - 1) for f in frequency.values())
        denominator = n * (n - 1)
        ic = numerator / denominator

        # Estimate the keyword length
        K0 = 0.0385  # Expected IC for random English text
        K1 = 0.0658  # Expected IC for normal English text

        numerator_r = 0.0273 * n
        denominator_r = ((n - 1) * ic) - (K0 * n) + K1

        if denominator_r == 0:
            print(f"Index of Coincidence (IC): {ic:.4f}")
            print("Cannot estimate keyword length due to zero denominator.")
            return

        estimated_r = numerator_r / denominator_r
        print(f"Index of Coincidence (IC): {ic:.4f}")
        print(f"Estimated keyword length (r): {estimated_r:.2f}")

        # Consider keyword lengths around the estimated length
        min_keyword_length = max(1, int(estimated_r) - 1)
        max_keyword_length = int(estimated_r) + 1

        # Limit the maximum keyword length to a reasonable number
        max_keyword_length = min(max_keyword_length, 10)

        for r in range(min_keyword_length, max_keyword_length + 1):
            print(f"\nAnalyzing for keyword length: {r}")

            # Perform frequency analysis for each segment
            top_shifts_per_position = []
            for i in range(r):
                segment = clean_text[i::r]
                segment_freq = {}
                for letter in segment:
                    segment_freq[letter] = segment_freq.get(letter, 0) + 1

                # Shift analysis to find the top N matches
                chi_squared_scores = []
                for shift in range(26):
                    chi_squared = 0.0
                    for letter in string.ascii_uppercase:
                        shifted_letter = chr(((ord(letter) - 65 - shift) % 26) + 65)
                        observed = segment_freq.get(shifted_letter, 0)
                        expected = (len(segment) * language_frequencies[letter]) / 100
                        if expected > 0:
                            chi_squared += ((observed - expected) ** 2) / expected
                    chi_squared_scores.append((shift, chi_squared))

                # Sort the shifts based on chi-squared scores and keep top N
                chi_squared_scores.sort(key=lambda x: x[1])
                top_shifts = chi_squared_scores[:top_n_shifts]
                top_shifts_per_position.append(top_shifts)

            # Generate candidate keywords
            candidate_keywords = []
            for shifts_combination in itertools.product(*top_shifts_per_position):
                keyword = ''
                for shift, _ in shifts_combination:
                    keyword_letter = chr((65 + shift) % 26 + 65)
                    keyword += keyword_letter
                candidate_keywords.append(keyword)
                if len(candidate_keywords) >= max_keywords:
                    break

            # Decrypt with candidate keywords
            for keyword in candidate_keywords:
                decrypted_message = Misc.vigenere_decrypt(text, keyword)
                print(f"\nKeyword: {keyword}")
                print("Decrypted Message:")
                print(decrypted_message)
                print("-" * 50)

    @staticmethod
    def vigenere_decrypt(ciphertext: str, keyword: str) -> str:

        keyword_length = len(keyword)
        keyword_indices = [ord(k.upper()) - ord('A') for k in keyword]

        plaintext = ""

        keyword_index = 0
        for char in ciphertext:
            if char.isalpha():
                offset = ord('A') if char.isupper() else ord('a')
                letter_index = ord(char.upper()) - ord('A')
                key_index = keyword_indices[keyword_index % keyword_length]
                decrypted_index = (letter_index - key_index) % 26
                decrypted_char = chr(decrypted_index + offset)
                plaintext += decrypted_char
                keyword_index += 1
            else:
                plaintext += char

        return plaintext