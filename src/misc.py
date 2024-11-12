from typing import List
import math

class Misc:
    @staticmethod
    def get_help_message():
        return ("BASIS matrix length base message\n"
                "Given a matrix with specified length and base, encodes the message with \"_\" at the parity bits.\n\n"
                "CODEWORD string\n"
                "Given a string, generates the ASCII character that provides the parity check for the string with the binary representation.\n\n"
                "COMMA decode/encode message\n"
                "Decodes or encodes a comma code.\n\n"
                "CORRECT binary\n"
                "Identifies a single error in an ASCII binary string.\n\n"
                "CHECK base matrix codeword\n"
                "Checks codeword against a linear code and attempts to correct a signle error.\n\n"
                "DECODABLE codewords | candidates\n"
                "Checks the candidate codewords to see if they are decodable with the given codewords.\n\n"
                "EXTENSION radix extension denominator numerators\n"
                "Calculates the average length of the extension.\n\n"
                "FACTORS int\n"
                "Gives factors\n\n"
                "HELP\n"
                "Prints this message to interface.\n\n"
                "HUFFMAN radix denominator numerators\n"
                "Gives the average length of the Huffman code given radix and probabilities.\n\n"
                "HUFFMANCONV radix numerators denominator\n"
                "Gives average codeword length as n approaches inf\n\n"
                "ISBN isbn\n"
                "Given a list of ISBNs, calculates the syndrome for each, or calculates a missing character signified with \"_\".\n\n"
                "INVERSE num base\n"
                "Returns inverse for num in base.\n\n"
                "KMT radix K codeword_lengths\n"
                "Applies Kraft-Mcmillan Theorem with one non-numeric unknown.\n\n"
                "MR value mod\n"
                "Checks if the value is a strong pseudoprime in the modulus. (small large)\n\n"
                "MUL base matrix length other length\n"
                "Multiplies the first matrix by the other matrix in the given base.\n\n"
                "NULL matrix length base message\n"
                "Given a matrix with specified length and base, encodes the message with \"_\" at the parity bits.\n\n"
                "ENTROPY elem1 elem2 elem3 elem4 numerator1 numerator2 denominator\n"
                "Gives Markov entropy for the given transition matrix and equilibrium distribution.\n\n"
                "POLYNOMIAL fieldsize terms\n"
                "Generates terms until = 1\n\n"
                "POWER base exponent mod\n"
                "Simplify that kinda thing lol idk\n\n"
                "QUIT\n"
                "Exits the interface.\n\n"
                "SF radix numerators denominator\n"
                "Given probability, prints Shannon-Fano length.\n\n"
                "SPB message errors radix length\n"
                "Given three parameters and one nonnumeric unknown, solves for the unknown.\n\n"
                "TOTIENT int\n"
                "Gives totient\n\n"
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

