from typing import List
import math

class Misc:
    @staticmethod
    def get_help_message():
        return ("CODEWORD string\n"
                "Given a string, generates the ASCII character that provides the parity check for the string with the binary representation.\n\n"
                "HAMMING CHECK parity_matrix codeword\n"
                "Given the parity matrix (any linear matrix, not necessarily a Hamming matrix) and a codeword, checks the syndrome of the codeword and recommends a bit to be corrected if one error is present.\n\n"
                "HAMMING ENCODE parity_matrix message length\n"
                "Given the parity matrix (any linear matrix, not necessarily a Hamming matrix), a message, and the length of the codeword, generates the codeword.\n\n"
                "HELP\n"
                "Prints this message to interface.\n\n"
                "ISBN isbn\n"
                "Given a list of ISBNs, calculates the syndrome for each, or calculates a missing character signified with \"_\".\n\n"
                "QUIT\n"
                "Exits the interface.\n\n"
                "SPB length errors\n"
                "Given the length of a code and its error correction capability, calculates the maximum value of |C|.\n\n")

    # Solves for one unknown in the sphere packing theorem
    # errors = minimum distance - 1 // 2
    @staticmethod
    def sphere_packing_bounds(input: List[str]):
        if len(input) < 2:
            return "Sphere Packing Bounds calculation requires arg1 = length and arg2 = errors."

        if any(not var.isnumeric() for var in input[0:3]):
            return "Sphere Packing Bounds calculation requires numeric arguments."

        length = int(input[0])
        errors = int(input[1])

        sum_comb = sum(math.comb(length, i) for i in range(errors + 1))
        len_pow = 2 ** length
        
        code_len = len_pow / sum_comb

        return "Maximum code length is " + f"{code_len:.4f}" + ".\n"
