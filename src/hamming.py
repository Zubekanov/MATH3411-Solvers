from enum import Enum
from typing import List

class Command(Enum):
    CHECK = "CHECK".upper()
    ENCODE = "ENCODE".upper()
    MINIMUM_DISTANCE = "MINDIST".upper()

class Hamming:
    @staticmethod
    def main(input: List[str]):
        result = ""

        input_head = input[0]
        match input_head.upper():
            case Command.CHECK.value:
                return Hamming.check(input[1:])
                
            case Command.ENCODE.value:
                return Hamming.encode(input[1:])
            
            case Command.MINIMUM_DISTANCE.value:
                return Hamming.get_minimum_distance(input[1:])

            case _:
                result = "\"" + input_head + "\" is not supported for Hamming codes."
                return result
    
    @staticmethod
    def check(input: List[str]):
        hamming = [int(char) for char in input[0]]
        codeword = [int(char) for char in input[1]]

        length = len(codeword)

        if len(hamming) % length != 0:
            return "Parity check matrix length (" + str(len(hamming)) + ") is not divisible by codeword length (" + str(length) + ").\n"
        
        # Reshape to 2d array
        hamming = Hamming.to2d(hamming, length)

        syndrome = []
        one_error_pos = 0
        
        for row in hamming:
            elem = 0
            for pos in range(length):
                elem += row[pos] * codeword[pos]
                elem %= 2
            one_error_pos *= 2
            one_error_pos += elem
            syndrome.append(elem)
        
        result = "Syndrome for " + str(codeword) + " is " + str(syndrome) + ".\n"

        if one_error_pos == 0:
            result += "Zero syndrome indicates that there are either zero or more than two errors."
        else:
            if Hamming.check_is_hamming(input[0], length):
                result += "This is a Hamming matrix.\n"
                result += "If there is only one error it is at position " + str(one_error_pos) + ", giving corrected codeword "
                codeword[one_error_pos - 1] += 1
                codeword[one_error_pos - 1] %= 2
                result += str(codeword) + ".\n"
            else:
                result += "This is not a Hamming matrix.\n"
                col_matrix = Hamming.get_col_matrix(input[0], length)
                try:
                    one_error_pos = col_matrix.index("".join([str(num) for num in syndrome])) + 1
                    codeword[one_error_pos - 1] += 1
                    codeword[one_error_pos - 1] %= 2
                    result += "If there is only one error it is at position " + str(one_error_pos) + ", giving corrected codeword " + str(codeword) + ".\n"
                except ValueError:
                    result += "No scalar multiple of the syndrome exists in the matrix, so there are two or more errors.\n"
        
        return result

    @staticmethod
    def encode(input: List[str]):
        hamming = [int(char) for char in input[0]]
        message = [int(char) for char in input[1]]
        length = input[2]

        if not length.isnumeric():
            return "Argument 4 must be an integer length.\n"
        
        length = int(length)

        if len(hamming) % length != 0:
            return "Parity check matrix length (" + str(len(hamming)) + ") is not divisible by codeword length (" + str(length) + ").\n"

        # Reshape to 2d array and use numpy for final processing so i dont have to do it manually
        # RAAAGH NUMPY SUCKS I HAVE TO DO THIS MANUALLY
        hamming = Hamming.to2d(hamming, length)
        leading_cols = Hamming.get_leading_cols(hamming)

        codeword = []
        non_leading_cols = []

        message_pos = 0
        for pos in range(length):
            if pos in leading_cols:
                codeword.append(None)
            else:
                codeword.append(message[message_pos])
                non_leading_cols.append(pos)
                message_pos += 1

        inv_row = 0
        for parity_bit in leading_cols[::-1]:
            row = hamming[::-1][inv_row]
            inv_row += 1

            value = 0
            pos = 0
            for bit in codeword[::-1]:
                if bit is not None:
                    value += bit * row[::-1][pos]
                    value %= 2
                    pos += 1
                else:
                    # we already know the hamming value at this is 1 otherwise it wouldnt be a leading row
                    codeword[parity_bit] = value
                    break

        return "Generated Hamming codeword as " + str(codeword) + "\n"

    @staticmethod
    def get_minimum_distance(input: List[str]):
        hamming = Hamming.to2d(input[0], int(input[1]))

        min_dist = 9999999
        row_orig = 0
        for row in hamming:
            for comp_index in range(row_orig + 1, len(hamming)):
                comp_row = hamming[comp_index]
                curr_dist = 0
                for index in range(len(row)):
                    if row[index] != comp_row[index]:
                        curr_dist += 1
            if curr_dist < min_dist: min_dist = curr_dist
            row_orig += 1
        
        return "Minimum distance is " + str(min_dist) + ", correctable errors is " + str((min_dist - 1) // 2) + ".\n"

    # Generating all keywords
    @staticmethod
    def get_min_weight():
        pass

    @staticmethod
    def get_col_matrix(matrix, length):
        height = len(matrix) // length
        return [matrix[i:i + height] for i in range(0, len(matrix), height)]

    @staticmethod
    def check_is_hamming(matrix, length):
        matrix = Hamming.get_col_matrix(matrix, length)
        comparison_matrix = [format(num, "03b") for num in range(1, length + 1)]
        return matrix == comparison_matrix

    @staticmethod
    def to2d(matrix, length):
        matrix = [int(char) for char in matrix]
        height = len(matrix) // length
        result = [[None] * length for _ in range(height)]
        for index, value in enumerate(matrix):
            row = index % height
            col = index // height
            result[row][col] = value
        return result
    
    @staticmethod
    def get_leading_cols(matrix):
        leading_col_indices = []

        for row in matrix:
            for pos, value in enumerate(row):
                if value != 0:
                    leading_col_indices.append(pos)
                    break

        return sorted(set(leading_col_indices))
    