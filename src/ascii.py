from typing import List

CHECKED_ASCII_LEN = 8

class ASCII:
    @staticmethod
    def get_codeword(input: List[str]):
        codeword = list("00000000")
        words = "".join(input)
        
        for char in words:
            word_binary = list(format(ord(char), "08b"))
            for pos in range(8):
                if codeword[pos] == word_binary[pos]:
                    codeword[pos] = "0"
                else:
                    codeword[pos] = "1"

        codeword = "".join(codeword)
        codeword_ascii = chr(int(codeword, 2))
        return "Codeword is \"" + codeword_ascii + "\", with binary " + codeword + "\n"
    
    @staticmethod
    def correct_error(input: List[str]):
        if not all((char in "01" for char in word) for word in input):
            return "Error correction currently only supports binary representation.\n"
        
        if any(len(word) != CHECKED_ASCII_LEN for word in input):
            return "Incorrect ASCII length given."

        word_parity = ["0"] * len(input)
        col_parity = ["0"] * CHECKED_ASCII_LEN

        for word_pos in range(len(input)):
            for bit_pos in range(CHECKED_ASCII_LEN):
                word_parity[word_pos] = ("0" if input[word_pos][bit_pos] == word_parity[word_pos] else "1")
                col_parity[bit_pos] = ("0" if input[word_pos][bit_pos] == col_parity[bit_pos] else "1")

        word_error = word_parity.index("1")
        col_error = col_parity.index("1")

        correction = list(input[word_error])
        correction[col_error] = ("0" if input[word_error][col_error] == "1" else "1")
        input[word_error] = "".join(correction)
        # Pretty sure non-emoji ASCII has all zeros in pos 0.
        for word_pos in range(len(input)):
            list_str = list(input[word_pos])
            list_str[0] = "0"
            input[word_pos] = "".join(list_str)

        codeword = [chr(int(word, 2)) for word in input]

        return "Error detected at word " + str(word_error + 1) + " at pos " + str(col_error + 1) + ", with corrected word \"" + str("".join(codeword)) + "\".\n"