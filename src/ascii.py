from typing import List

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