from typing import List

class ISBN:
    # Input will be an array from interface.
    @staticmethod
    def main(input: List[str]):
        result = ""

        arg_pos = 1
        for arg in input:
            syndrome = 0
            unknown_pos = -1
            loop_pos = 1

            isbn = []
            for char in arg:
                if char.isdigit() or char == "_":
                    isbn.append(char)
                if char.upper() == "X":
                    isbn.append("10")

            if len(isbn) != 10:
                result += "".join(isbn) + " had " + str(len(isbn)) + " digits but expected 10.\n"
            else:
                for digit in isbn:
                    if digit.isnumeric():
                        syndrome += int(digit) * loop_pos
                    else:
                        if unknown_pos < 0:
                            unknown_pos = loop_pos
                        else:
                            result += "Multiple unknown values in " + "".join(isbn) + "\n"
                            break

                    loop_pos += 1
                
                if unknown_pos < 0:
                    result += "Syndrome for " + "".join(isbn) + " is " + str(syndrome % 11) + "\n"
                else:
                    syndrome %= 11
                    # Not bothered to look through my number theory notes to do this the smart way,
                    # TODO: do this in a smarter way
                    unknown = 0
                    while syndrome != 0:
                        unknown += 1
                        syndrome += unknown_pos
                        syndrome %= 11
                    if unknown == 10:
                        unknown = "X"
                    isbn[unknown_pos - 1] = str(unknown)

                    result += "Unknown digit is " + str(unknown) + ", giving ISBN " + "".join(isbn) + "\n"

            arg_pos += 1
        
        return result