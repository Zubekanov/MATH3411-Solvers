from ascii import ASCII
from comma import Comma
from enum import Enum
from isbn import ISBN
from matrix import *
from misc import *

class Command(Enum):
    BASIS           = "BASIS"       .upper()
    CODEWORD        = "CODEWORD"    .upper()
    CORRECT         = "CORRECT"     .upper()
    CHECK           = "CHECK"       .upper()
    DECODABLE       = "DECODABLE"   .upper()
    EXTENSION       = "EXTENSION"   .upper()
    HELP            = "HELP"        .upper()
    HUFFMAN         = "HUFFMAN"     .upper()
    ISBN            = "ISBN"        .upper()
    KMT             = "KMT"         .upper()
    MUL             = "MUL"         .upper()
    NULL            = "NULL"        .upper()
    QUIT            = "QUIT"        .upper()
    SPHERE_PACKING  = "SPB"         .upper()
    SYMBOLS         = "SYMBOLS"     .upper()

class Interface:
    exit_flag = False

    def print(self, text: str):
        text_tokens = text.split("\n")
        for text_token in text_tokens:
            print("\t" + text_token)

    def await_input(self):
        return input("> ")
    
    def parse_input(self, query: str):
        query_tokens = query.split(" ")
        query_tokens = [string for string in query_tokens if string != ""]

        if len(query_tokens) == 0:
            return
        
        query_head = query_tokens[0]
        match query_head.upper():
            case Command.BASIS.value:
                self.print(MatrixSolver.find_span_vector(Matrix(query_tokens[1], length=int(query_tokens[2]), base=int(query_tokens[3])) , query_tokens[4]))
                return

            case Command.CODEWORD.value:
                self.print(ASCII.get_codeword(query_tokens[1:]))
                return
            
            case Command.CORRECT.value:
                self.print(ASCII.correct_error(query_tokens[1:]))
                return
            
            case Command.CHECK.value:
                self.print(MatrixSolver.check_codeword(query_tokens[1:]))
                return
            
            case Command.DECODABLE.value:
                self.print(Comma.decode(query_tokens[1:]))
                return
            
            case Command.EXTENSION.value:
                self.print(Comma.extension_length(query_tokens[1:]))
                return

            case Command.HELP.value:
                self.print(Misc.get_help_message())
                return
            
            case Command.HUFFMAN.value:
                self.print(Comma.huffman_average_length(query_tokens[1:]))
                return

            case Command.ISBN.value:
                self.print(ISBN.main(query_tokens[1:]))
                return

            case Command.KMT.value:
                self.print(Comma.kraftmcmillan(query_tokens[1:]))
                return

            case Command.NULL.value:
                self.print(MatrixSolver.find_null_vector(Matrix(query_tokens[1], length=int(query_tokens[2]), base=int(query_tokens[3])) , query_tokens[4]))
                return

            case Command.QUIT.value:
                self.exit_flag = True
                return
            
            case Command.SPHERE_PACKING.value:
                self.print(Misc.sphere_packing_bounds(query_tokens[1:]))
                return

            case _:
                self.print("\"" + query_head + "\" is not a supported command.")

if __name__ == "__main__":
    interface = Interface()
    try:
        while not interface.exit_flag:
            user_input = interface.await_input()
            interface.parse_input(user_input)
    except KeyboardInterrupt:
        pass

    interface.print("Exiting interface.")