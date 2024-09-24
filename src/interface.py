from ascii import ASCII
from enum import Enum
from hamming import Hamming
from isbn import ISBN
from matrix import *
from misc import *

class Command(Enum):
    CODEWORD = "CODEWORD".upper()
    HAMMING = "HAMMING".upper()
    HELP = "HELP".upper()
    ISBN = "ISBN".upper()
    NULL = "NULL".upper()
    QUIT = "QUIT".upper()
    SPHERE_PACKING = "SPB".upper()

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

        if len(query_tokens) == 0:
            return
        
        query_head = query_tokens[0]
        match query_head.upper():
            case Command.CODEWORD.value:
                self.print(ASCII.get_codeword(query_tokens[1:]))
                return

            case Command.HAMMING.value:
                self.print(Hamming.main(query_tokens[1:]))
                return

            case Command.HELP.value:
                self.print(Misc.get_help_message())
                return

            case Command.ISBN.value:
                self.print(ISBN.main(query_tokens[1:]))
                return

            case Command.NULL.value:
                self.print(MatrixSolver.find_null_vector(Matrix(query_tokens[1], length=int(query_tokens[2]), base=int(query_tokens[3])) , query_tokens[4]))
                return

            case Command.QUIT.value:
                self.exit_flag = True
                return
            
            case Command.SPHERE_PACKING.value:
                self.print(Misc.sphere_packing_bounds(query_tokens[1:]))

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