from enum import Enum

import sys

from ascii import ASCII
from isbn import ISBN
from hamming import Hamming

class Command(Enum):
    CODEWORD = "CODEWORD".upper()
    ISBN = "ISBN".upper()
    HAMMING = "HAMMING".upper()
    HELP = "HELP".upper()
    QUIT = "QUIT".upper()

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

            case Command.ISBN.value:
                self.print(ISBN.main(query_tokens[1:]))
                return

            case Command.HAMMING.value:
                self.print(Hamming.main(query_tokens[1:]))
                return

            case Command.HELP.value:
                self.print("TODO write help text")
                return

            case Command.QUIT.value:
                self.exit_flag = True
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