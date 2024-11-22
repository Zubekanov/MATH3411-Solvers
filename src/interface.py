from ascii import ASCII
from comma import Comma
from comma import BCHCode
from enum import Enum
from isbn import ISBN
from matrix import *
from misc import *
from numtheory import *
from poly import *

class Command(Enum):
    BASIS           = "BASIS"       .upper()
    CODEWORD        = "CODEWORD"    .upper()
    CORRECT         = "CORRECT"     .upper()
    COPRIME         = "COPRIME"     .upper()
    CHECK           = "CHECK"       .upper()
    DECODABLE       = "DECODABLE"   .upper()
    EXTENSION       = "EXTENSION"   .upper()
    FACTORS         = "FACTORS"     .upper()
    HELP            = "HELP"        .upper()
    HUFFMAN         = "HUFFMAN"     .upper()
    HUFFMANCONV     = "HUFFMANCONV" .upper()
    ISBN            = "ISBN"        .upper()
    INVERSE         = "INVERSE"     .upper()
    IOC             = "IOC"         .upper()
    KMT             = "KMT"         .upper()
    MUL             = "MUL"         .upper()
    NULL            = "NULL"        .upper()
    MARKOVENTROPY   = "ENTROPY"     .upper()
    MILLERRABIN     = "MR"          .upper()
    POLYNOMIAL      = "POLYNOMIAL"  .upper()
    POWER           = "POWER"       .upper()
    PRIMITIVES      = "PRIMITIVES"  .upper()
    QUIT            = "QUIT"        .upper()
    SHANNON_FANO    = "SF"          .upper()
    SHANNON_CODE    = "SFC"         .upper()
    SPHERE_PACKING  = "SPB"         .upper()
    SYMBOLS         = "SYMBOLS"     .upper()
    TOTIENT         = "TOTIENT"     .upper()
    V_DECRYPT       = "V_DECRYPT"   .upper()
    BCH_ENCODE      = "BCH_ENCODE"  .upper()
    BCH_DECODE      = "BCH_DECODE"  .upper()

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
            
            case Command.COPRIME.value:
                self.print(NumberTheory.interface_coprime(query_tokens[1:]))
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
            
            case Command.FACTORS.value:
                self.print(NumberTheory.interface_factors(query_tokens[1:]))
                return

            case Command.HELP.value:
                self.print(Misc.get_help_message())
                return
            
            case Command.HUFFMAN.value:
                self.print(Comma.huffman_average_length(query_tokens[1:]))
                return

            case Command.HUFFMANCONV.value:
                self.print(Comma.huffman_conv(query_tokens[1:]))
                return

            case Command.ISBN.value:
                self.print(ISBN.main(query_tokens[1:]))
                return

            case Command.INVERSE.value:
                self.print(NumberTheory.inverse(query_tokens[1:]))
                return

            case Command.IOC.value:
                Misc.analyze_and_decrypt_ciphertext("".join(query_tokens[1:]))
                return

            case Command.KMT.value:
                self.print(Comma.kraftmcmillan(query_tokens[1:]))
                return

            case Command.NULL.value:
                self.print(MatrixSolver.find_null_vector(Matrix(query_tokens[1], length=int(query_tokens[2]), base=int(query_tokens[3])) , query_tokens[4]))
                return

            case Command.MARKOVENTROPY.value:
                self.print(MatrixSolver.markov_entropy(Matrix(query_tokens[1:5], length = 2), Matrix([float(query_tokens[5]) / float(query_tokens[7]), float(query_tokens[6]) / float(query_tokens[7])], length = 1)))
                return

            case Command.MILLERRABIN.value:
                self.print(NumberTheory.miller_rabin(query_tokens[1:]))
                return

            case Command.POLYNOMIAL.value:
                self.print(generate_powers(int(query_tokens[1]), [int(item) for item in query_tokens[2:]]))
                return

            case Command.POWER.value:
                self.print(NumberTheory.powwow(query_tokens[1:]))
                return

            case Command.PRIMITIVES.value:
                self.print(NumberTheory.primitives(query_tokens[1:]))
                return

            case Command.QUIT.value:
                self.exit_flag = True
                return
            
            case Command.SHANNON_FANO.value:
                self.print(Comma.shannon_fano(query_tokens[1:]))
                return
            
            case Command.SHANNON_CODE.value:
                self.print(Comma.generate_shannon_fano_code(query_tokens[1:]))
                return
            
            case Command.SPHERE_PACKING.value:
                self.print(Misc.sphere_packing_bounds(query_tokens[1:]))
                return
            
            case Command.TOTIENT.value:
                self.print(NumberTheory.interface_totient(query_tokens[1:]))
                return

            case Command.V_DECRYPT.value:
                self.print(Misc.vigenere_decrypt("".join(query_tokens[1:-1]), query_tokens[-1]))

            case Command.BCH_ENCODE.value:
                # Handle BCH encoding
                result = BCHCode.bch_encode(query_tokens[1:])
                self.print("Encoded codeword polynomial coefficients: " + ','.join(map(str, result)))
                return

            case Command.BCH_DECODE.value:
                # Handle BCH decoding
                corrected_codeword, message = BCHCode.bch_decode(query_tokens[1:])
                self.print("Corrected codeword polynomial coefficients: " + ','.join(map(str, corrected_codeword)))
                self.print("Decoded message polynomial coefficients: " + ','.join(map(str, message)))
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