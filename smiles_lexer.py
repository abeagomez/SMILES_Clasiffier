from ply import lex
from functools import reduce

# Cheminformatics
# A Guide to Cheminformatics
# By Metamolecular
# SMILES Formal Grammar v1.1
# SMILES Grammar in W3C EBNF Format
# /* Converted from PEG Grammar by http://bottlecaps.de/convert/  */
# SMILES   ::= Atom ( Chain | Branch )*
# Chain    ::= ( Bond? ( Atom | RingClosure ) )+
# Branch   ::= '(' Bond? SMILES+ ')'
# Atom     ::= OrganicSymbol
#            | AromaticSymbol
#            | AtomSpec
#            | WILDCARD
# Bond     ::= '-'
#            | '='
#            | '#'
#            | '$'
#            | ':'
#            | '/'
#            | '\'
#            | '.'
# AtomSpec ::= '[' Isotope? ( 'se' | 'as' | AromaticSymbol | ElementSymbol | WILDCARD ) ChiralClass? HCount? Charge? Class? ']'
# OrganicSymbol
#          ::= 'B' 'r'?
#            | 'C' 'l'?
#            | 'N'
#            | 'O'
#            | 'P'
#            | 'S'
#            | 'F'
#            | 'I'
# AromaticSymbol
#          ::= 'b'
#            | 'c'
#            | 'n'
#            | 'o'
#            | 'p'
#            | 's'
# WILDCARD ::= '*'
#
# <?TOKENS?>
#
# ElementSymbol
#          ::= [A-Z] [a-z]?
# RingClosure
#          ::= '%' [1-9] [0-9]
#            | [0-9]
# ChiralClass
#          ::= ( '@' ( '@' | 'TH' [1-2] | 'AL' [1-2] | 'SP' [1-3] | 'TB' ( '1' [0-9]? | '2' '0'? | [3-9] ) | 'OH' ( '1' [0-9]? | '2' [0-9]? | '3' '0'? | [4-9] ) )? )?
# Charge   ::= '-' ( '-' | '0' | '1' [0-5]? | [2-9] )?
#            | '+' ( '+' | '0' | '1' [0-5]? | [2-9] )?
# HCount   ::= 'H' [0-9]?
# Class    ::= ':' [0-9]+
# Isotope  ::= [1-9] [0-9]? [0-9]?



def union(*args):
    return reduce((lambda x, y: r'(' + x + r')' + r'|(' + y + r')'),args)

def concat(*args):
    return reduce((lambda x, y: r'(' + x + r')(' + y + r')'),args)

def optional(exp):
    return r'(' + exp + r')?'

tokens = (
    'BOND',
    'LPAREN',
    'RPAREN',
    'ATOM',
    'RINGCLOSURE'
    )

# Tokens
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_BOND    = r'[-=#$:/\\.]'
t_RINGCLOSURE = r'\%[1-9][0-9]|[0-9]'

# Atom
organic_symbol = r'Br?|Cl?|N|O|P|S|F|I'
aromatic_symbol = r'[bcnops]'
wildcard = r'\*'
isotope = r'([1-9][0-9]?[0-9]?)?'
element_symbol = r'([A-Z][a-z]?)?'
chiral_class = r'(@(@|TH[1-2]|AL[1-2]|SP[1-3]|TB(1[0-9]?|20?|[3-9])|OH(1[0-9]?|2[0-9]?|30?|[4-9]))?)?'
charge = r'(\-(\-|0|1[0-5]?|[2-9])?|\+(\+|0|1[0-5]?|[2-9])?)?'
h_count = r'(H[0-9]?)?'
mclass  = r'(:[0-9]+)?'
atom_spec = concat(r'\[', isotope, union(r'se', r'as', aromatic_symbol, element_symbol, wildcard), chiral_class, h_count, charge, mclass, r']')
t_ATOM    = union(organic_symbol, aromatic_symbol, atom_spec, wildcard)


def t_error(t):
    raise(Exception("Illegal character '%s'" % t.value[0]))

lexer = lex.lex()

def tokenize_smiles(smiles):
    """"":returns iterable of strings that form the smiles. each string is a token in the smiles grammar"""
    lexer.input(smiles)
    while True:
        token = lexer.token()
        if not token:
            break
        yield token.value


