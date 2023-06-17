# my tools

def hamming(str1, str2):
    assert len(str1) == len(str2)
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def complement(inbase):
    cDic = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return cDic[inbase]

def rev_complement(instr):
    compl = [complement(c) for c in instr]
    return "".join(compl[::-1])

def translate(inCodon):
    return CODON_TABLE[inCodon]
