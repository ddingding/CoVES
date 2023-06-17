def viz_compare_seqs(seq1, seq2, printout=True, color=False):
    '''visually compare sequences'''
    match = ''
    diff = ''

    for s1, s2 in zip(seq1, seq2):
        if s1 == s2:
            match += '|'
            diff += ' '
        elif s1.upper()==s2.upper():
            match += ':'
            diff += ' '
        else:
            match += ' '
            diff += '*'
            
    txt_view = viz_line_len([seq1, match, seq2, diff])
    if printout:
        print(txt_view)
    return txt_view

    
def viz_line_len(my_strings, N=75):
    '''split lengthy aligned content into conesecutive chunks'''
    split_strings = []
    for string in my_strings:
        split_strings.append([])
        for i in range(0, len(string), N):
            split_strings[-1].append(string[i:i+N])
        
    txt_view = []
    for short_strs in zip(*split_strings):

        for s in short_strs:
            txt_view.append(s+' '*(N-len(s)))
        
        txt_view.append('')
        txt_view.append('')

    return '\n'.join(txt_view)