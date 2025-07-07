BRACKETS = ['()', '[]', '{}', '<>', 'Aa', 'Bb', 'Cc', 'Dd', 'Ee', 'Ff', 'Gg', 'Hh', 'Ii', 'Jj', 'Kk', 'Ll', 'Mm', 'Nn', 'Oo', 'Pp', 'Qq', 'Rr', 'Ss', 'Tt', 'Uu', 'Vv', 'Ww', 'Xx', 'Yy', 'Zz']

get_close_of = lambda open_br : {x[0]: x[1] for x in BRACKETS}[open_br]
get_open_of = lambda close_br : {x[1]: x[0] for x in BRACKETS}[close_br]
all_opens = [x[0] for x in BRACKETS]
all_closes = [x[1] for x in BRACKETS]

is_open = lambda br : br in all_opens
is_close = lambda br : br in all_closes


class DyckString():

    def __init__(self):
        self.s = ""
        self.stack = []

    def __str__(self):
        return self.s
    
    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, i):
        return self.s[i]
    
    def __eq__(self, other):
        return self.s == other.s

    def append_open(self, open_br):
        self.s += open_br
        self.stack.append(open_br)
    
    def append_close(self):
        self.s += get_close_of(self.stack.pop())
    
    def elevation(self):
        return len(self.stack)