with open('dd.txt', 'r') as f:
    tb = {}
    Lines = f.readlines()
    for line in Lines:
        lst = line.split(' ')
        if int(lst[0]) in tb:
            if lst[1][0] == 's':
                tb[int(lst[0])] = 1 + tb[int(lst[0])]
            else:
                tb[int(lst[0])] = -1 + tb[int(lst[0])]
        else:
            if lst[1][0] == 's':
                tb[int(lst[0])] = 1
            else:
                tb[int(lst[0])] = -1
    for k, v in tb.items():
        if v != 0:
            print(k, v)

