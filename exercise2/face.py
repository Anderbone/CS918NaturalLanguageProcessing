import re
with open('emotion.txt', 'r') as f:
    P = []
    N = []
    for line in f:
        # print(line)
        face = line.strip().split(' ')
        # print(face)
        if face[-1] == 'Positive':
            P.extend(face[:-1])
            # print(face[:-1])
        if face[-1] == 'Negative':
            N.extend(face[:-1])
            # print(face[:-1])
    # print(P)
    newP = []
    for i in P:
        new = re.escape(i)
        newP.append(new)
    # print(newP)

    print(N)
    newN = []
    for i in N:
        new = re.escape(i)
        newN.append(new)
    # print(newN)
    # P = ' '.join(P)
    # # print(P)
    # N = ' '.join(N)
    # new = []
    # for i in P.split():
    #     i = re.escape(i)
    #     new.append(i)
    # print(new)

    # P = re.escape(P)
    # N = re.escape(N)

    P = ')|('.join(newP)
    N = ')|('.join(newN)
    print(P)
    print('--------')
    print(N)

    Pdic = {}
    # # Ndic = {}
    for i in P:
        Pdic[i] = 'HAPPYFACE'
    for i in N:
        Pdic[i] = 'SADFACE'
    # # Face = Pdic.update(Ndic)
    with open('myemo.txt', 'w') as f1:
        print(Pdic, file=f1)
    #     # print(Ndic, file=f1)
    #     print(Pdic, file=f1)



