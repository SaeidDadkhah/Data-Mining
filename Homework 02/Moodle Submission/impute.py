def imputation(row, t):
    zeros = []
    tmp = []
    for i in range(1, 7):
        if row.get_value(i, 0) == 0:
            zeros.append(i)
        else:
            tmp.append(t[i-1].cdf(row.get_value(i, 0)))
    if len(tmp) != 0:
        tmp = sum(tmp)/len(tmp)
        for i in zeros:
            res = t[i-1].ppf(tmp)
            res = min(res, 20)
            res = max(res, 0)
            row.set_value(i, value=res)
    else:
        for i in zeros:
            res = row.set_value(i, value=t[i-1].ppf(0.5))
