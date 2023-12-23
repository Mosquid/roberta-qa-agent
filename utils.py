def flatten_json(y):
    out = {}

    def flatten(x, name=""):
        if type(x) is dict:
            for a in x:
                flatten(x[a], f"{name}{a}.")
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, f"{name}{i}.")
                i += 1
        else:
            out[str.lower(name[:-1])] = str.lower(str(x))

    flatten(y)
    return out
