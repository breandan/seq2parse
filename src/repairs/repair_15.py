def wife():
    pass

def family(nikte):
    def child():
        print(wife())
        print("Tabitha")
    return child

def nikte_new():
    print("Tabiket")

nikte_new = family(nikte_new)
