
rank = None

def log(str, end="\n"):
    strOut = f"{rank}: {str}"
    f = open("log.txt", "a")
    f.write(strOut + end)
    f.close()
    print(strOut, end=end)