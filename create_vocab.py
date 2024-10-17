rows = ["1", "2", "3", "4", "5", "6", "7", "8"]
columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
squares = [column + row for column in columns for row in rows]
pieces = ["r", "n", "b", "q", "k"]

with open("vocab.txt", "w") as f:
    for square in squares:
        f.write(f"{square}\n")
    for piece in pieces:
        f.write(f"{piece}\n")

                        

