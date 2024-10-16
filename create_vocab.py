rows = ["1", "2", "3", "4", "5", "6", "7", "8"]
columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
pieces = ["r", "n", "b", "q", "k"]

with open("vocab.txt", "w") as f:
    for row in rows:
        for column in columns:
            for row_2 in rows:
                for column_2 in columns:
                    if not (row == row_2 and column == column_2):
                        f.write(f"{column}{row}{column_2}{row_2}\n")
                        if row_2 in ["1", "8"] and row != row_2:
                            for piece in pieces:
                                f.write(f"{column}{row}{column_2}{row_2}{piece}\n")

                        

