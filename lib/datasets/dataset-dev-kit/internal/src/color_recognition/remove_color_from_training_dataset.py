lines = []
with open("training.data", "r") as file_reader:
    lines = file_reader.readlines()

with open("training.data", "w") as file_writer:
    for line in lines:
        color = line.split(",")[3]
        print(color)
        if color != "Purple\n":
            file_writer.write(line)
