def generator():
    print("Función generator")
    i = 0
    while True:
        print("Dentro del bucle de la función")
        i += 1
        yield i


for item in generator():
    print(item)
    if item > 4:
        break
print("Adios")


for item in generator():
    print(item)
    break
