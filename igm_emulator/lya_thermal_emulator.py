def main():
    Redshift = input("Give me a Boolean: ")
    String = input("Give me a string: ")
    Number = int(input("Give me a number: "))

    if Boolean == "True":
        print('"{s}"\n{s}'.format(s=String))
    try:
        print('{}\n{}'.format(int(Number)))
    except ValueError as err:
        print('Error you did not give a number: {}'.format(err))

if __name__ == "__main__":
    main()