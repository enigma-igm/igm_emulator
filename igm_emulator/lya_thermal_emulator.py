def main():
    redshift = input("Emulate at Redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]): ")
    small_scale = input("Less velocity bins of Ly-a forest ([True, False]): ")
    n_testing = int(input("Extra testing data number: "))

    return redshift, small_scale, n_testing

redshift, small_scale, n_testing = main()

#if __name__ == "__main__":

