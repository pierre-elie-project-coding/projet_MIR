from data_process.read_and_plot import plot_one_data, read_data_from_text


def main():
    data = read_data_from_text(
        path_read_data="learning_test.fa",
        path_read_par="learning_test_parameters.txt",
        path_read_sol="learning_test_states.fa",
        stop=16,
    )
    print(f"See DF : \n {data.head()}")
    e1 = "c82b24c4-a3a7-4000-b2a6-95bd3815d150"
    e2 = "0659dd4c-cf20-4afd-8674-eb9e6769909d"
    plot_one_data(df=data, element=e2, with_time_profile=True)


if __name__ == "__main__":
    main()
