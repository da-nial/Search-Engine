import matplotlib.pyplot as plt
import math
import seaborn as sns

sns.set()


def plot_zipf(tokens_info):
    max_number_with_stop_words = list(tokens_info.values())[0]
    max_number_without_stop_words = list(sorted_descending_word_freq_dict_without_stop_words.values())[0]
    max_number_without_high_freq = list(sorted_descending_word_freq_dict_without_high_freq.values())[0]
    print("max_number_with_stop_words: ", max_number_with_stop_words)
    print("max_number_without_stop_words: ", max_number_without_stop_words)
    print("max_number_without_high_freq: ", max_number_without_high_freq)

    # when have stop words
    L1, L2, L3 = [], [], []

    for word, freq in sorted_descending_word_freq_dict_with_stop_words.items():
        L3.append(math.log(freq, 10))
        word_index = list(sorted_descending_word_freq_dict_with_stop_words.keys()).index(word)
        L1.append(math.log(word_index + 1, 10))
        L2.append(math.log(max_number_with_stop_words / (word_index + 1), 10))

    plt.plot(L1, L2)
    plt.plot(L1, L3)
    plt.xlabel("Log 10 Rank")
    plt.ylabel("Log 10 cf")
    plt.title("With stop words")
    plt.show()

    # when remove stop words
    L4, L5, L6 = [], [], []
    for word, freq in sorted_descending_word_freq_dict_without_stop_words.items():
        L6.append(math.log(freq, 10))
        word_index = list(sorted_descending_word_freq_dict_without_stop_words.keys()).index(word)
        L4.append(math.log(word_index + 1, 10))
        L5.append(math.log(max_number_without_stop_words / (word_index + 1), 10))

    plt.plot(L4, L5)
    plt.plot(L4, L6)
    plt.xlabel("Log 10 Rank")
    plt.ylabel("Log 10 cf")
    plt.title("Without stop words")
    plt.show()

    # when remove high freq words
    L7, L8, L9 = [], [], []
    for word, freq in sorted_descending_word_freq_dict_without_high_freq.items():
        L9.append(math.log(freq, 10))
        word_index = list(sorted_descending_word_freq_dict_without_high_freq.keys()).index(word)
        L7.append(math.log(word_index + 1, 10))
        L8.append(math.log(max_number_without_high_freq / (word_index + 1), 10))

    plt.plot(L7, L8)
    plt.plot(L7, L9)
    plt.xlabel("Log 10 Rank")
    plt.ylabel("Log 10 cf")
    plt.title("Without high freq words")
    plt.show()
