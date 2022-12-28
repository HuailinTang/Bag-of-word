"""
COMP 614
Homework 5: Bag of Words
"""

import numpy
import re
import string
import comp614_module5


def get_title_and_text(filename):
    """
    Given a the name of an XML file, extracts and returns the strings contained 
    between the <title></title> and <text></text> tags.
    """
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
        # read title
        title_pattern = re.compile("<title>(.*)</title>")
        title_search = title_pattern.search(content)
        title = title_search.group(1)
        # read text. text might span multiple lines, use \n
        text_pattern = re.compile("<text.*?>((.|\n)*)</text>")
        text_search = text_pattern.search(content)
        text = text_search.group(1).replace("\n", ' ')
    return title, text


def get_words(text):
    """
    Given the full text of an XML file, filters out the non-body text (text that
    is contained within {{}}, [[]], [], <>, etc.) and punctuation and returns a 
    list of the remaining words, each of which should be converted to lowercase.
    """
    # filter non-body text
    pattern1 = re.compile("({{.*?}})|({\\|.*?\\|})|(\\[\\[.*?\\]\\])|(\\[.*?\\])|"
                          "(<.*?>)|(&lt;.*?&gt;)|(File:)")
    text_replace = pattern1.sub(" ", text)
    # remove punctuation
    punctuation_to_remove = "[" + string.punctuation + "](?![st]\\s)"
    pattern2 = re.compile(punctuation_to_remove)
    text_replace = pattern2.sub(" ", text_replace)
    # switch to lowercase
    text_list = text_replace.lower().split()
    return text_list


def count_words(words):
    """
    Given a list of words, returns the total number of words as well as a 
    dictionary mapping each unique word to its frequency of occurrence.
    """
    length = len(words)
    word_dict = {}
    for word in words:
        # if word not in word_dict, add it and set the frequency to 1
        if word not in word_dict.keys():
            word_dict[word] = 1
        # if word already in word_dict, increase its frequency by 1
        else:
            word_dict[word] += 1
    return length, word_dict


def count_all_words(filenames):
    """
    Given a list of filenames, returns three things. First, a list of the titles,
    where the i-th title corresponds to the i-th input filename. Second, a
    dictionary mapping each filename to an inner dictionary mapping each unique
    word in that file to its relative frequency of occurrence. Last, a dictionary 
    mapping each unique word --- including all words found across all files --- 
    to its total frequency of occurrence across all of the input files.
    """
    all_titles = []
    title_to_counter = {}
    total_counts = {}

    for file in filenames:
        title_text = get_title_and_text(file)
        title, text = title_text[0], title_text[1]
        # add title to add_titles
        all_titles.append(title)

        texts = get_words(text)
        # for each title, use count_words() to get text length and frequency of each word
        text_length, text_dict = count_words(texts)[0], count_words(texts)[1]

        for key in text_dict:
            # first add raw frequency to total_counts
            if key not in total_counts.keys():
                total_counts[key] = text_dict[key]
            else:
                total_counts[key] += text_dict[key]
            # change to relative
            # divide frequency of each word by text length to get relative frequency of each word
            text_dict[key] = text_dict[key] / text_length
        title_to_counter[title] = text_dict

    return all_titles, title_to_counter, total_counts


def encode_word_counts(all_titles, title_to_counter, total_counts, num_words):
    """
    Given two dictionaries in the format output by count_all_words and an integer
    num_words representing the number of top words to encode, finds the top 
    num_words words in total_counts and builds a matrix where the element in 
    position (i, j) is the relative frequency of occurrence of the j-th most 
    common overall word in the i-th article (i.e., the article corresponding to 
    the i-th title in titles).
    """
    lst = []
    sorted_words = sorted(total_counts.items(), key=lambda tup: (-1 * tup[1], tup[0]))

    if num_words >= len(sorted_words):
        num_column = len(sorted_words)
    elif num_words == 0:
        return numpy.empty((0, 0))
    else:
        num_column = num_words

    # set up an empty numpy matrix first, with 0 row and num_column column
    final_array = numpy.empty((0, num_column))
    # take top num_column in sorted_words as a list
    for word_num in sorted_words[0:num_column]:
        lst.append(word_num[0])

    # value is relative word frequency for each title
    for value in title_to_counter.values():
        temp_lst = []
        for element in lst:
            # if word in sorted_words appear in relative word frequency for this title
            if element in value.keys():
                # add relative frequency to temp_list
                temp_lst.append(value[element])
                # not appear, append 0
            else:
                temp_lst.append(0)
        # turn temp_lst to numpy array and append to empty numpy matrix
        array1 = numpy.array([temp_lst])
        final_array = numpy.append(final_array, array1, axis=0)

    return final_array


def nearest_neighbors(matrix, all_titles, title, num_nbrs):
    """
    Given a matrix, a list of all titles whose data is encoded in the matrix, such
    that the i-th title corresponds to the i-th row, a single title whose data is
    encoded in the matrix, and the desired number of neighbors to be found, finds 
    and returns the closest neighbors to the article with the given title.
    """
    ind = all_titles.index(title)
    # get the title's row
    num = matrix[ind]
    dist_lst = []
    for row in matrix:
        pos = 0
        dist = 0
        # calculate distance between each row and title's row
        while pos < len(matrix[ind]):
            dist += (num[pos] - row[pos]) ** 2
            pos += 1
        dist_lst.append(dist ** (1 / 2))
    # sort the titles based on the distance
    sort_titles = [all_titles for _, all_titles in sorted(zip(dist_lst, all_titles))]
    # don't return index 0 because it is title itself (always distance 0)
    return sort_titles[1:num_nbrs + 1]


def run():
    """
    Encodes the wikipedia dataset into a matrix, prompts the user to choose an
    article, and then runs the knn algorithm to find the 5 nearest neighbors
    of the chosen article.
    """
    # Encode the wikipedia dataset in a matrix
    filenames = comp614_module5.ALL_FILES
    all_titles, title_to_counter, total_counts = count_all_words(filenames)
    mat = encode_word_counts(all_titles, title_to_counter, total_counts, 20000)

    # Print all articles
    print("Enter the integer corresponding to the article whose nearest" +
          " neighbors you would like to find. Your options are:")
    for idx in range(len(all_titles)):
        print("\t" + str(idx) + ". " + all_titles[idx])

    # Prompt the user to choose an article
    while True:
        choice = input("Enter your choice here: ")
        try:
            choice = int(choice)
            break
        except ValueError:
            print("Error: you must enter an integer between 0 and " +
                  str(len(all_titles) - 1) + ", inclusive.")

    # Compute and print the results
    nbrs = nearest_neighbors(mat, all_titles, all_titles[choice], 5)
    print("\nThe 5 nearest neighbors of " + all_titles[choice] + " are:")
    for nbr in nbrs:
        print("\t" + nbr)

# Leave the following line commented when you submit your code to OwlTest/CanvasTest,
# but uncomment it to perform the analysis for the discussion questions.
run()
