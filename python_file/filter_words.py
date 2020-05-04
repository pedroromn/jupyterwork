"""
We provide you with the following beginning of a famous children song.
Print out only words that start with "o", ignoring case::

    My Bonnie lies over the ocean.
    My Bonnie lies over the sea.
    My Bonnie lies over the ocean.
    Oh bring back my Bonnie to me.
             

Bonus: Print out words only once.
"""

lyrics = """ My Bonnie lies over the ocean.
My Bonnie lies over the sea.
My Bonnie lies over the ocean.
Oh bring back my Bonnie to me."""

lyrics = lyrics.replace('.',"")
lyrics = lyrics.replace(',',"")


def get_o_words ():
    words = lyrics.split()
    # Your code here
    o_words = [] # Your code replaces the "[ ]" here.
    return o_words


unique_o_words = [] # Your code replaces the "[ ]" here.
print("unique words that start with 'o':")
print(unique_o_words)

o_word_frequencies = []  # your code replaces the "[ ]" here
print("unique words that start with 'o':")
print(o_word_frequencies)


if __name__ == '__main__':
    o_words = get_o_words()
    print("words that start with 'o':")
    print(o_words)

