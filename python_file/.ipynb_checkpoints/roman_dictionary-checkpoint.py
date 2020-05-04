"""
Mark Antony keeps a list of the people he knows in several dictionaries 
based on their relationship to him::
    
    friends = {'julius': '100 via apian', 'cleopatra': '000 pyramid parkway'}
    romans = dict(brutus='234 via tratorium', cassius='111 aqueduct lane')
    countrymen = dict([('plebius','786 via bunius'), 
                       ('plebia', '786 via bunius')])


1. Print out the names for all of Antony's friends.
2. Now all of their addresses.
3. Now print them as "pairs".
4. Hmmm.  Something unfortunate befell Julius.  Remove him from the 
   friends list.
5. Antony needs to mail everyone for his second-triumvirate party.  Make
   a single dictionary containing everyone.
6. Antony's stopping over in Egypt and wants to swing by Cleopatra's 
   place while he is there. Get her address.
7. The barbarian hordes have invaded and destroyed all of Rome.
   Clear out everyone from the dictionary.
   
"""

friends = {'julius': '100 via apian', 'cleopatra': '000 pyramid parkway'}
romans = dict(brutus='234 via tratorium', cassius='111 aqueduct lane')
countrymen = dict([('plebius','786 via bunius'), ('plebia', '786 via bunius')])

if __name__ == '__main__':

    # Question One
    antony_friends = [] # Your code here.  Replace the empty list with
                        # the correct answer
    print(antony_friends)
    
    # Question Two
    friend_addresses = [] # Your code here.  Replace the empty list with
                          # the correct answer
    print(friend_addresses)

    # Question Three
    friend_address_pairs = [] # Your code here.  Replace the empty list with
                        # the correct answer
    print(friend_address_pairs)
    
    # Question Four
    # Your code here.  Remove julius


    # Question Five
    # Your code here. Create a single dictionary with everyone
    # This can be done with dictionary methods that merge dicts

    # Question Six
    cleos_address = '' # Your code here.  Replace the empty string with
                        # an expression that retrieves the correct answer
    print(cleos_address)
    
    # Question Seven
    # Your code here.  Remove everyone from the dictionary


