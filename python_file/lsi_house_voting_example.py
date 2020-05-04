"""


The data is United States Congressional Voting Records 1984, taken from

archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

Origin:

Congressional Quarterly Almanac,
98th Congress, 2nd session 1984, Volume XL:
Congressional Quarterly Inc. Washington, D.C., 1985.

It is also available in R as data included with the mlbench package.

In R::

> library(mlbench)
> data(HouseVotes84)

The HouseVotes84 data set includes votes for each of the U.S. House of
Representatives Congressmen on the 16 key votes identified by the
Congressional Quarterly Almanac (CQA).

The CQA contains 16 variables, and consider nine different types
of votes represented by three classes: yea (voted for, paired for,
announced for), nay (voted against, paired against, announced against)
and unknown (voted present, voted present to avoid conflict of
interest, did not vote or otherwise make a position known).

The bills voted on are
2(0). handicapped-infants: 2 (y,n)
3(1). water-project-cost-sharing: 2 (y,n)
4(2). adoption-of-the-budget-resolution: 2 (y,n)
5(3). physician-fee-freeze: 2 (y,n)
6(4). el-salvador-aid: 2 (y,n)   Dems opposed military aid to El Salvador. Repub cause
7(5). religious-groups-in-schools: 2 (y,n)
8(6). anti-satellite-test-ban: 2 (y,n)
9(7). aid-to-nicaraguan-contras: 2 (y,n)  Prohibited aid to contras, Big DemCause
10(8). mx-missile: 2 (y,n)
11(9). immigration: 2 (y,n)
12(10). synfuels-corporation-cutback: 2 (y,n)
13(11). education-spending: 2 (y,n)
14(12). superfund-right-to-sue: 2 (y,n)
15(13). crime: 2 (y,n)
16(14). duty-free-exports: 2 (y,n)
17(15). export-administration-act-south-africa: 2 (y,n)

For mapping this problem to the orginal LSA  doc retrieval example,
House members are docs, Votes are terms. So in place of a
term doc matrix, we have a House member/vote matrix.

For help with selecting a subset of the data using R's Selector package,
see
http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Dimensionality_Reduction/Feature_Selection

Fo more data of the smae type (more recent hopefully) look here:

http://web.mit.edu/17.251/www/data_page.html
"""


from numpy import *
#from svd_demo import make_sigma
from clustering.lsi import make_k_space_term_reps,print_labeled_lexes_to_file,print_to_class_sorted_files

import read_data
import sys, os.path
from collections import Counter

vote_to_int = dict(n=-1,NA=0,y=1)
republican_support =Counter()
democratic_support = Counter()
support_dict = dict(democrat= democratic_support,
                   republican=republican_support)
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

## Hack to allow eval of "'NA'"
NA = 'NA'

## Order of these labels corresponds to order of cols in R_data
bills = ['handicapped-infants',
'water-project-cost-sharing',
'budget-resolution',
'physician-fee-freeze',
'el-salvador-aid',
'religious-groups-in-schools',
'anti-satellite-test-ban',
'aid-to-contras',
'mx-missile',
'immigration',
'synfuels-corporation-cutback',
'education-spending',
'superfund-right-to-sue',
'crime',
'duty-free-exports',
'export-act-south-africa']

def convert_house_data_to_ints (R_data,rows,cols): 
    zero_v = zeros(rows*cols,float)
    matrix = reshape(zero_v,(rows,cols))
    for (i,row) in enumerate(R_data):
        votes = [vote_to_int[eval(v)] for v in row[1:]]
        affiliation = eval(row[0])
        # Change the original row too.
        row[0] = affiliation
        for (j,v) in enumerate(votes):
            if v > 0:
                support_dict[affiliation][j] += 1
            matrix[i][j] = v
    return matrix


def convert_house_data_to_ints_constant_length (R_data,rows,cols): 
    zero_v = zeros(rows*cols*2,float)
    matrix = reshape(zero_v,(rows,cols*2))
    for (i,row) in enumerate(R_data):
        votes = [vote_to_int[eval(v)] for v in row[1:]]
        affiliation = eval(row[0])
        # Change the original row too.
        row[0] = affiliation
        for (j,v) in enumerate(votes):
            # -1 => 0, 0 + 1, 1 -> 2
            vote_val = v + 1
            if v > 0:
                support_dict[affiliation][j] += 1
            if v == 0:
                matrix[i][j*2] = 1.0
            else:
                matrix[i][j*2] = 0.0
            matrix[i][(j*2)+1] = v
    return matrix


def euclidean_normalize( matrix):
    sq_matrix = matrix**2
    for index in range(len(matrix)):
        row = sq_matrix[index,:]
        norm = sqrt(sum(row))
        matrix[index,:] = matrix[index,:]/norm

    deletes = []
    for row in range(len(matrix)):
        x = sum(matrix[row,:]**2)
        if x <> x:
            deletes.append(row)
        elif not allclose(x,1.0):
            print row, x

    (M,N) = matrix.shape
    new_M = M-len(deletes)
    new_size = new_M * N
    new_matrix = zeros(new_size,float)
    new_matrix = new_matrix.reshape((new_M,N))

    print new_matrix.shape
    ctr = 0
    for row in range(len(matrix)):
        if row in deletes:
            continue
        new_matrix[ctr] = matrix[row]
        ctr += 1
        

    print 'New length of matrix %d' % (len(new_matrix),)
    return new_matrix
        

def print_support_dict (support_dict,bills):
    parties = support_dict.keys()
    #width0 = max(map(lambda x: len(x[0]), bills))
    width0 = max(map(len, bills))
    width1 = len(parties[0])
    width2 = len(parties[1])
    banner = '{:<{width0}} {:<{width1}} {:<{width2}}'.format('Bill', parties[0].capitalize(), parties[1].capitalize(),width0=width0,width1=width1,width2=width2)
    print banner
    print '=' * len(banner)
    for bill in range(len(bills)):
        print '{:>2}. {:<{width0}} {:<{width1}} {:<{width2}}'.format(bill, bills[bill],
                                                              support_dict[parties[0]][bill],
                                                              support_dict[parties[1]][bill],
                                                              width0=width0,width1=width1,width2=width2)



def plot_class_sorted_data(matrix, cls_seq, color_list, k=2):
    """
    Nothing could simpler than plotting C{matrix}. the first col is the x-coord of all points
    The second col is the y-coord.  We want a scatter of points and we want to choose colors for
    them.

    Basic tool:

    plt.scatter(x, y, s=20, c='b', marker='o', cmap=None)
    s size,  c color, cmap color map
    
    cmap (color map) requires c (color) to be an array of floats and assigns colors to real num INTERVALS.
    We can use ints to represent levels as below.

    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
    """
    # list of 435 ints (0,1), 1 for repub, 0 for dem.

    (nrows,ncols) = matrix.shape
    # MPL could CHOOSE colors for us with this list of 0s and 1s in the same order as the points.
    # To make SURE repubs are RED and dems are BLUE we call helper func to make color map.
    # NB: levels arg is intended to help "quantize" continuous values into colors, so it
    #     must be a sequence of NUMBERS in increasing order. Levels and colors must be of same length.

    # Make repubs (=1) red, dems (=0) blue.
    cmap, norm = mpl.colors.from_levels_and_colors(levels=[0,1], colors=['b','r'], extend='max')

    if k == 2:

        fig, ax = plt.subplots(figsize=(5,5))

        ax.scatter(matrix[:,0],matrix[:,1], s=40, c=color_list, cmap=cmap, norm=norm, edgecolor='none')
        # Let mpl choose colors for you.
        # ax.scatter(matrix[:,0],matrix[:,1], s=40, c=color_m, marker='o', edgecolor='none')
    elif k == 3:
        fig = plt.figure()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(matrix[:,0],matrix[:,1], matrix[:,2], s=40, c=color_list, cmap=cmap, norm=norm, edgecolor='none')
    else:
        print '4D and higher graph display not yet implemented!'
        return
    plt.show()



def plot_labeled_data(matrix, labels, class_list, k=2):
    """
    http://matplotlib.org/1.3.1/users/text_intro.html
    """
    (nrows,ncols) = matrix.shape

    # Make repubs (=1) red, dems (=0) blue.
    cmap, norm = mpl.colors.from_levels_and_colors(levels=[0,1], colors=['b','r'], extend='max')
    
    if k == 2:

        fig,ax = plt.subplots(figsize=(10,10))
        ax.scatter(matrix[:,0],matrix[:,1], c=class_list, s=30, cmap=cmap, norm = norm, edgecolor='none')

        for r in range(nrows):
            # Just have to reposition these two to be over their points
            if labels[r].startswith('el') or labels[r].startswith('aid'):
                al = 'bottom'
            else:
                al = 'top'
            ax.text(matrix[r][0],matrix[r][1],labels[r],
                    verticalalignment=al,horizontalalignment='center'
                    )
    elif k == 3:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #fig,ax = plt.subplots(figsize=(10,10))
        ax.scatter(matrix[:,0],matrix[:,1], matrix[:,2],c=class_list, s=30, cmap=cmap, norm = norm, edgecolor='none')

        for r in range(nrows):
            # Not clear this is still needed in 3D image, but doesnt seem to hurt.
            if labels[r].startswith('el') or labels[r].startswith('aid'):
                al = 'bottom'
            else:
                al = 'top'
            ax.text(matrix[r][0],matrix[r][1],matrix[r][2], labels[r],
                    verticalalignment=al,horizontalalignment='center'
                    )
    else:
        print 'Unimplemented dimensions!'
        return
    plt.show()
                    


if __name__ == '__main__':

    ######################################################################
    ######################################################################
    #
    #  Data location and parameters
    #
    ######################################################################
    ######################################################################
    
    home = os.getenv('HOME')
    data_dir = os.path.join(home,'ext','src','Rdata')
    R_data_file = os.path.join(data_dir, 'HouseVotes84.dat')
    #euc_norm = True
    euc_norm = False
    if euc_norm == True:
        norm_str = '_euc_norm'
    else:
        norm_str = ''
    ## the number of dimensions in our reduced rep.
    k = 2
    #k = 3

    ######################################################################
    ######################################################################
    #
    #  End Data location and parameters
    #
    ######################################################################
    ######################################################################
    
    (R_data, data_sums,row_labels,col_labels) = read_data.read_R_data_file(R_data_file, data_type=str)
    matrix = convert_house_data_to_ints (R_data,435,16)

    # Were interested in the reduced dimensionality rep of a member
    # (a row, in this data, as described in query_transformation_notes)

    # We will go from the original data matrix, C{matrix}, to a new representation C{member_reps}.
    # The original data matrix is a 435 x 16 matrix representing 435 house members' votes on 16 bills,
    # C{member_reps} is going to be a 435 x k matrix containing a k-dimensional representation
    # of each member.

    if euc_norm:
        # this doesnt work particularly well.  Not sure why....
        matrix = euclidean_normalize(matrix)

    # Q: Must the Eigenvalues in s always be positive?  That is, is
    # S = MM' (whose Eigenvalues are contained in s) positive definite?
    # Returning 2 or 3 D reps of both the bills and the house members
    # NB: member_function is the transform to get from a row of matrix to
    #     a row of member_reps.  That is:
    # member_function(matrix[i,:]) == member_reps[i,:]
    # So it can be used to make a k-dimensional rep  of a new House member.
    (member_reps,bill_reps, U,s,Vh, member_function) = make_k_space_term_reps (matrix, k)


    ####################################################################################################
    ####################################################################################################
    #
    #   Making a picture of the House Members
    #
    ####################################################################################################
    ####################################################################################################

    # Reflect matrix around the y axis to put Dems on left hand side, Repubs on right
    if k == 2:
        member_reps[:,0] *= -1

    # Get the list of parties found in the data. These are our "classes"
    cls_seq = list(set((r[0] for r in R_data)))
    # A function that given a row number, retrieves the party affiliation for that row of R_data
    # So for any member, it returns that member "class"
    isrepub = dict(republican=1,democrat=0)
    (nrows,cols) = member_reps.shape
    color_list = [isrepub[R_data[r][0]] for r in range(nrows)]
    # This plots the picture in matplotlib.
    plot_class_sorted_data(member_reps, cls_seq, color_list, k=k)

    #cls_func = lambda r: R_data[r][0]
    #print_to_class_sorted_files(member_reps, cls_seq, cls_func, k=k, data_dir = data_dir, suffix=norm_str)

    ####################################################################################################
    ####################################################################################################
    #
    #   Making a picture of the Bills
    #
    ####################################################################################################
    ####################################################################################################
    
    if k == 2:
        bill_reps[:,0] *= -1

    # To show what's in the support dict
    print_support_dict(support_dict,bills)

    # Now make a list bill_class s/t bill_class[i] == the party
    # that provided the majority of the support for bills[i],
    # where Repubs are represented by 1, and dems by 0.
    # To be used for coloring.
    
    bill_class = []
    for x in range(16):  # 16 bills
        if support_dict['republican'][x] > support_dict['democrat'][x]:
            bill_class.append(1)
        else:
            bill_class.append(0)    

    plot_labeled_data(bill_reps, bills, bill_class, k=k)


    # print_labeled_lexes_to_file(os.path.join(data_dir,'bills.dat'),bill_reps, 2, bills_tweaked,
    #                            y_distortion=1, swap=False)
